import logging
import warnings
from abc import ABC
from typing import Tuple, TYPE_CHECKING, Union

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from scipy.stats import differential_entropy as entr

from sim.params import ParamCollection
from sim.utils import (
    Array, JaxGaussian, JaxRKey, Number, Output, ParamEvolution, ParamIterator, ParamIteratorConfig
)

if TYPE_CHECKING:
    from sim.vehicle import Vehicle

logger = logging.getLogger("av-sim")


class ModelBase(ABC):
    """
    Base class for our models
    """

    def __init__(self, *args, **kwargs):
        # We leave kwargs open
        self.key = JaxRKey(seed=kwargs.get("seed", 8675309))


class Policy(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, env, params: ParamCollection):
        raise NotImplementedError


class DefaultPolicy(Policy):
    """
    This policy is the HighwayEnv default policy
    """

    def sample(self, env, params: ParamCollection):
        ...


class LossModel(ModelBase):
    def __init__(self, alpha: float, v_r: Number, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.v_r = v_r
        self.beta = beta

    def __call__(self, v: float, neighbors: Array) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, world_draws]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param v: speed of vehicle
        :param neighbors: num_neighbors x 3 array, columns are [distance, speed, lane]
        """
        speed_loss = - self.alpha * jnp.exp(-(v - self.v_r) ** 2 / self.alpha)
        collision_loss = self.beta * jnp.mean(
            1 / (2 ** neighbors[:, 2]) * jnp.divide(jnp.max(0, v - neighbors[:, 1]), neighbors[:, 0])
        )
        loss = speed_loss + collision_loss
        return loss, -10 * jnp.ones(loss.shape)  # Set log_probs to -10 to show high precision


class NoisyLossModel(LossModel):
    def __init__(self, scale: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def __call__(self, v: float, neighbors: Array) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, world_draws]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param v: speed of vehicle
        :param neighbors: num_neighbors x 3 array, columns are [distance, speed, lane]
        """
        loss, _ = super(NoisyLossModel, self).__call__(t)
        key = self.key.next_seed()
        jax_loss = jnp.asarray(loss)
        rloss, log_probs = JaxGaussian.sample(key, jax_loss, self.scale)
        return rloss, log_probs


class PreferencePrior(ModelBase):
    def __init__(self, *args, **kwargs):
        """
        Also store param histories as they evolve, keyed by their name
        """
        super().__init__(*args, **kwargs)
        self._param_history = {}

    def get_param_history(self):
        # Return param history if non-null
        return {
            k: np.array(v) for k, v in self._param_history.items() if len(v) > 0
        }

    def _init_param(self, p: Union[Number, ParamIteratorConfig]) -> ParamIterator:
        """
        Use this method to initialize you parameters as ParamIterators
        """
        if isinstance(p, ParamIterator):
            return p

        if isinstance(p, (dict, DictConfig)):
            return ParamIterator(**p)
        elif isinstance(p, (int, float)):
            return ParamIterator(evolution=ParamEvolution.CONSTANT, x_0=p)
        else:
            raise ValueError(f"Unrecognized parameter type: {p}")

    def step(self):
        """
        Evolve the preference prior
        Be sure to also store its last value
        """
        raise NotImplementedError


class ExponentialPreferencePrior(PreferencePrior):
    """
    k is an empirical constant related to stakeholder loss aversion
    k = -ln(p*)/L* where p* is the stakeholder's probability that loss will surpass L*
    """

    def __init__(
        self,
        p_star: Number,
        l_star: Union[ParamIterator, ParamIteratorConfig],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._param_history = {
            "p_star": [],
            "l_star": [],
            "k": [],
        }

        p_star_iter = self._init_param(p_star)
        l_star_iter = self._init_param(l_star)

        self.p_star_iter = p_star_iter
        self.l_star_iter = l_star_iter
        # Init value
        self.p_star = self.p_star_iter()
        self.l_star = self.l_star_iter()

    @property
    def k(self):
        return -jnp.log(self.p_star) / self.l_star

    def step(self):
        self._param_history["p_star"].append(self.p_star)
        self._param_history["l_star"].append(self.l_star)
        self._param_history["k"].append(self.k)

        self.p_star = self.p_star_iter()
        self.l_star = self.l_star_iter()

    def __call__(self, Lt: Array) -> Array:
        """
        Compute the exponential preference prior
        Returns an array of shape [m, world_draws]
        """
        return -self.k * Lt


class UniformPreferencePrior(PreferencePrior):
    def __init__(
        self,
        l_bar: Union[ParamIterator, ParamIteratorConfig],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._param_history["l_bar"] = []

        l_bar_iter = self._init_param(l_bar)
        self.l_bar_iter = l_bar_iter
        # Init value
        self.l_bar = l_bar_iter()

    def step(self):
        self._param_history["l_bar"].append(self.l_bar)
        self.l_bar = self.l_bar_iter()

    def __call__(self, Lt):
        """
        Compute the uniform preference prior
        Returns an array of shape [m, world_draws]
        """
        return np.zeros(Lt.shape) - np.log(self.l_bar)


class RiskModel(ModelBase):
    def __init__(self, preference_prior, min_risk: Number = 0, *args, **kwargs):
        """
        Risk model
        Takes a preference prior and optionally a minimum risk to be subtracted from values for normalization
        """
        super().__init__(*args, **kwargs)
        self.preference_prior = preference_prior
        self.min_risk = min_risk

    def compute_entropy(self, Lt, Lt_logprob, Vt):
        raise NotImplementedError

    def __call__(self, Lt: Array, Lt_logprob: Array, Vt: Array) -> Tuple[Array, Array, Array]:
        """
        Compute an array of risk values at a given timestep
        Arrays have shape [n_montecarlo, world_draws]
        """
        # this printing is important for evolving the preference model
        sample_mean = self.preference_prior(Lt).mean(axis=0)
        entropy = self.compute_entropy(Lt, Lt_logprob, Vt)
        Gt = - entropy - sample_mean
        # Gt = - sample_mean
        return Gt, entropy, sample_mean


class DifferentialEntropyRiskModel(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array, Vt: Array) -> Array:
        """
        Compute the differential entropy of the loss distribution
        Input Arrays have shape [n_montecarlo, world_draws] since this isn't called for real timesteps
        Return Array has shape [world_draws] since we reduce along the montecarlo axis=0
        """
        ent = entr(Lt, axis=0) if len(Lt) > 1 else 0.
        if np.any(ent == float('-inf')):
            warnings.warn("-inf encountered in entropy")
            # if Vt is 0, then just set the differential entropy to 0
            ent = np.where(np.logical_and(ent == -float('inf'), Vt == 0), 0, ent)
            # set arbitrarily to -10
            ent = np.where(ent == -float('inf'), -10, ent)
        return ent


class MonteCarloRiskModel(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array, Vt: Array) -> Array:
        """
        Compute the Monte Carlo estimate of the entropy of the loss distribution
        Input Arrays have shape [n_montecarlo, world_draws] since this isn't called for real timesteps
        Return Array has shape [world_draws] since we reduce along the montecarlo axis=0
        """
        return Lt_logprob.mean(axis=0)


class NullEntropy(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array, Vt: Array) -> Array:
        """
        Zero out the entropy term
        """
        return jnp.zeros(Lt.shape[1])


class AsyncWorldModel(ModelBase):
    """
    Fires off world_draws number of randomly initialized envs
    Within, egos use `n_montecarlo` predictive action samples
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        params: ParamCollection,
        world_draws: int,
        n_montecarlo: int,
        duration: int,
        plan_duration: int,
        ego: "Vehicle",
        loss_model,
        risk_model,
        seed=8675309,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.env_cfg = env_cfg
        self.params = params
        self.world_draws = world_draws
        self.n_montecarlo = n_montecarlo
        self.ego = ego
        self.duration = duration
        self.plan_duration = plan_duration
        self.loss_model = loss_model
        self.risk_model = risk_model
        self.seed = seed

    def sample_policy(self, params: ParamCollection) -> ParamCollection:
        raise NotImplementedError

    def timestep(
        self, t: int, old_params: ParamCollection, planning: bool, omega_iter: ParamIterator | None = None
    ) -> Tuple[Array, Array, Array, Array, ParamCollection]:
        """
        Shapes of variables (like Bt, Ct, ...) will be
        (m, world_draws)
        where m is either 1 for the "real" timestep or n_montecarlo for planning
        """
        raise NotImplementedError

    def plan(self, params: ParamCollection) -> Array:
        """
        Calculate the risk value at the present timestep and the current set of params
        by simulating across a planning horizon with n_montecarlo simulations
        Return shape is [world_draws]
        """
        Rt_sim = jnp.zeros(self.world_draws)
        ent = jnp.zeros(self.world_draws)
        extrinsic = jnp.zeros(self.world_draws)
        for t_plan in range(self.plan_duration):
            ...

        # ... = self.timestep(t_plan, params, planning=True)
        # Gt, entropy, sample_mean = self.risk_model(Lt, Lt_logprob, Vt)
        # Rt_sim += Gt
        # ent += entropy
        # extrinsic += sample_mean

        logger.debug(
            "plan last: -entropy.mean {} | -sample_mean.mean {} | risk.mean {}".format(
                -ent.mean(), -extrinsic.mean(), Rt_sim.mean()
            )
        )

        return Rt_sim

    def get_montecarlo_params(self):
        """
        Return a replicated stack of current params for running MC predictive simulations
        Returned object is a params tuple object where each param is an n_montecarlo x world_draws size
        The params are identical across n_montecarlo dimension, but differ across world_draws dimension
        """
        return self.params.monte_carlo_stack(self.n_montecarlo)

    def __call__(self, *args, **kwargs) -> Output:
        """
        We collect various values at each real timestep and store them.
        Collected values will have dimension either (1, world_draws) or (world_draws).
        However, these get squeezed in the Output object.
        The final [duration, num_param_batch] set of results will be passed into an Output object.
        """
        # For some reason these two values can't be passed directly into AsyncVectorEnv
        max_steps = int(self.duration)
        n_envs = int(self.world_draws)
        n_envs = 2
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    'highway-v0',
                    render_mode=None,
                    max_episode_steps=max_steps,
                    # **cfg.env
                ) for _ in range(n_envs)
            ]
        )

        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=self.world_draws)
        observations, info = envs_wrapper.reset(seed=self.seed)

        results = {
            "rewards": []
        }
        print(f"MW TYPE -- {envs_wrapper.unwrapped.action_space}")
        for step in range(self.duration):
            actions = envs.action_space.sample()
            # action = env.unwrapped.action_type.actions_indexes["IDLE"]
            # print(f"MW ACTIONS -- {actions}")
            observations, rewards, terminated, truncated, infos = envs_wrapper.step(actions)
            # print(f"MW SHAPES -- {observations.shape}, {rewards.shape}, {infos}")
            results['rewards'].append(rewards)

            # # select an action A_{t} using S_{t} as input for the agent
            # actions, action_log_probs, state_value_preds, entropy = agent.select_action(
            #     states
            # )
            #
            # # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            # states, rewards, terminated, truncated, infos = envs_wrapper.step(
            #     actions.cpu().numpy()
            # )
            #
            # ep_value_preds[step] = torch.squeeze(state_value_preds)
            # ep_rewards[step] = torch.tensor(rewards, device=device)
            # ep_action_log_probs[step] = action_log_probs
            #
            # # add a mask (for the return calculation later);
            # # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            # masks[step] = torch.tensor([not term for term in terminated])


        # print(f"MW RESULTS -- {results}")
        # Save the preference params and omega history
        param_history = {
            **self.risk_model.preference_prior.get_param_history()
        }

        return 0
        # return Output(...)
