import logging
import warnings
from abc import ABC
from copy import copy
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from scipy.stats import differential_entropy as entr

from sim.utils import (
    Array, JaxGaussian, JaxRKey, Number, Output, ParamEvolution, ParamIterator, ParamIteratorConfig, Params
)

logger = logging.getLogger("fish-risk")


class ModelBase(ABC):
    """
    Base class for our models
    """

    def __init__(self, *args, **kwargs):
        # We leave kwargs open
        self.key = JaxRKey(seed=kwargs.get("seed", 8675309))


class EulerMaruyamaDynamics(ModelBase):
    """
    Evolve an ODE
    """

    def __init__(self, t_end: int, num_points: int, D: float, max_b: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_points = jnp.linspace(0., t_end, num_points)
        self.dt = t_end / num_points
        self.D = D  # diffusion coefficient

    def rhs_pymcode(self) -> Array:
        """
        Will be passed into DifferentialEquation
        p is our parameter tuple (r, k, qE)
        """
        raise NotImplementedError

    def __call__(self, params: Params) -> List[Array]:
        """
        Generate sample data
        Processes batch of params at once
        Shape of Bs is [m, num_param_batches]
        where m is either 1 for the "real" timestep or n_montecarlo for planning
        """
        raise NotImplementedError


class Policy(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, params: Params):
        raise NotImplementedError


class LossModel(ModelBase):
    def __call__(self, t: int) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, num_param_batches]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param t: Future time step
        """
        raise NotImplementedError


class NoisyLossModel(LossModel):
    def __init__(self, scale: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def __call__(self, t: int) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, num_param_batches]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param V_t: Net profit (revenue - cost) at time t | shap
        :param t: Future time step
        :param omega: Discount factor
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


class SigmoidPreferencePrior(PreferencePrior):
    """
    l_bar is a maximal loss we can expect from the simulation
    """

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

    def __call__(self, Lt: Array) -> Array:
        """
        Compute the sigmoid preference prior using loss and l_bar
        Returns an array of shape [m, num_param_batches]
        """
        return jax.nn.log_sigmoid(self.l_bar - Lt)


class ExponentialPreferencePrior(PreferencePrior):
    """
    k is an empirical constant related to stakeholder loss aversion
    k = -ln(p*)/L* where p* is the stakeholder's probability that loss will surpass L*
    """

    def __init__(
        self,
        p_star: Union[ParamIterator, ParamIteratorConfig],
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
        Returns an array of shape [m, num_param_batches]
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
        Returns an array of shape [m, num_param_batches]
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
        Arrays have shape [n_montecarlo, num_param_batches]
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
        Input Arrays have shape [n_montecarlo, num_param_batches] since this isn't called for real timesteps
        Return Array has shape [num_param_batches] since we reduce along the montecarlo axis=0
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
        Input Arrays have shape [n_montecarlo, num_param_batches] since this isn't called for real timesteps
        Return Array has shape [num_param_batches] since we reduce along the montecarlo axis=0
        """
        return Lt_logprob.mean(axis=0)


class NullEntropy(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array, Vt: Array) -> Array:
        """
        Zero out the entropy term
        """
        return jnp.zeros(Lt.shape[1])


class WorldModel(ModelBase):
    """
    Main model that computes risk etc. through simulation of fishery evolution.
    Uses `n_montecarlo` MonteCarlo predictive simulations at a given realworld
    time step to calculate statistics.
    """

    def __init__(
        self,
        params,
        num_param_batches,
        n_montecarlo,
        real_horizon,
        plan_horizon,
        dynamics,
        policy,
        revenue_model,
        cost_model,
        loss_model,
        risk_model,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.params = params
        self.num_param_batches = num_param_batches
        self.n_montecarlo = n_montecarlo
        self.dynamics = dynamics
        self.real_horizon = real_horizon
        self.plan_horizon = plan_horizon
        self.policy = policy
        self.revenue_model = revenue_model
        self.cost_model = cost_model
        self.loss_model = loss_model
        self.risk_model = risk_model

    def sample_policy(self, params: Params) -> Params:
        raise NotImplementedError

    def timestep(
        self, t: int, old_params: Params, planning: bool, omega_iter: ParamIterator | None = None
    ) -> Tuple[Array, Array, Array, Array, Params]:
        """
        Shapes of variables (like Bt, Ct, ...) will be
        (m, num_param_batches)
        where m is either 1 for the "real" timestep or n_montecarlo for planning
        """
        raise NotImplementedError

    def plan(self, params: Params) -> Array:
        """
        Calculate the risk value at the present timestep and the current set of params
        by simulating across a planning horizon with n_montecarlo simulations
        Return shape is [num_param_batches]
        """
        Rt_sim = jnp.zeros(self.num_param_batches)
        ent = jnp.zeros(self.num_param_batches)
        extrinsic = jnp.zeros(self.num_param_batches)
        for t_plan in range(self.plan_horizon):
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
        Returned object is a Params object where each param is an n_montecarlo x num_param_batches size
        The params are identical across n_montecarlo dimension, but differ across num_param_batches dimension
        """
        param_montecarlo_dup = [copy(self.params) for _ in range(self.n_montecarlo)]
        # Bs = jnp.vstack([np.array(p.B) for p in param_montecarlo_dup])
        # return Params(Bs, ws, rs, ks, qEs)
        raise NotImplementedError

    def __call__(self) -> Output:
        """
        Run the main world model simulation.
        We collect various values at each real timestep and store them.
        Collected values will have dimension either (1, num_param_batches) or (num_param_batches).
        However, these get squeezed in the Output object.
        The final [real_horizon, num_param_batch] set of results will be passed into an Output object.
        """
        for t_sim in range(self.real_horizon):
            ...

        # Save the preference params and omega history
        param_history = {
            **self.risk_model.preference_prior.get_param_history()
        }

        return Output(...)
