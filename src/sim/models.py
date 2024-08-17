import logging
from abc import ABC
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from scipy.stats import differential_entropy as entr

from sim.params import ParamCollection
from sim.utils import (
    Array, JaxGaussian, JaxRKey, Number, ParamEvolution, ParamIterator, ParamIteratorConfig
)

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
    def __init__(self, alpha: float, target_speed: Number, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.target_speed = target_speed
        self.beta = beta

    def __call__(self, v: float, neighbors: Array) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, world_draws]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param v: speed of vehicle
        :param neighbors: num_neighbors x 3 array, columns are [distance, speed, lane]
        """
        speed_loss = - self.alpha * jnp.exp(-(v - self.target_speed) ** 2 / self.alpha)
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
        loss, _ = super(NoisyLossModel, self).__call__(v, neighbors)
        key = self.key.next_key()
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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SoftmaxPreferencePrior(PreferencePrior):
    """
    Softmax preference prior weighted by constant kappa
    """

    def __init__(
        self,
        kappa: Number,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._param_history = {
            "kappa": [],
        }
        self.kappa = kappa

    def __call__(self, Lt: Array, take_log: bool = True) -> Array:
        """
        Compute the softmax preference prior
        By default, we take the logarithm, for use in the risk calculation
        Input and return shape: [m, world_draws] for m montecarlo samples
        """
        if take_log:
            return -self.kappa * Lt - jnp.log(jnp.sum(jnp.exp(-self.kappa * Lt), axis=0))
        else:
            return jnp.exp(-self.kappa * Lt) / jnp.sum(jnp.exp(-self.kappa * Lt), axis=0)


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

    def __call__(self, Lt: Array, take_log: bool = True) -> Array:
        """
        Compute the exponential preference prior
        By default, we take the logarithm, for use in the risk calculation
        Returns an array of shape [m, world_draws]
        """
        if take_log:
            return -self.k * Lt
        else:
            return jnp.exp(-self.k * Lt)


class RiskModel(ModelBase):
    def __init__(self, preference_prior, min_risk: Number = 0, *args, **kwargs):
        """
        Risk model
        Takes a preference prior and optionally a minimum risk to be subtracted from values for normalization
        """
        super().__init__(*args, **kwargs)
        self.preference_prior = preference_prior
        self.min_risk = min_risk

    def compute_entropy(self, Lt, Lt_logprob):
        raise NotImplementedError

    def __call__(self, Lt: Array, Lt_logprob: Array) -> Tuple[Array, Array, Array]:
        """
        Compute an array of risk values at a given timestep
        Arrays have shape [n_montecarlo, world_draws]
        """
        # Note that the value computed here, the mean of the log of the preference model,
        # is also called the 'energy' in a VFE framework
        log_pref_mean = self.preference_prior(Lt, take_log=True).mean(axis=0)

        entropy = self.compute_entropy(Lt, Lt_logprob)
        Gt = - entropy - log_pref_mean
        return Gt, entropy, log_pref_mean


class DifferentialEntropyRiskModel(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        """
        Compute the differential entropy of the loss distribution
        Input Arrays have shape [n_montecarlo[, world_draws]]
        Return Array has shape [world_draws] since we reduce along the montecarlo axis=0
        """
        ent = entr(Lt, axis=0) if len(Lt) > 1 else 0.
        if np.any(ent == float('-inf')):
            logger.warning("-inf encountered in entropy. You may need more MC samples")
            # set arbitrarily to -10
            ent = np.where(ent == -float('inf'), -10, ent)
        return ent


class MonteCarloRiskModel(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        """
        Compute the Monte Carlo estimate of the entropy of the loss distribution
        Input Arrays have shape [n_montecarlo, world_draws] since this isn't called for real timesteps
        Return Array has shape [world_draws] since we reduce along the montecarlo axis=0
        """
        return Lt_logprob.mean(axis=0)


class NullEntropy(RiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        """
        Zero out the entropy term
        """
        return jnp.zeros(Lt.shape[1])
