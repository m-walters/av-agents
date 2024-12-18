import logging
from abc import ABC
from typing import Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import differential_entropy as entr

from sim.utils import (Array, NpyRKey, Number)

logger = logging.getLogger("av-sim")


class ModelBase(ABC):
    """
    Base class for our models
    """

    def __init__(self, *args, **kwargs):
        # We leave kwargs open
        self.key = NpyRKey(seed=kwargs.get("seed", 8675309))


class LossModel(ModelBase):
    def __init__(self, alpha: float, reward_speed: Number, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.reward_speed = reward_speed
        self.beta = beta

    def __call__(self, v: float, neighbors: Array) -> Tuple[Array, Array]:
        """
        Return loss and log-prob arrays of shape [m, world_draws]
        where m is either 1 for the "real" timestep or n_montecarlo for planning

        :param v: speed of vehicle
        :param neighbors: num_neighbors x 3 array, columns are [distance, speed, lane]
        """
        speed_loss = - self.alpha * np.exp(-(v - self.reward_speed) ** 2 / self.alpha)
        collision_loss = self.beta * np.mean(
            1 / (2 ** neighbors[:, 2]) * np.divide(np.max(0, v - neighbors[:, 1]), neighbors[:, 0])
        )
        loss = speed_loss + collision_loss
        return loss, -10 * np.ones(loss.shape)  # Set log_probs to -10 to show high precision


####
# Deprecated until we put Jax back online
#
# class NoisyLossModel(LossModel):
#     def __init__(self, scale: float, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.scale = scale
#
#     def __call__(self, v: float, neighbors: Array) -> Tuple[Array, Array]:
#         """
#         Return loss and log-prob arrays of shape [m, world_draws]
#         where m is either 1 for the "real" timestep or n_montecarlo for planning
#
#         :param v: speed of vehicle
#         :param neighbors: num_neighbors x 3 array, columns are [distance, speed, lane]
#         """
#         loss, _ = super(NoisyLossModel, self).__call__(v, neighbors)
#         key = self.key.next_key()
#         jax_loss = jnp.asarray(loss)
#         rloss, log_probs = JaxGaussian.sample(key, jax_loss, self.scale)
#         return rloss, log_probs


class SoftmaxPreferencePrior(ModelBase):
    """
    Softmax preference prior weighted by constant kappa
    """

    def __init__(
        self,
        kappa: Number,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kappa = kappa

    def __call__(self, Lt: Array, take_log: bool = True) -> Array:
        """
        Compute the softmax preference prior
        By default, we take the logarithm, for use in the risk calculation
        Input and return shape: [m, world_draws] for m montecarlo samples
        """
        if take_log:
            # Get fancy with scipy
            return -self.kappa * Lt - logsumexp(-self.kappa * Lt, axis=0)
        else:
            return np.exp(-self.kappa * Lt) / np.sum(np.exp(-self.kappa * Lt), axis=0)


class ExponentialPreferencePrior(ModelBase):
    """
    k is an empirical constant related to stakeholder loss aversion
    k = -ln(p*)/L* where p* is the stakeholder's probability that loss will surpass L*
    """

    def __init__(
        self,
        p_star: Number,
        l_star: Number,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p_star = p_star
        self.l_star = l_star

    @property
    def k(self):
        return -np.log(self.p_star) / self.l_star

    def __call__(self, Lt: Array, take_log: bool = True) -> Array:
        """
        Compute the exponential preference prior
        By default, we take the logarithm, for use in the risk calculation
        Returns an array of shape [m, world_draws]
        """
        if take_log:
            return -self.k * Lt
        else:
            return np.exp(-self.k * Lt)


class SatisficingPreferencePrior(ModelBase):
    """
    Preference prior similar to Exponential, but with a max at L < 0
    which provides a satisficing condition
    """

    def __init__(
        self,
        p_star: Number,
        l_star: Number,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p_star = p_star
        self.l_star = l_star

    @property
    def k(self):
        return -np.log(self.p_star) / self.l_star

    def __call__(self, Lt: Array, take_log: bool = True) -> Array:
        """
        Compute the satisficing exponential preference prior
        By default, we take the logarithm, for use in the risk calculation
        Returns an array of shape [m, world_draws]
        """
        if take_log:
            return -self.k * np.maximum(0, Lt)
        else:
            return np.exp(-self.k * np.maximum(0, Lt))


class EFERiskModel(ModelBase):
    def __init__(self, preference_prior, *args, **kwargs):
        """
        Expected Free Energy Risk Model
        An inverse temperature parameter beta is used to control the tradeoff between entropy and energy
        We use a low-entropy optimization
        """
        super().__init__(*args, **kwargs)
        self.preference_prior = preference_prior

    def __call__(self, Lt: Array, Lt_logprob: Array) -> Tuple[Array, Array, Array]:
        """
        Compute an array of risk values
        Arrays have shape [n_montecarlo, world_draws]
        """
        # Note that the value computed here, the mean of the log of the preference model,
        # is also called the 'energy' in a VFE framework
        log_pref_mean = self.preference_prior(Lt, take_log=True).mean(axis=0)

        entropy = self.compute_entropy(Lt, Lt_logprob)
        Gt = - log_pref_mean - entropy
        return Gt, entropy, log_pref_mean

    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        raise NotImplementedError("Subclasses must implement this method")


class DifferentialEntropyRiskModel(EFERiskModel):
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


class MonteCarloRiskModel(EFERiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        """
        Compute the Monte Carlo estimate of the entropy of the loss distribution
        Input Arrays have shape [n_montecarlo, world_draws] since this isn't called for real timesteps
        Return Array has shape [world_draws] since we reduce along the montecarlo axis=0
        """
        return Lt_logprob.mean(axis=0)


class NullEntropyRiskModel(EFERiskModel):
    def compute_entropy(self, Lt: Array, Lt_logprob: Array) -> Array:
        """
        Zero out the entropy term
        """
        return np.zeros(Lt.shape[1])
