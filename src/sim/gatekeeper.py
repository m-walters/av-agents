"""
Gatekeeper module
"""
import copy
import logging
import multiprocessing
from enum import Enum
from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
from highway_env import utils
from omegaconf import DictConfig

from sim import models
from sim.envs import highway, intersection
from sim.utils import JaxRKey
from sim.vehicles.highway import AVVehicleType, VehicleBase

logger = logging.getLogger("av-sim")


class Behaviors(Enum):
    NOMINAL = "nominal"
    CONSERVATIVE = "conservative"


def behavior_to_num(behavior: Behaviors) -> int:
    if behavior == Behaviors.NOMINAL:
        return 0
    elif behavior == Behaviors.CONSERVATIVE:
        return 1


class BehaviorConfig(TypedDict):
    enable: bool
    nominal_class: str
    nominal_risk_threshold: float
    conservative_class: str
    conservative_risk_threshold: float


class GatekeeperConfig(TypedDict):
    n_controlled: int
    n_montecarlo: int
    mc_horizon: int
    mc_period: int
    # Models
    preference_prior: dict
    risk_model: dict
    # Risk and behavior
    nominal_risk_threshold: float
    conservative_risk_threshold: float
    behavior_cfg: BehaviorConfig


class Gatekeeper:
    """
    Gatekeeper class.
    Concerns itself with actions and surroundings of a single ego vehicle.
    """

    def __init__(
        self,
        gk_idx: int,
        vehicle: AVVehicleType,
        behavior_cfg: BehaviorConfig,
        rstar_range: tuple[float, float],
        seed: int,
    ):
        self.gk_idx = gk_idx  # Tracks this gatekeepers index in the results array for the GKCommand
        # Don't store the vehicle object because env duplicating will change the memory address during MC
        self.vehicle_id = vehicle.av_id
        self.rkey = JaxRKey(seed)

        # See AVHighway class for types
        # These must be callables in the Env class
        self.reward_types = [
            "defensive_reward",
            "speed_reward",
            "crash_reward",
        ]

        # Behavior control logic
        self.behavior_ctrl_enabled = behavior_cfg["enable"]
        self.behavior = Behaviors.NOMINAL
        if self.behavior_ctrl_enabled:
            self.nominal_behavior = utils.class_from_path(behavior_cfg["nominal_class"])
            self.nominal_risk_threshold = rstar_range[0]
            self.conservative_behavior = utils.class_from_path(behavior_cfg["conservative_class"])
            self.conservative_risk_threshold = rstar_range[1]

    def get_vehicle(self, env):
        return env.vehicle_lookup[self.vehicle_id]

    def crashed(self, env) -> bool:
        """
        Use ID to see if this vehicle crashed
        """
        vehicle = self.get_vehicle(env)
        return vehicle.crashed or (env.config['offroad_terminal'] and not vehicle.on_road)

    def calculate_reward(self, env: Union[highway.AVHighway, intersection.AVIntersection]):
        """
        Calculate the reward for this vehicle
        """
        vehicle = self.get_vehicle(env)
        rewards = {
            reward: getattr(env, reward)(vehicle) for reward in self.reward_types
        }
        reward = sum(rewards.values())
        if env.config["normalize_reward"]:
            # Best is 1 for the normalized speed reward
            # Worst is -2 for the normalized crash penalty and the normalized defensive penalty
            best = 1
            worst = -2
            reward = utils.lmap(
                reward,
                [worst, best],
                [0, 1]
            )

            # Assert numpy array values are in range [0, 1]
            assert np.all(0 <= reward) and np.all(reward <= 1)

        return reward

    def update_policy(self, nbrhood_cre, env):
        """
        Update policy based on nbrhood cre
        """
        if not self.behavior_ctrl_enabled:
            return

        if self.behavior == Behaviors.NOMINAL and nbrhood_cre > self.nominal_risk_threshold:
            # Get the vehicle and change its policy
            vehicle = self.get_vehicle(env)
            vehicle.set_behavior_params(self.conservative_behavior)
            self.behavior = Behaviors.CONSERVATIVE
        elif self.behavior == Behaviors.CONSERVATIVE and nbrhood_cre < self.conservative_risk_threshold:
            vehicle = self.get_vehicle(env)
            vehicle.set_behavior_params(self.nominal_behavior)
            self.behavior = Behaviors.NOMINAL

    def __str__(self):
        return f"Gatekeeper[{self.gk_idx} | {self.vehicle_id}]"

    def __repr__(self):
        return self.__str__()


class GatekeeperCommand:
    """
    Central coordinator of gatekeepers
    """

    def __init__(
        self,
        env: Union[highway.AVHighway, intersection.AVIntersection],
        gk_cfg: DictConfig[GatekeeperConfig],
        control_vehicles: list[AVVehicleType],
        seed: int,
    ):
        """
        Spawn and assign new gatekeepers for each vehicle provided
        """
        if gk_cfg.n_controlled != len(control_vehicles):
            raise ValueError(
                f"GatekeeperCommand received {len(control_vehicles)}, but is configured "
                f"for {gk_cfg.n_controlled}"
            )

        self.env = env
        self.seed = seed
        self.config = gk_cfg.copy()
        self.n_montecarlo: int = gk_cfg.n_montecarlo
        self.mc_horizon: int = gk_cfg.mc_horizon
        self.mc_period: int = gk_cfg.mc_period
        self.n_controlled: int = gk_cfg.n_controlled
        self.rkey = JaxRKey(seed)

        # Init models
        self._init_models(gk_cfg)

        # Determine risk thresholds
        # For reference as (p_star, r_star) tuples:
        #   (0.01, 4.6)
        #   (0.1, 2.3)
        #   (0.2, 1.6)
        #   (0.9, 0.11)
        # This quantity here has the nice property that it is L* fraction of the max normalized risk,
        # which in the Loss-normalized [0,1] case, the max risk is k = -log(p_star)/l_star
        nominal_rstar = -np.log(gk_cfg['preference_prior']['p_star']) * 1.1
        conservative_rstar = nominal_rstar * 0.9 / 1.1

        # Init GKs
        self.nbr_distance = VehicleBase.MAX_SPEED * 0.7  # For GK neighborhood discovery
        self.gatekeepers: List["Gatekeeper"] = []
        self.gatekeeper_lookup: Dict[int, Gatekeeper] = {}
        for i, v in enumerate(control_vehicles):
            self._spawn_gatekeeper(v, i, gk_cfg.behavior_cfg, (nominal_rstar, conservative_rstar))

    @property
    def gatekept_vehicles(self):
        return self.gatekeeper_lookup.keys()

    def _init_models(self, gk_cfg: DictConfig[GatekeeperConfig]):
        """
        Initialize the preference and risk models
        """
        self.preference_prior = getattr(models, gk_cfg.preference_prior.model)(
            **gk_cfg.preference_prior, seed=self.seed
        )
        self.risk_model = getattr(models, gk_cfg.risk_model.model)(
            preference_prior=self.preference_prior, **gk_cfg.risk_model, seed=self.seed
        )

    def _spawn_gatekeeper(self, vehicle: AVVehicleType, gk_idx, behavior_cfg, rstar_range):
        """
        Spawn a gatekeeper for a vehicle
        """
        gk = Gatekeeper(
            gk_idx,
            vehicle,
            behavior_cfg,
            rstar_range,
            self.rkey.next_seed()
        )
        self.gatekeepers.append(gk)
        self.gatekeeper_lookup[vehicle.av_id] = gk

    def _mc_trajectory(
        self, seed: int,
    ):
        """
        Run an MC trajectory and accumulate the results across controlled vehicles
        """
        env = copy.deepcopy(self.env)
        env.seed_montecarlo(env, seed)

        losses = np.zeros(self.n_controlled)
        collisions = np.zeros_like(losses)

        for j in range(self.mc_horizon):
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()

            # Like the regular Env sim but we ignore reward calculations and such
            env.time += 1 / env.config["policy_frequency"]
            env._simulate(action)

            if j + 1 < self.mc_horizon:
                continue

            # Else, last step so we take measurements
            for i_gk, gk in enumerate(self.gatekeepers):
                crashed = gk.crashed(env)
                if crashed:
                    collisions[i_gk] = 1
                reward = gk.calculate_reward(env)
                losses[i_gk] = 1 - reward

        return losses, collisions

    def run(self, pool: Optional["multiprocessing.Pool"], gamma: float = 1.0) -> dict:
        """
        Perform montecarlo simulations and calculate risk equations etc.

        :return: Several result arrays. Dimensions [n_controlled, n_mc]
        """
        if gamma != 1.0:
            raise NotImplementedError("Discounting not yet implemented")

        # losses = np.zeros((self.n_montecarlo, self.n_controlled))
        # collisions = np.zeros_like(losses)

        # Very sensitive that these seeds are python integers...
        seeds = [int(s) for s in self.rkey.next_seeds(self.n_montecarlo)]

        if pool:
            # Issue trajectories to workers
            # Use starmap if you need to provide more arguments
            results = pool.map(self._mc_trajectory, seeds)
            # Stack results along first dimension
            results = np.stack(results, axis=0)
        else:
            results = np.zeros((self.n_montecarlo, 2, self.n_controlled))
            for i, seed in enumerate(seeds):
                results[i] = self._mc_trajectory(seed)

        losses = results[:, 0, :]
        collisions = results[:, 1, :]

        # The uncertainty in the loss essentially
        # For now, we assume no uncertainty though, and set as a low value
        loss_log_probs = -10 * np.ones_like(losses)
        # Alternatively we may use
        # key = self.env.key.next_seed()
        # jax_losses = jnp.asarray(losses)
        # scale = 1.0
        # rloss, loss_log_probs = JaxGaussian.sample(key, jax_losses, scale)

        risk, entropy, energy = self.risk_model(losses, loss_log_probs)

        # Let's discover gatekeeper neighborhoods, then compute average risks for each,
        # followed by policy updating
        global_cre = np.sum(risk)
        measured_gks = []
        nbrhoods = []

        for gk in self.gatekeepers:
            if gk in measured_gks:
                continue

            # Get the neighborhood around this gatekeeper
            if gk.get_vehicle(self.env).crashed:
                # Crashed are excluded from neighborhoods
                nbrhood = [gk]
            else:
                nbrhood = self._discover_neighborhood(gk, [])
            measured_gks += nbrhood
            nbrhoods.append(nbrhood)

        assert len(set(measured_gks)) == len(measured_gks), f"Duplicated GK measured"

        for nbrhood in nbrhoods:
            # Average the CRE for the nbrhood
            # Convert GKs to their 0-based indices
            nbrhood_idx = np.array([_gk.gk_idx for _gk in nbrhood])
            nbrhood_cre = np.array(risk[nbrhood_idx].mean())

            # Update the risk value for the nbrhood
            # JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another
            # .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
            risk = risk.at[nbrhood_idx].set(nbrhood_cre)

            # Update the policies
            for gk in nbrhood:
                gk.update_policy(nbrhood_cre, self.env)

        return {
            "losses": losses,
            "loss_log_probs": loss_log_probs,
            "collisions": collisions,
            "risk": risk,
            "entropy": entropy,
            "energy": energy,
            "global_cre": global_cre,
        }

    def _discover_neighborhood(self, gk: Gatekeeper, found_gks: List[Gatekeeper]):
        """
        Recursive GK neighborhood collect.
        Ignore found_gks.
        """
        found_gks += [gk]
        for nbr in self.env.road.close_vehicles_to(gk.get_vehicle(self.env), self.nbr_distance):
            if nbr.av_id not in self.gatekept_vehicles:
                # Ignore
                continue

            if nbr.crashed:
                # Also exclude from neighborhood
                continue

            nbr_gk = self.gatekeeper_lookup[nbr.av_id]
            if nbr_gk not in found_gks:
                found_gks = self._discover_neighborhood(nbr_gk, found_gks)

        return found_gks

    def collect_behaviors(self):
        return np.array([behavior_to_num(gk.behavior) for gk in self.gatekeepers])
