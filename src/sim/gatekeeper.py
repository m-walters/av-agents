"""
Gatekeeper module
"""
import logging
import multiprocessing
from typing import List, TypedDict, Union

import numpy as np
from omegaconf import DictConfig

from sim.envs import highway, intersection
from sim.utils import JaxRKey
from sim.vehicles.highway import AVVehicleType

logger = logging.getLogger("av-sim")


class GatekeeperConfig(TypedDict):
    multiprocessing_cpus: int | None
    n_controlled: int
    risk_threshold: float
    n_montecarlo: int
    mc_horizon: int
    mc_period: int


class Gatekeeper:
    """
    Gatekeeper class.
    Concerns itself with actions and surroundings of a single ego vehicle.
    """

    def __init__(
        self,
        gk_idx: int,
        vehicle: AVVehicleType,
        risk_threshold: float,
        seed: int,
    ):
        self.gk_idx = gk_idx  # Tracks this gatekeepers index in the results array for the GKCommand
        self.vehicle_id = vehicle.av_id
        self.risk_threshold: float = risk_threshold
        self.rkey = JaxRKey(seed)

        # See AVHighway class for types
        # These must be callables in the Env class
        self.reward_types = [
            "collision_reward",
            "speed_reward",
        ]

    def crashed(self, env) -> bool:
        """
        Use ID to see if this vehicle crashed
        """
        vehicle = env.vehicle_lookup[self.vehicle_id]
        return vehicle.crashed or (env.config['offroad_terminal'] and not vehicle.on_road)

    def calculate_reward(self, env: Union[highway.AVHighway, intersection.AVIntersection]):
        """
        Calculate the reward for this vehicle
        """
        vehicle = env.vehicle_lookup[self.vehicle_id]
        return {
            reward: getattr(env, reward)(vehicle) for reward in self.reward_types
        }


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
        self.risk_threshold: float = gk_cfg.risk_threshold
        self.n_montecarlo: int = gk_cfg.n_montecarlo
        self.mc_horizon: int = gk_cfg.mc_horizon
        self.mc_period: int = gk_cfg.mc_period
        self.n_controlled: int = gk_cfg.n_controlled
        self.rkey = JaxRKey(seed)

        # Init GKs
        self.gatekeepers: List["Gatekeeper"] = []
        for i, v in enumerate(control_vehicles):
            self._spawn_gatekeeper(v, i)

    def _spawn_gatekeeper(self, vehicle: AVVehicleType, gk_idx):
        """
        Spawn a gatekeeper for a vehicle
        """
        self.gatekeepers.append(Gatekeeper(gk_idx, vehicle, self.risk_threshold, self.rkey.next_seed()))

    def _mc_trajectory(
        self, seed: int,
    ):
        """
        For use within the multiprocessing workers.
        We can take advantage of how mp copies objects to skip copy.deepcopy(env)
        """
        self.env.seed_montecarlo(self.env, seed)

        losses = np.zeros(self.n_controlled)
        collisions = np.zeros_like(losses)

        for j in range(self.mc_horizon):
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = self.env.action_space.sample()

            # Like the regular Env sim but we ignore reward calculations and such
            self.env.time += 1 / self.env.config["policy_frequency"]
            self.env._simulate(action)

            if j + 1 < self.mc_horizon:
                continue

            # Else, last step so we take measurements
            for i_gk, gk in enumerate(self.gatekeepers):
                crashed = gk.crashed(self.env)
                if crashed:
                    collisions[i_gk] = 1
                reward = sum(gk.calculate_reward(self.env).values())
                losses[i_gk] -= reward

            # if self.env.render_mode == 'human':
            #     self.env.render()

            # if terminated or truncated:
            #     if terminated:
            #         collisions[i] = 1
            #     break
            #
            # losses[i] = -reward

        return losses, collisions

    def simulate_mc(self, pool: multiprocessing.Pool, gamma: float = 1.0):
        """
        Perform Montecarlo simulations for this gatekeepers vehicle

        :return: Tuple of losses, loss_log_probs, collisions. Dimensions [n_controlled, n_mc]
        """
        if gamma != 1.0:
            raise NotImplementedError("Discounting not yet implemented")

        # losses = np.zeros((self.n_montecarlo, self.n_controlled))
        # collisions = np.zeros_like(losses)

        # Very sensitive that these seeds are python integers...
        seeds = map(int, self.rkey.next_seeds(self.n_montecarlo))

        # Issue trajectories to workers
        # Use starmap if you need to provide more arguments
        results = pool.map(self._mc_trajectory, seeds)

        # Stack results along first dimension
        results = np.stack(results, axis=0)

        # for i, (loss, collision) in enumerate(results):
        #     losses[i, :] = loss
        #     collisions[i, :] = collision
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

        return losses, loss_log_probs, collisions
