import copy
import logging
from typing import Dict, Optional, Text, Tuple

import numpy as np
from gymnasium.envs.registration import register
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from omegaconf import DictConfig

from sim.road import AVRoad
from sim.utils import Array
from sim.vehicle import Vehicle

logger = logging.getLogger("av-sim")

Observation = np.ndarray


class AVHighway(HighwayEnv):
    """
    Override the HighwayEnv class for our purposes
    """
    ACC_MAX = Vehicle.ACC_MAX

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics"
                },
                "action": {
                    "type": "ContinuousAction",
                },
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "right_lane_reward": 0.1,
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "normalize_reward": True,
                "offroad_terminal": False
            }
        )
        # Update with our homebrew config defaults
        config.update(cls.av_default_config())

        return config

    @staticmethod
    def av_default_config() -> dict:
        """
        Our default config overrides and new params
        """
        return {
            "target_speed": 30,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            # MonteCarlo horizon; Note that a given step is sim_freq // policy_freq frames (see self._simulate)
            "mc_horizon": 5,
            "n_montecarlo": 10,
            "alpha": 10.,
            "beta": 0.25,
        }

    def update_config(self, cfg: DictConfig, reset: bool=True) -> None:
        """
        Update this env with our run config
        Requires a reset
        """
        self.config.update(cfg)
        logger.info("Resetting env with new config")
        if reset:
            self.reset()

    def _reset(self) -> None:
        """
        Init road and vehicles
        """
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes
        """
        self.road = AVRoad(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
            np_random=self.np_random, record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        """
        Create some new random vehicles of a given type, and add them on the road.
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config[
                "controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            # vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            # Randomize its behavior
            vehicle.randomize_behavior()

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _simulate(self, action: Optional[Action] = None) -> None:
        """
        Perform several steps of simulation with constant action
        """
        # Ratio of sim frames per policy update
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])

        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        info = self._info(obs, action)
        reward = self._reward(action, rewards=info.get('rewards', None))
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "rewards": self._rewards(action),
        }
        return info

    def speed_reward(self) -> float:
        """
        Reward for being near the target speed target_speed

        :return: speed reward
        """
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        score = self.config['alpha'] * np.exp(
            - (forward_speed - self.config['target_speed']) ** 2 / (self.config['alpha'] ** 2)
        )
        logger.debug(f">>> SPEED REWARD: v_forward {forward_speed:0.4f} | score {score:0.4f}")

        if self.config['normalize_reward']:
            max_score = self.config['alpha']
            return score / max_score

        return score

    def collision_reward(self) -> float:
        """
        Reward based on collision-related factors, like braking distance, and proximity

        :return: collision reward
        """
        v_x = self.vehicle.speed * np.cos(self.vehicle.heading)

        # Grab the front and behind cars across different lanes and consider their speed deltas from ego
        logger.debug(
            f">>> COLLISION REWARD START | EGO [{self.vehicle}]\n\tv_x {v_x:0.4f} | lane {self.vehicle.lane_index}"
        )

        # See our AV Project, "Vehicle Agent" section for derivation
        beta = 3 * self.ACC_MAX / 2

        n_nbr = 0
        penalty = 0
        for lane in self.road.network.all_side_lanes(self.vehicle.lane_index):
            logger.debug(f"CHECKING LANE {lane}")
            front, rear = self.road.neighbour_vehicles(self.vehicle, lane)
            if front is not None:
                logger.debug(f"FOUND FRONT [{front}]")
                n_nbr += 1
                front_speed = front.speed * np.cos(front.heading)
                front_delta = front_speed - v_x
                if front_delta < 0:
                    # Front car is slower
                    # Note, lane_distance_to: <argument>.x - self.x
                    front_distance = self.vehicle.lane_distance_to(front)
                    assert front_distance > 0, f"Front distance <= 0: {front_distance}"
                    front_distance = max(front_distance, self.vehicle.LENGTH / 2)
                    _penalty = beta * front_delta ** 2 / (front_distance * (
                            2 ** np.abs(lane[2] - self.vehicle.lane_index[2])))
                    penalty += _penalty
                    logger.debug(
                        f">> {front_speed:0.4f} | {front_delta:0.4f} | {front_distance:0.4f} | {_penalty:0.4f}"
                    )

            if rear is not None:
                logger.debug(f"FOUND REAR [{rear}]")
                n_nbr += 1
                rear_speed = rear.speed * np.cos(rear.heading)
                rear_delta = v_x - rear_speed
                if rear_delta < 0:
                    # Rear car approaching
                    rear_distance = rear.lane_distance_to(self.vehicle)
                    assert rear_distance > 0, f"Rear distance <= 0: {rear_distance}"
                    rear_distance = max(rear_distance, rear.LENGTH / 2)
                    _penalty = beta * rear_delta ** 2 / (rear_distance * (
                            2 ** np.abs(lane[2] - self.vehicle.lane_index[2])))
                    penalty += _penalty
                    logger.debug(f">> {rear_speed:0.4f} | {rear_delta:0.4f} | {rear_distance:0.4f} | {_penalty:0.4f}")

        logger.debug(f"COLLISION PENALTY: {penalty}\n")

        return - penalty

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        speed_reward = self.speed_reward()
        collision_reward = self.collision_reward()

        r = {
            "collision_reward": collision_reward,
            # "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "speed_reward": speed_reward,
            # "on_road_reward": float(self.vehicle.on_road),
        }
        logger.debug(f">>>> REWARDS: {r}")

        return r

    def _reward(self, action: Action, rewards: dict | None = None) -> float:
        """
        :param action: the last action performed
        :param rewards: optionally provide the computed rewards if available, for efficiency
        :return: the corresponding reward
        """
        rewards = rewards or self._rewards(action)
        reward = sum(rewards.values())

        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(
        #         reward,
        #         [self.config["collision_reward"], self.config["right_lane_reward"]],
        #         [0, 1]
        #     )
        # reward *= rewards['on_road_reward']
        return reward

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def simplify(self) -> "AVHighway":
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        distance = self.PERCEPTION_DISTANCE * self.config["mc_horizon"]
        state_copy: "AVHighway" = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, distance
        )

        return state_copy

    def simulate_mc(self) -> tuple[Array, Array, Array]:
        """
        MonteCarlo simulation
        Discount future values by gamma
        Currently operates only on single ego vehicle

        For now, let's just record the single value at the end of the mc_horizon trajectory
        So, mc_horizon is how far ahead we are interested in scoring.

        :return: Tuple of losses, loss_log_probs, collisions. Dimensions [n_mc]
        """
        n_mc = self.config["n_montecarlo"]
        losses = np.zeros(n_mc)
        collisions = np.zeros_like(losses)

        for i in range(n_mc):
            env = self.simplify()
            # Randomize the ego vehicle
            env.vehicle.randomize_behavior()
            for j in range(self.config["mc_horizon"]):
                # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if terminated:
                        collisions[i] = 1
                    break

            losses[i] = -reward

        # The uncertainty in the loss essentially
        # For now, we assume no uncertainty though, and set as a low value
        loss_log_probs = -10 * np.ones_like(losses)
        # Alternatively we may use
        # key = self.key.next_seed()
        # jax_losses = jnp.asarray(losses)
        # scale = 1.0
        # rloss, loss_log_probs = JaxGaussian.sample(key, jax_losses, scale)

        return losses, loss_log_probs, collisions


def register_av_highway():
    register(
        id=f"AVAgents/highway-v0",
        entry_point='sim.envs.highway:AVHighway',
        vector_entry_point="sim.envs.highway:AVHighwayVectorized",
        # max_episode_steps=1000,  # Adjust the configuration as needed
    )
