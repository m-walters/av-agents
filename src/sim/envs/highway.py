import copy
import logging
from typing import Dict, Optional, Text, Tuple

import numpy as np
from gymnasium.utils import seeding
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import RoadNetwork
from omegaconf import DictConfig

from sim.road import AVRoad
from sim.utils import Array
from sim.vehicles.highway import IDMVehicle, MetaActionVehicle

logger = logging.getLogger("av-sim")

Observation = np.ndarray
AVVehicle = IDMVehicle | MetaActionVehicle


class AVHighway(HighwayEnv):
    """
    Override the HighwayEnv class for our purposes
    """
    ACC_MAX = IDMVehicle.ACC_MAX
    VEHICLE_MAX_SPEED = IDMVehicle.MAX_SPEED
    PERCEPTION_DISTANCE = 5 * IDMVehicle.MAX_SPEED

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        self.vehicle_lookup = {}  # Get vehicle by its ID
        super().__init__(config, render_mode)

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
                "defensive_reward": -1,  # The reward received when colliding with a vehicle.
                # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "right_lane_reward": 0.1,
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "normalize_reward": True,
                "offroad_terminal": False,
                "speed_limit": 30
            }
        )
        # Update with our homebrew config defaults
        config.update(cls.av_default_config())

        return config

    @staticmethod
    def av_default_config() -> dict:
        """
        Our custom config params used in this class
        """
        return {
            "reward_speed": 40,
            "control_vehicle_type": "sim.vehicles.highway.IDMVehicle",
            "simulation_frequency": 15,  # frames per second
            "policy_frequency": 5,  # policy checks per second (and how many 'steps' per second)
            "n_montecarlo": 10,
            # MonteCarlo horizon; Note that a given step is sim_freq // policy_freq frames (see self._simulate)
            "mc_horizon": 5,
            "alpha": 10.,
            "beta": 0.25,
        }

    def update_config(self, cfg: DictConfig, reset: bool = True) -> None:
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
        # Clear the vehicle lookup
        self.vehicle_lookup = {}

        self._create_road()
        self._create_vehicles()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # We need to seed our RNG here again so that it works with async parallel runs
        np.random.seed(seed)
        # seeding.np_random(seed)

        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created

        obs, info = super().reset(seed=seed, options=options)
        self.action_space.seed(seed)

        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == 'human':
            self.render()
        return obs, info

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes
        Super-method forces speed_limit=30
        """
        self.road = AVRoad(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=self.config["speed_limit"]
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        """
        Create some new random vehicles of a given type, and add them on the road.
        """
        control_vehicle_class = utils.class_from_path(self.config["control_vehicle_type"])
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config[
                "controlled_vehicles"]
        )

        self.controlled_vehicles = []
        av_id = 1
        for others in other_per_controlled:
            vehicle = control_vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                av_id=str(av_id),
                target_speed=self.config["reward_speed"]
            )
            self.vehicle_lookup[vehicle.av_id] = vehicle
            av_id += 1
            # vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            # Randomize its behavior
            if hasattr(vehicle, "randomize_behavior"):
                vehicle.randomize_behavior()

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                # Since this may be an external class..
                setattr(vehicle, "av_id", str(av_id))
                self.vehicle_lookup[vehicle.av_id] = vehicle
                av_id += 1
                if hasattr(vehicle, "randomize_behavior"):
                    vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

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

    def speed_reward(self, vehicle: AVVehicle | None = None) -> float:
        """
        Reward for being near the reward_speed

        :param vehicle: Optional select vehicle. Defaults to self.vehicle
        :return: speed reward
        """
        vehicle = vehicle or self.vehicle
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        score = self.config['alpha'] * np.exp(
            - (forward_speed - self.config['reward_speed']) ** 2 / (self.config['alpha'] ** 2)
        )
        logger.debug(f">>> SPEED REWARD: v_forward {forward_speed:0.4f} | score {score:0.4f}")

        if self.config['normalize_reward']:
            max_score = self.config['alpha']
            return score / max_score

        return score

    def defensive_reward(self, vehicle: AVVehicle | None = None) -> float:
        """
        Reward based on collision-related factors, like braking distance, and proximity

        :param vehicle: Optional select vehicle. Defaults to self.vehicle
        :return: collision reward
        """
        vehicle = vehicle or self.vehicle
        v_x = vehicle.speed * np.cos(vehicle.heading)

        # Grab the front and behind cars across different lanes and consider their speed deltas from ego
        logger.debug(
            f">>> COLLISION REWARD START | EGO [{vehicle}]\n\tv_x {v_x:0.4f} | lane {vehicle.lane_index}"
        )

        # See our AV Project, "Vehicle Agent" section for derivation
        # In IDMVehicle.act you can see acceleration getting clipped by [-ACC_MAX, ACC_MAX]
        beta = 1 / (2 * self.ACC_MAX)
        logger.debug(f">> BETA: {beta}")

        n_nbr = 0
        penalty = 0
        for lane in self.road.network.all_side_lanes(vehicle.lane_index):
            logger.debug(f"CHECKING LANE {lane}")
            if abs(lane[2] - vehicle.lane_index[2]) > 1:
                # Don't worry about lanes that are more than 1 away
                logger.debug(f"SKIPPING LANE {lane}")
                continue

            front, rear = self.road.neighbour_vehicles(vehicle, lane)
            if front is not None:
                logger.debug(f"FOUND FRONT [{front}] | LANE {front.lane_index[2]}")
                n_nbr += 1
                front_speed = front.speed * np.cos(front.heading)
                front_delta = front_speed - v_x
                if front_delta < 0:
                    # Front car is slower
                    # Note, lane_distance_to: <argument>.x - self.x
                    front_distance = vehicle.lane_distance_to(front)
                    assert front_distance > 0, f"Front distance <= 0: {front_distance}"
                    front_distance = max(front_distance, vehicle.LENGTH * 1.2)
                    _penalty = beta * front_delta ** 2 / (
                            front_distance * (2 ** np.abs(lane[2] - vehicle.lane_index[2])))
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
                    rear_distance = rear.lane_distance_to(vehicle)
                    assert rear_distance > 0, f"Rear distance <= 0: {rear_distance}"
                    rear_distance = max(rear_distance, rear.LENGTH * 1.2)
                    _penalty = beta * rear_delta ** 2 / (
                            rear_distance * (2 ** np.abs(lane[2] - vehicle.lane_index[2])))
                    penalty += _penalty
                    logger.debug(f">> {rear_speed:0.4f} | {rear_delta:0.4f} | {rear_distance:0.4f} | {_penalty:0.4f}")

        logger.debug(f"COLLISION PENALTY: {penalty}\n across {n_nbr} nbrs")
        # Average over the neighbors
        # penalty /= n_nbr if n_nbr > 0 else 1

        # Multiply it by the sin(theta) of our heading -- the harder we are turning the more dangerous this is
        penalty *= 1 + np.abs(np.sin(vehicle.heading))

        return -penalty

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Compute and collect the suite of rewards for our control vehicle.
        TODO -- Consider a speed-limit penalty (see IDMVehicle.accelerate)
        """
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]

        speed_reward = self.speed_reward()
        defensive_reward = self.defensive_reward()

        r = {
            "defensive_reward": defensive_reward,
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
        #         [self.config["defensive_reward"], self.config["right_lane_reward"]],
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

        distance = self.PERCEPTION_DISTANCE
        # Add to the distance the expected amount traveled by the car after mc_horizon steps at the given car speed
        # Since policy_frequency is how many calls to step per second and mc_horizon is num steps per mc sim..
        distance += self.VEHICLE_MAX_SPEED * self.config["mc_horizon"] / self.config["policy_frequency"]

        state_copy: "AVHighway" = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, distance
        )

        return state_copy

    @staticmethod
    def seed_montecarlo(mc_env: "AVHighway", seed):
        """
        Seed a montecarlo environment
        """
        mc_env.road.np_random = np.random.RandomState(seed)
        mc_env.np_random = np.random.RandomState(seed)
        mc_env.action_space.seed(seed)
        mc_env.observation_space.seed(seed)

        # Randomize the behavior of all the alter vehicles on the road
        for vehicle in mc_env.road.vehicles:
            if vehicle is not mc_env.vehicle:
                vehicle.randomize_behavior()

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
        horizon = self.config["mc_horizon"]
        losses = np.zeros(n_mc)
        collisions = np.zeros_like(losses)
        # We have to at least do one action sample before an MC loop
        # otherwise, if the action sample isn't properly called in the world sim loop,
        # the MC action sampling will always start from the same point
        # self.action_space.sample()
        # ..or mb its ok

        # Very sensitive that these seeds are python integers...
        seeds = [int(s) for s in self.np_random.integers(0, 1_000, size=n_mc)]
        for i, i_seed in zip(range(n_mc), seeds):
            env = self.simplify()
            # Add randomness for different MC sims
            self.seed_montecarlo(env, i_seed)

            for j in range(horizon):
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

    def action_sample(self):
        """
        Shorthand method for action sampling that can be called from async env
        """
        return self.action_space.sample()

    def observation_sample(self):
        """
        Shorthand method for observation sampling that can be called from async env
        """
        return self.observation_space.sample()
