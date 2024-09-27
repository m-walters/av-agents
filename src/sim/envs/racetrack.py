import copy
import logging
from typing import Optional, Tuple

import numpy as np
from gymnasium.utils import seeding
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from omegaconf import DictConfig

from sim.utils import Array
from sim.vehicles.highway import IDMVehicle, MetaActionVehicle

logger = logging.getLogger("av-sim")

Observation = np.ndarray
AVVehicle = IDMVehicle | MetaActionVehicle


class AVRacetrack(RacetrackEnv):
    """
    Override the HighwayEnv class for our purposes
    """
    ACC_MAX = IDMVehicle.ACC_MAX
    VEHICLE_MAX_SPEED = IDMVehicle.MAX_SPEED
    PERCEPTION_DISTANCE = 5 * IDMVehicle.MAX_SPEED

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        self.vehicle_lookup = {}  # Get vehicle by its ID
        self.multiagent: bool = False
        self.alter_vehicles: list[AVVehicle] = []
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "crash_penalty": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "vehicles_count": 2,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],

                "lanes_count": 4,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "normalize_reward": True,
                "offroad_terminal": False,
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
            "reward_speed": 20,
            "speed_limit": 20,
            "crash_penalty": -1,  # The reward received when colliding with a vehicle.
            "max_defensive_penalty": -3,  # Cap the defensive reward/penalty
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

    @property
    def vehicle(self) -> AVVehicle:
        # if self.multiagent:
        #     raise RuntimeError("Multi-agent environment. AVHighway.vehicle is ambiguous.")

        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    def _reset(self) -> None:
        """
        Init road and vehicles
        """
        # Clear the vehicle lookup
        self.vehicle_lookup = {}

        self._make_road()
        self._make_vehicles()
        self.multiagent = len(self.controlled_vehicles) > 1

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

    def _make_road(self) -> None:
        net = RoadNetwork()
        w = 5
        w2 = 2 * w
        default_speedlimit = 10

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [200, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w],
                [200, w],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w2],
                [200, w2],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 2 - Circular Arc #1
        center1 = [200, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w2,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 3 - Vertical Straight
        delta_extension = -1.  # Better join
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220, -20],
                [220, -60 + delta_extension],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w, -20],
                [220 + w, -60 + delta_extension],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w2, -20],
                [220 + w2, -60 + delta_extension],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 4 - Circular Arc #2
        center4 = [205, -60]
        radii4 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 5 - Circular Arc #3
        center5 = [170, -60]
        radii5 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=w,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 - w,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 6 - Slant
        # Extending [-30,-30]
        extend = np.array([-30, -30])
        start6 = np.array([155.7, -45.7])
        end6 = np.array([135.7, -65.7]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6,
                end6,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_2 = np.array([159.3934, -49.2])
        end6_2 = np.array([139.3934, -69.2]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_2,
                end6_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_3 = np.array(
            [
                start6[0] + 2 * (start6_2[0] - start6[0]),
                start6[1] + 2 * (start6_2[1] - start6[1])
            ]
        )
        end6_3 = np.array(
            [
                end6[0] + 2 * (end6_2[0] - end6[0]),
                end6[1] + 2 * (end6_2[1] - end6[1]),
            ]
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_3,
                end6_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 7 - Circular Arc #4
        # Reflect it with the slant
        center7 = np.array([118.1, -48.1]) + extend
        radii7 = 25
        theta7 = 317
        # theta7_end = 270 - (theta7 - 270) - 10
        theta7_end = 205
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end - 3),  # nicer
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + w2,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 8 - Next slant
        # Reflected from the last arc's center
        start8 = np.array(
            [
                center7[0] + radii7 * np.cos(np.deg2rad(theta7_end)),
                center7[1] + radii7 * np.sin(np.deg2rad(theta7_end))
            ]
        )
        start8_2 = np.array(
            [
                center7[0] + (radii7 + w) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w) * np.sin(np.deg2rad(theta7_end))
            ]
        )
        start8_3 = np.array(
            [
                center7[0] + (radii7 + w2) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w2) * np.sin(np.deg2rad(theta7_end))
            ]
        )

        # We preemptively take section 9's radius to make a nice join.
        radii9 = 15
        rad = np.deg2rad(30)
        end8 = np.array(
            [
                42 - radii9 * np.cos(rad),
                -radii9 - radii9 * np.sin(rad)
            ]
        )
        end8_2 = np.array(
            [
                42 - (radii9 + w) * np.cos(rad),
                -radii9 - (radii9 + w) * np.sin(rad)
            ]
        )
        end8_3 = np.array(
            [
                42 - (radii9 + w2) * np.cos(rad),
                -radii9 - (radii9 + w2) * np.sin(rad)
            ]
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8,
                end8,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_2,
                end8_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_3,
                end8_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 9 - Circular arc 7, end
        # Since y2 = 0...
        center9 = np.array([42, -radii9])
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9,
                np.deg2rad(210),
                np.deg2rad(88),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w,
                np.deg2rad(210),
                np.deg2rad(90),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w2,
                np.deg2rad(212),
                np.deg2rad(88),  # nicer join
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _make_vehicles(self) -> None:
        rng = self.np_random
        control_vehicle_class = utils.class_from_path(self.config["control_vehicle_type"])
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Controlled vehicles
        self.controlled_vehicles = []
        self.alter_vehicles = []
        av_id = 1
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", 0)
            )
            # Long
            random_lane_index = self.road.network.random_lane_index(rng)
            controlled_vehicle = control_vehicle_class.make_on_lane(
                self.road,
                random_lane_index,
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(random_lane_index).length
                ),
                speed=0.7 * rng.normal(self.config["speed_limit"])
            )
            setattr(controlled_vehicle, "av_id", str(av_id))
            av_id += 1
            self.vehicle_lookup[controlled_vehicle.av_id] = controlled_vehicle
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            # Randomize its behavior
            if hasattr(controlled_vehicle, "randomize_behavior"):
                controlled_vehicle.randomize_behavior()

        other_vehicles = self.config["vehicles_count"] - self.config["controlled_vehicles"]
        for i in range(other_vehicles):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                random_lane_index,
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(random_lane_index).length
                ),
                speed=0.7 * rng.normal(self.config["speed_limit"]),
            )
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 12:
                    break
            else:
                setattr(vehicle, 'av_id', str(av_id))
                av_id += 1
                self.vehicle_lookup[vehicle.av_id] = vehicle
                self.alter_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
                # Randomize its behavior
                if hasattr(vehicle, "randomize_behavior"):
                    vehicle.randomize_behavior()

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or not self.road.vehicles:
            raise NotImplementedError("The road and vehicles must be initialized in the environment implementation")

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

    def crash_reward(self, vehicle: AVVehicle | None = None) -> float:
        """
        Negative penalty for crashing
        """
        vehicle = vehicle or self.vehicle
        return self.config['crash_penalty'] if vehicle.crashed else 0

    def defensive_reward(self, vehicle: AVVehicle | None = None) -> float:
        """
        Reward based on collision-related factors, like braking distance, and proximity

        :param vehicle: Optional select vehicle. Defaults to self.vehicle
        :return: collision reward
        """
        vehicle = vehicle or self.vehicle
        if vehicle.crashed:
            return self.config['max_defensive_penalty']

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

        if -penalty < self.config['max_defensive_penalty']:
            logger.warning(f"MAX DEFENSIVE PENALTY EXCEEDED: {-penalty}")
            return self.config['max_defensive_penalty']

        return -penalty

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        if not self.multiagent:
            return {
                "speed": self.vehicle.speed,
                "crashed": self.vehicle.crashed,
                "action": action,
                "rewards": self._rewards(action),
                "av_id": self.vehicle.av_id,
            }

        return {
            "speed": [v.speed for v in self.controlled_vehicles],
            "crashed": [v.crashed for v in self.controlled_vehicles],
            "action": action,
            "rewards": self._rewards(action),
            "av_ids": [v.av_id for v in self.controlled_vehicles],
        }

    def _reward(self, action: np.ndarray, rewards: dict | None = None) -> float | list[float]:
        if not self.multiagent:
            rewards = rewards or self._rewards(action)
            reward = sum(
                self.config.get(name, 0) * reward for name, reward in rewards.items()
            )
            worst, best = self.config["crash_penalty"], 1
            reward = utils.lmap(reward, [worst, best], [0, 1])
            reward *= rewards["on_road_reward"]

        else:
            rewards = rewards or self._rewards(action)
            # Accumulated reward for each vehicle
            reward = np.array(
                [
                    sum(reward_tup) for reward_tup in zip(
                    rewards["lane_centering_reward"], rewards["crash_reward"], rewards["on_road_reward"]
                )
                ]
            )
            if self.config["normalize_reward"]:
                worst, best = self.config['crash_penalty'], 1
                reward = utils.lmap(
                    reward,
                    [worst, best],
                    [0, 1]
                )

            # Multiply each by their on_road_reward
            reward *= rewards["on_road_reward"]

        return reward

    def _rewards(self, action: np.ndarray) -> dict:
        if self.multiagent:
            result = {
                "lane_centering_reward": [],
                "crash_reward": [],
                "on_road_reward": [],
            }
            for vehicle in self.controlled_vehicles:
                lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
                result["lane_centering_reward"].append(
                    1 / (1 + self.config["lane_centering_cost"] * lateral ** 2)
                )
                result["crash_reward"].append(self.crash_reward(vehicle))
                result["on_road_reward"].append(vehicle.on_road)

            return result
        else:
            _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
            return {
                "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
                # "action_reward": np.linalg.norm(action),
                "crash_reward": self.crash_reward(self.vehicle),
                "on_road_reward": self.vehicle.on_road,
            }

    def _is_terminated(self) -> bool | Array:
        """The episode is over if the ego vehicle crashed."""
        if not self.multiagent:
            return (self.vehicle.crashed or
                    self.config["offroad_terminal"] and not self.vehicle.on_road)

        return np.array(
            [v.crashed or self.config["offroad_terminal"] and not v.on_road for v in self.controlled_vehicles],
            dtype=int
        )

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
        for vehicle in mc_env.alter_vehicles:
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

    def set_vehicle_field(self, args: Tuple[str, object]) -> "AVRacetrack":
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v not in self.controlled_vehicles:
                setattr(v, field, value)
        return env_copy
