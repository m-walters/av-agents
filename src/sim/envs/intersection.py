"""
Intersection environment
"""
import copy
import logging
from typing import Dict, Optional, Text, Tuple

import numpy as np
from gymnasium.utils import seeding
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from omegaconf import DictConfig

from sim.utils import Array
from sim.vehicles.intersection import Vehicle as AVVehicle

logger = logging.getLogger("av-sim")

Observation = np.ndarray


def make_road(env: "AVIntersection") -> None:
    """
    Make an 4-way intersection.

    The horizontal road has the right of way. More precisely, the levels of priority are:
        - 3 for horizontal straight lanes and right-turns
        - 1 for vertical straight lanes and right-turns
        - 2 for horizontal left-turns
        - 0 for vertical left-turns

    The code for nodes in the road network is:
    (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

    :return: the intersection road
    """
    lane_width = AbstractLane.DEFAULT_WIDTH
    right_turn_radius = lane_width + 5  # [m}
    left_turn_radius = right_turn_radius + lane_width  # [m}
    outer_distance = right_turn_radius + lane_width / 2
    access_length = 50 + 50  # [m]

    net = RoadNetwork()
    n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
    for corner in range(4):
        angle = np.radians(90 * corner)
        is_horizontal = corner % 2
        priority = 3 if is_horizontal else 1
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        # Incoming
        start = rotation @ np.array(
            [lane_width / 2, access_length + outer_distance]
        )
        end = rotation @ np.array([lane_width / 2, outer_distance])
        net.add_lane(
            "o" + str(corner),
            "ir" + str(corner),
            StraightLane(
                start, end, line_types=[s, c], priority=priority, speed_limit=10
            ),
        )
        # Right turn
        r_center = rotation @ (np.array([outer_distance, outer_distance]))
        net.add_lane(
            "ir" + str(corner),
            "il" + str((corner - 1) % 4),
            CircularLane(
                r_center,
                right_turn_radius,
                angle + np.radians(180),
                angle + np.radians(270),
                line_types=[n, c],
                priority=priority,
                speed_limit=10,
            ),
        )
        # Left turn
        l_center = rotation @ (
            np.array(
                [
                    -left_turn_radius + lane_width / 2,
                    left_turn_radius - lane_width / 2,
                ]
            )
        )
        net.add_lane(
            "ir" + str(corner),
            "il" + str((corner + 1) % 4),
            CircularLane(
                l_center,
                left_turn_radius,
                angle + np.radians(0),
                angle + np.radians(-90),
                clockwise=False,
                line_types=[n, n],
                priority=priority - 1,
                speed_limit=10,
            ),
        )
        # Straight
        start = rotation @ np.array([lane_width / 2, outer_distance])
        end = rotation @ np.array([lane_width / 2, -outer_distance])
        net.add_lane(
            "ir" + str(corner),
            "il" + str((corner + 2) % 4),
            StraightLane(
                start, end, line_types=[s, n], priority=priority, speed_limit=10
            ),
        )
        # Exit
        start = rotation @ np.flip(
            [lane_width / 2, access_length + outer_distance], axis=0
        )
        end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
        net.add_lane(
            "il" + str((corner - 1) % 4),
            "o" + str((corner - 1) % 4),
            StraightLane(
                end, start, line_types=[n, c], priority=priority, speed_limit=10
            ),
        )

    road = RegulatedRoad(
        network=net,
        np_random=env.np_random,
        record_history=env.config["show_trajectories"],
    )
    env.road = road


class AVIntersection(IntersectionEnv):
    """
    Override the IntersectionEnv class for our purposes
    """
    ACC_MAX = AVVehicle.ACC_MAX
    VEHICLE_MAX_SPEED = AVVehicle.MAX_SPEED
    PERCEPTION_DISTANCE = 5 * AVVehicle.MAX_SPEED

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9]
                },
                "duration": 13,  # [s]
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                # "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False
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
            "show_trajectories": False,
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

    def _make_road(self) -> None:
        """
        Use the custom road maker
        """
        make_road(self)

    def _reset(self) -> None:
        """
        Init road and vehicles
        """
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

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

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        longitudes = np.linspace(0, 80, n_vehicles)
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(float(longitudes[t]))
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in
                range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            # f"MW TODO -- Consider the 'create_random' and 'randomize_vehicle' approach use in
            #  `AVHighway._create_vehicles`
            ego_vehicle = AVVehicle(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60)
            )
            # ego_vehicle = self.action_type.vehicle_class(
            #                  self.road,
            #                  ego_lane.position(60 + 5*self.np_random.normal(1), 0),
            #                  speed=ego_lane.speed_limit,
            #                  heading=ego_lane.heading_at(60))
            try:
                # f"MW TODO -- This indicates the function may be expecting/requiring an MDPVehicle class,
                #  yet the silent exception suggests otherwise
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        We override the super-method because self._info calculates rewards, so we don't need to
        do it again with the self._reward call.

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

        # Functionality for IntersectionEnv
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

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

    def _reward(self, action: int, rewards: dict | None = None) -> float:
        """
        Aggregated average of rewards across controlled vehicles
        """
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        """
        Multi-objective rewards, for cooperative agents
        """
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        Per-agent reward signal
        """
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """
        Per-agent per-objective reward signal
        """
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            # "collision_reward": vehicle.crashed,
            "speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
            ### Custom
            "collision_reward": self.collision_reward(vehicle),
        }

    def collision_reward(self, vehicle: Vehicle) -> float:
        """
        Reward based on collision-related factors, like braking distance, and proximity

        :return: collision reward
        """
        v_x = vehicle.speed * np.cos(vehicle.heading)

        # Grab the front and behind cars across different lanes and consider their speed deltas from ego
        logger.debug(
            f">>> COLLISION REWARD START | EGO [{vehicle}]\n\tv_x {v_x:0.4f} | lane {vehicle.lane_index}"
        )

        # See our AV Project, "Vehicle Agent" section for derivation
        # beta = 3 * self.ACC_MAX / 2
        beta = 1 / (2 * self.ACC_MAX)
        logger.debug(f">> BETA: {beta}")

        n_nbr = 0
        penalty = 0
        for lane in self.road.network.all_side_lanes(vehicle.lane_index):
            logger.debug(f"CHECKING LANE {lane}")
            front, rear = self.road.neighbour_vehicles(vehicle, lane)
            if front is not None:
                logger.debug(f"FOUND FRONT [{front}]")
                n_nbr += 1
                front_speed = front.speed * np.cos(front.heading)
                front_delta = front_speed - v_x
                if front_delta < 0:
                    # Front car is slower
                    # Note, lane_distance_to: <argument>.x - self.x
                    front_distance = vehicle.lane_distance_to(front)
                    assert front_distance > 0, f"Front distance <= 0: {front_distance}"
                    front_distance = max(front_distance, vehicle.LENGTH)
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
                    rear_distance = max(rear_distance, rear.LENGTH)
                    _penalty = beta * rear_delta ** 2 / (
                            rear_distance * (2 ** np.abs(lane[2] - vehicle.lane_index[2])))
                    penalty += _penalty
                    logger.debug(f">> {rear_speed:0.4f} | {rear_delta:0.4f} | {rear_distance:0.4f} | {_penalty:0.4f}")

        logger.debug(f"COLLISION PENALTY: {penalty}\n")
        # Average over the neighbors
        penalty /= n_nbr if n_nbr > 0 else 1

        return -penalty

    def simplify(self) -> "AVIntersection":
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """

        distance = self.PERCEPTION_DISTANCE
        # Add to the distance the expected amount traveled by the car after mc_horizon steps at the given car speed
        # Since policy_frequency is how many calls to step per second and mc_horizon is num steps per mc sim..
        distance += self.VEHICLE_MAX_SPEED * self.config["mc_horizon"] / self.config["policy_frequency"]

        state_copy: "AVIntersection" = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, distance
        )

        return state_copy

    @staticmethod
    def seed_montecarlo(mc_env: "AVIntersection", seed):
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
        # f"MW TODO -- Review for multiagent setting
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
