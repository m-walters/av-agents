import copy
import logging
from typing import Optional, Tuple

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
        self.multiagent: bool = False
        self.alter_vehicles: list[AVVehicle] = []
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
                "crash_penalty": -1,  # The reward received when colliding with a vehicle.
                "max_defensive_penalty": -6,  # Cap the defensive reward/penalty
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
            "default_control_behavior": "sim.vehicles.highway.NominalParams",
            # Number of vehicles with the reward_speed target
            # Remainder will have a target speed lower than speed_limit
            "num_vehicles_control_speed": 8,
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
        if not self.config["normalize_reward"]:
            raise ValueError("Only normalized rewards configured")

        # Clear the vehicle lookup
        self.vehicle_lookup = {}

        self._create_road()
        self._create_vehicles()
        # self.multiagent = len(self.controlled_vehicles) > 1
        # Let's just force things to always be multiagent
        self.multiagent = True

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
        control_behavior = utils.class_from_path(self.config["default_control_behavior"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config[
                "controlled_vehicles"]
        )

        self.controlled_vehicles = []
        self.alter_vehicles = []
        # Number of alters that will also share the target reward speed
        num_alter_control_speed = self.config["num_vehicles_control_speed"] - self.config["controlled_vehicles"]
        av_id = 1
        for others in other_per_controlled:
            vehicle = control_vehicle_class.create_random(
                self.road,
                speed=0.8 * self.config["speed_limit"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                av_id=str(av_id),
                target_speed=self.config["reward_speed"]
            )
            # Imbue default behavior
            vehicle.set_behavior_params(control_behavior)
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
                if num_alter_control_speed > 0:
                    vehicle.target_speed = self.config["reward_speed"]
                    num_alter_control_speed -= 1
                else:
                    vehicle.target_speed = self.config["speed_limit"]

                if hasattr(vehicle, "randomize_behavior"):
                    vehicle.randomize_behavior()
                self.alter_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

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
        # logger.debug(f">>> SPEED REWARD: v_forward {forward_speed:0.4f} | score {score:0.4f}")

        if self.config['normalize_reward']:
            max_score = self.config['alpha']
            score /= max_score
            assert 0 <= score <= 1

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
            if self.config['normalize_reward']:
                return -1
            return self.config['max_defensive_penalty']

        # See our AV Project, "Vehicle Agent" section for derivation
        # In IDMVehicle.act you can see acceleration getting clipped by [-ACC_MAX, ACC_MAX]
        beta = 1 / (2 * self.ACC_MAX)

        n_nbr = 0
        penalty = 0
        # Boundary of neighbour consideration
        boundary = max(vehicle.speed, 10)
        for v in self.road.close_vehicles_to(vehicle, distance=boundary, count=None, see_behind=True, sort=False):
            lane = v.lane_index
            if abs(lane[2] - vehicle.lane_index[2]) > 1:
                # Don't worry about lanes that are more than 1 away
                continue

            # Determine if we are approaching this car either from behind or in front
            relative_pos = v.position - vehicle.position
            relative_velocity = v.velocity - vehicle.velocity
            dist = np.linalg.norm(relative_pos)

            # if vehicle.av_id == "1":
            #     print(
            #         f"MW DEFENCE COMPARE [{vehicle}, {v}]\n\t pos {vehicle.position} | {v.position} -> {relative_pos}"
            #         f"\n\t dir {vehicle.direction} | {v.direction} -> {v.direction - vehicle.direction}"
            #         f"\n\t vel {vehicle.velocity} | {v.velocity} -> {relative_velocity}"
            #         f"\n\t dist {dist:0.4f} | rel_pos . v.direction = {np.dot(relative_pos, v.direction):0.4f} |"
            #         f" rel_vel . vehicle.direction = {np.dot(relative_velocity, vehicle.direction):0.4f}"
            #     )

            if np.dot(relative_pos, vehicle.direction) < 0:
                # Behind
                # Compare relative velocity with our velocity to see if its approaching or not
                if np.dot(relative_velocity, vehicle.direction) > 0:
                    # Approaching from behind
                    ...
                else:
                    # Moving away, ignore
                    continue
            else:
                # In front
                # Compare relative velocity with our velocity to see if its approaching or not
                if np.dot(relative_velocity, vehicle.direction) < 0:
                    # Approaching from front
                    ...
                else:
                    # Moving away, ignore
                    continue

            # Found approacher
            n_nbr += 1
            dist = max(dist, vehicle.LENGTH * 1.)
            relative_speed = np.linalg.norm(relative_velocity)
            _penalty = beta * relative_speed ** 2 / (dist * (2 ** np.abs(lane[2] - vehicle.lane_index[2])))
            penalty += _penalty

        # Average over the neighbors
        # penalty /= n_nbr if n_nbr > 0 else 1

        if -penalty < self.config['max_defensive_penalty']:
            logger.warning(f"MAX DEFENSIVE PENALTY EXCEEDED: {-penalty}")
            penalty = -self.config['max_defensive_penalty']

        if self.config['normalize_reward']:
            # Penalty is in range [0, -max_defensive_penalty], return in range [0,-1]
            reward = -penalty / abs(self.config['max_defensive_penalty'])
            assert -1 <= reward <= 0
            return reward

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

    def _rewards(self, action: Action) -> dict:
        """
        Compute and collect the suite of rewards for our control vehicles.
        TODO -- Consider a speed-limit penalty (see IDMVehicle.accelerate)
        """
        if self.multiagent:
            return {
                "defensive_reward": [self.defensive_reward(v) for v in self.controlled_vehicles],
                "speed_reward": [self.speed_reward(v) for v in self.controlled_vehicles],
                "crash_reward": [self.crash_reward(v) for v in self.controlled_vehicles],
            }
        else:
            return {
                "defensive_reward": self.defensive_reward(self.vehicle),
                "speed_reward": self.speed_reward(self.vehicle),
                "crash_reward": self.crash_reward(self.vehicle),
            }

    def _reward(self, action: Action, rewards: dict | None = None) -> float | list[float]:
        """
        Scalar reward value

        :param action: the last action performed
        :param rewards: optionally provide the computed rewards if available, for efficiency
        :return: the corresponding reward
        """
        if not self.multiagent:
            rewards = rewards or self._rewards(action)
            reward = sum(rewards.values())

            if self.config["normalize_reward"]:
                # Best is 1 for the normalized speed reward
                # Worst is -2 for the normalized crash penalty and the normalized defensive penalty
                best = 1
                worst = -2
                reward = utils.lmap(
                    reward,
                    [worst, best],
                    [0, 1]
                )

                assert 0 <= reward <= 1

            return reward

        # Multi-agent rewards
        rewards = rewards or self._rewards(action)
        # Accumulated reward for each vehicle
        reward = np.array(
            [
                sum(reward_tup) for reward_tup in zip(
                rewards["defensive_reward"], rewards["speed_reward"], rewards["crash_reward"]
            )
            ]
        )
        if self.config["normalize_reward"]:
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

    @property
    def crashed(self) -> list:
        """
        Return crashed vehicles
        """
        return [v for v in self.road.vehicles if v.crashed]

    @property
    def off_road(self) -> list:
        """
        Return vehicles off the road
        """
        return [v for v in self.road.vehicles if not v.on_road]

    def _is_terminated(self) -> bool | Array:
        """The episode is over if the ego vehicle crashed."""
        # if not self.multiagent:
        #     return self.vehicle.crashed or self.config["offroad_terminal"] and not self.vehicle.on_road
        #
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

            losses[i] = 1 - reward

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

    def set_vehicle_field(self, args: Tuple[str, object]) -> "AVHighway":
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v not in self.controlled_vehicles:
                setattr(v, field, value)
        return env_copy
