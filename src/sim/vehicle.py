from typing import Optional

import numpy as np
from highway_env import utils
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle as HEVehicle


class Vehicle(LinearVehicle):
    """
    IDMVehicle override for our purposes

    Since technically every vehicle is "autonomous" and operated by some policy, we don't need to
    to distinguish classes between AVs and human drivers -- we simply give human drivers a human policy etc.
    """
    """ Vehicle length [m] """
    LENGTH = 5.0
    """ Vehicle width [m] """
    WIDTH = 2.0
    """ Range for random initial speeds [m/s] """
    DEFAULT_INITIAL_SPEEDS = [40, 50]
    """ Maximum reachable speed [m/s] """
    MAX_SPEED = 100.
    """ Minimum reachable speed [m/s] """
    MIN_SPEED = -20.
    """ Length of the vehicle state history, for trajectory display"""
    HISTORY_SIZE = 30

    # Longitudinal policy parameters
    """Maximum acceleration."""
    ACC_MAX = 6.0  # [m/s2]

    """Desired maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]

    """Desired maximum deceleration."""
    COMFORT_ACC_MIN = -5.0  # [m/s2]

    """Exponent of the velocity term."""
    DELTA = 4.0  # []

    """Range of delta when chosen randomly."""
    DELTA_RANGE = [3.5, 4.5]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        data: dict = None
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route,
            enable_lane_change, timer
        )
        self.data = data if data is not None else {}
        self.collecting_data = True

    def randomize_behavior(self):
        """
        Call this method to initiate some randomization of behavior
        """
        self.DELTA = np.random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    def sample_action(self, obs) -> Optional[dict]:
        """
        Sample an action given an observation
        """
        # The IDMVehicle (which LinearVehicle inherits) has actions automated, so we return null
        return None

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        # Normalize the current action
        self.clip_actions()

        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta),
                np.sin(self.heading + beta)]
        )

        # Update its position and compute other logic
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

    def on_state_update(self) -> None:
        """
        For now, just updates the lane and history
        """
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    @classmethod
    def create_random(
        cls, road: Road,
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 1
    ) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7 * lane.speed_limit, 0.8 * lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12 + 1.0 * speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3 * offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: HEVehicle = None,
        rear_vehicle: HEVehicle = None
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, HEVehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(
            max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)), self.DELTA
        ))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration
