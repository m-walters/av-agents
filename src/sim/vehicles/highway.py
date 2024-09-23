from typing import Optional, Union

import numpy as np
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.behavior import AggressiveVehicle
from highway_env.vehicle.controller import ControlledVehicle


class VehicleBase:
    """ Vehicle length [m] """
    LENGTH = 5.0
    """ Vehicle width [m] """
    WIDTH = 2.0
    """ Range for random initial speeds [m/s] """
    DEFAULT_INITIAL_SPEEDS = [20, 30]
    """ Maximum reachable speed [m/s] """
    MAX_SPEED = 40.
    """ Minimum reachable speed [m/s] """
    MIN_SPEED = -20.
    """ Length of the vehicle state history, for trajectory display"""
    HISTORY_SIZE = 30


class AggressiveParams:
    """
    From the AggressiveVehicle class
    """
    # Longitudinal policy parameters
    """Maximum acceleration."""
    ACC_MAX = 6.0  # [m/s2]

    """Desired maximum acceleration."""
    COMFORT_ACC_MAX = 4.0  # [m/s2]

    """Desired maximum deceleration."""
    COMFORT_ACC_MIN = -5.0  # [m/s2]

    """Exponent of the velocity term."""
    DELTA = 4.0  # []

    """Range of delta when chosen randomly."""
    DELTA_RANGE = [3.5, 4.5]

    """Desired jam distance to the front vehicle."""
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]

    """Desired time gap to the front vehicle."""
    TIME_WANTED = 1.5  # [s]

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 1.0  # [s]; lower means more frequent lane checks. Default = 1.0

    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30


class HotshotParams(AggressiveParams):
    """
    Based off the AggressiveVehicle params, but reckless
    """
    # Longitudinal policy parameters
    ACC_MAX = 10.0  # [m/s2]
    COMFORT_ACC_MAX = ACC_MAX  # [m/s2]

    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH  # [m]
    TIME_WANTED = 0.3

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 10.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 1.0  # [s]; lower means more frequent lane checks. Default = 1.0


class IDMVehicle(HotshotParams, VehicleBase, AggressiveVehicle):
    """
    IDM Vehicle override.
    IDMVehicles don't respond to action inputs, but instead operate
    intelligently from their surroundings.
    """

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int | None = None,
        target_speed: float | None = None,
        route: Route | None = None,
        enable_lane_change: bool = True,
        timer: float | None = None,
        data: dict | None = None,
        av_id: str | None = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route,
            enable_lane_change, timer
        )
        self.data = data if data is not None else {}
        self.collecting_data = True
        # Our internal tracking ID
        self.av_id = av_id

    def act(self, action: Union[dict, str] = None):
        super().act(action)

    def randomize_behavior(self):
        """
        Call this method to initiate some randomization of behavior
        """
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (self.ACCELERATION_RANGE[1] -
                                                                          self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    @classmethod
    def create_random(
        cls, road: Road,
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 1,
        target_speed: Optional[float] = None,
        av_id: str | None = None,
    ) -> "IDMVehicle":
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
        :param target_speed: target speed in [m/s] the vehicle should reach
        :param av_id: the ID of the new vehicle
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
                speed = road.np_random.uniform(cls.DEFAULT_INITIAL_SPEEDS[0], cls.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12 + 1.0 * speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3 * offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(
            road, lane.position(x0, 0), heading=lane.heading_at(x0), speed=speed,
            target_speed=target_speed, av_id=av_id
        )
        return v

    def __str__(self):
        return f"IDMVehicle [#{id(self) % 1000}, AV-{self.av_id}]"


class MetaActionVehicle(ControlledVehicle):
    """
    Vehicle controlled by a discrete set of meta actions (see DiscreteMetaAction)
    """
    """ Vehicle length [m] """
    LENGTH = 5.0
    """ Vehicle width [m] """
    WIDTH = 2.0
    """ Range for random initial speeds [m/s] """
    DEFAULT_INITIAL_SPEEDS = [20, 30]
    """ Maximum reachable speed [m/s] """
    MAX_SPEED = 40.
    """ Minimum reachable speed [m/s] """
    MIN_SPEED = -20.
    """ Length of the vehicle state history, for trajectory display"""
    HISTORY_SIZE = 30

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int | None = None,
        target_speed: float | None = None,
        route: Route | None = None,
        av_id: str | None = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route,
        )
        # Our internal tracking ID
        self.av_id = av_id

    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 1,
        target_speed: float = None,
        av_id: str | None = None,
    ) -> "MetaActionVehicle":
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
        :param target_speed: target speed in [m/s]. If None, will be set to 'speed'
        :param av_id: Vehicle ID
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    cls.DEFAULT_INITIAL_SPEEDS[0], cls.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        offset = (
                spacing
                * default_spacing
                * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(
            road, lane.position(x0, 0), lane.heading_at(x0), speed, target_speed=target_speed, av_id=av_id
        )
        return v

    def __str__(self):
        return f"MetaActionVehicle [#{id(self) % 1000}, AV-{self.av_id}]"


AVVehicleType = IDMVehicle | MetaActionVehicle
