import logging
from typing import Optional

import numpy as np
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.behavior import AggressiveVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Obstacle, RoadObject

from sim.plotting import AV_COLORS

logger = logging.getLogger("av-sim")


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


class NominalParams:
    """
    Somewhere between Defensive and Hotshot
    """
    COLOR = tuple(int(c * 255) for c in AV_COLORS["nominal"])

    # Longitudinal policy parameters
    """Maximum acceleration."""
    ACC_MAX = 6.0  # [m/s2]

    """Desired maximum acceleration."""
    COMFORT_ACC_MAX = 4.0  # [m/s2]

    """Desired maximum deceleration."""
    COMFORT_ACC_MIN = -4.0  # [m/s2]

    """Exponent of the velocity term."""
    DELTA = 4.0  # []

    """Range of delta when chosen randomly."""
    DELTA_RANGE = [3.5, 4.5]

    """Desired jam distance to the front vehicle."""
    DISTANCE_WANTED = 2.0 + ControlledVehicle.LENGTH  # [m]

    """Desired time gap to the front vehicle."""
    TIME_WANTED = .6  # [s]

    # Lateral policy parameters
    # See IDMVehicle.mobil() for what these mainly do
    # in [0, 1]; Is added to the 'jerk' sum, which is compared as jerk < LANE_CHANGE_MIN_ACC_GAIN;
    POLITENESS = 0.3

    # [m/s2]; smaller means more likely to accept a lane change. Default = 0.2
    # Proposed change will be rejected if 'jerk' is less than this value
    LANE_CHANGE_MIN_ACC_GAIN = 0.4

    # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    # Value a threshold of how much braking you care to impose on other vehicles for a mobil change.
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0

    LANE_CHANGE_DELAY = 1.0  # [s]; lower means more frequent lane checks. Default = 1.0

    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30

    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        0.5,
    ]


class DefensiveParams(NominalParams):
    """
    - Increased politeness
    - Less lane-changing
    - Increased TimeDist
    """
    COLOR = tuple(int(c * 255) for c in AV_COLORS["defensive"])

    POLITENESS = 0.5
    DISTANCE_WANTED = 3.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 1.5


# class DefensiveParams2(NominalParams):
#     """
#     Try to be even safer than DefensiveParams
#     """
#     POLITENESS = 0.5
#     DISTANCE_WANTED = 6.0 + ControlledVehicle.LENGTH
#     TIME_WANTED = 2.0


class HotshotParams(NominalParams):
    """
    Based on the old ReckMax1 policy

    - Decreased politeness
    - More lane-changing
    - Decreased TimeDist
    """
    COLOR = tuple(int(c * 255) for c in AV_COLORS["hotshot"])

    POLITENESS = 0.
    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 0.3


class AlterParams(NominalParams):
    """
    Nominal, but they dgaf about lane change risk.
    """
    COLOR = tuple(int(c * 255) for c in AV_COLORS["alter"])

    POLITENESS = 0.
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 20.0
    #
    # TIME_WANTED = 2.0


class IDMVehicle(NominalParams, VehicleBase, AggressiveVehicle):
    """
    IDM Vehicle override.
    IDMVehicles don't respond to action inputs, but instead operate
    intelligently from their surroundings.
    """
    # COLOR = NominalParams.COLOR
    CRASH_COOLDOWN = 0.2  # seconds

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
        self.crashed_timer = 0
        self.color = self.COLOR

    def set_behavior_params(self, param_class, color = None):
        """
        Given one of the param classes above, change our vehicle's values
        """
        self.ACC_MAX = param_class.ACC_MAX
        self.COMFORT_ACC_MAX = param_class.COMFORT_ACC_MAX
        self.COMFORT_ACC_MIN = param_class.COMFORT_ACC_MIN
        self.DELTA = param_class.DELTA
        self.DELTA_RANGE = param_class.DELTA_RANGE
        self.DISTANCE_WANTED = param_class.DISTANCE_WANTED
        self.TIME_WANTED = param_class.TIME_WANTED
        self.POLITENESS = param_class.POLITENESS
        self.LANE_CHANGE_MIN_ACC_GAIN = param_class.LANE_CHANGE_MIN_ACC_GAIN
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = param_class.LANE_CHANGE_MAX_BRAKING_IMPOSED
        self.LANE_CHANGE_DELAY = param_class.LANE_CHANGE_DELAY
        self.MERGE_ACC_GAIN = param_class.MERGE_ACC_GAIN
        self.MERGE_VEL_RATIO = param_class.MERGE_VEL_RATIO
        self.MERGE_TARGET_VEL = param_class.MERGE_TARGET_VEL
        # Color
        self.color = color or param_class.COLOR

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

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        # From IDMVehicle
        self.timer += dt

        # From Vehicle
        self.clip_actions()
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self._set_crash()
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()

        # Reduce the crash cooldown
        if self.crashed:
            self.crashed_timer = self.crashed_timer - dt
            if self.crashed_timer <= 0:
                self._recover_from_crash()

    def _set_crash(self):
        if not self.crashed:
            self.crashed = True
            self.crashed_timer = self.CRASH_COOLDOWN
            self.color = VehicleGraphics.RED
            self.speed = 0

    def _recover_from_crash(self):
        """
        Try and start driving again
        """
        # Check that there's enough space in front
        front_vehicle, _ = self.road.neighbour_vehicles(self, self.lane_index)
        if front_vehicle and self.lane_distance_to(front_vehicle) < self.LENGTH:
            # Not safe to start driving again
            return

        if not front_vehicle:
            # Check upcoming lane
            next_lane = self.road.network.next_lane(self.lane_index, position=self.position)
            nxt_front, _ = self.road.neighbour_vehicles(self, next_lane)
            if nxt_front and self.lane_distance_to(nxt_front) < self.LENGTH:
                # Not safe to start driving again
                return

        self.crashed = False
        self.impact = None
        self.crashed_timer = 0
        self.color = self.COLOR

    def clip_actions(self) -> None:
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed
        self.action["steering"] = float(self.action["steering"])
        self.action["acceleration"] = float(self.action["acceleration"])
        if self.speed > self.MAX_SPEED:
            self.action["acceleration"] = min(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )
        elif self.speed < self.MIN_SPEED:
            self.action["acceleration"] = max(
                self.action["acceleration"], 1.0 * (self.MIN_SPEED - self.speed)
            )

    def handle_collisions(self, other: "RoadObject", dt: float = 0) -> None:
        """
        Method override to handle the crashed-timer.

        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                if isinstance(other, Obstacle):
                    self.impact = transition
                elif isinstance(self, Obstacle):
                    other.impact = transition
                else:
                    self.impact = transition / 2
                    other.impact = -transition / 2
        if intersecting:
            if self.solid and other.solid:
                self._set_crash()
                try:
                    other._set_crash()
                except AttributeError:
                    other.crashed = True
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True

    def __str__(self):
        return f"IDMVehicle [#{id(self) % 1000}, AV-{self.av_id}]"

    def __repr__(self):
        return self.__str__()


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

    COLOR = VehicleGraphics.GREEN

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
        self.color = self.COLOR

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

    def __repr__(self):
        return self.__str__()


class AlterIDMVehicle(AlterParams, IDMVehicle):
    """
    Distinguish the alter class for visualizations and any other configuring
    """
    # COLOR = VehicleGraphics.BLUE

    def __str__(self):
        return f"AlterIDMVehicle [#{id(self) % 1000}, AV-{self.av_id}]"

    def __repr__(self):
        return self.__str__()


AVVehicleType = IDMVehicle | MetaActionVehicle
