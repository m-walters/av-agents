from typing import List, Tuple

import numpy as np
from highway_env.road.road import Road
from highway_env.vehicle.objects import Landmark

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

from sim.vehicles.highway import IDMVehicle, MetaActionVehicle

AVVehicle = IDMVehicle | MetaActionVehicle


class AVRoad(Road):
    """
    Road override
    """

    def close_vehicles_to(
        self, vehicle: AVVehicle, distance: float, count: int | None = None,
        see_behind: bool = True, sort: bool = True
    ) -> object:
        """
        Get vehicles within a certain range, with a few different controls
        """
        vehicles = [v for v in self.vehicles
            if np.linalg.norm(v.position - vehicle.position) < distance
               and v is not vehicle
               and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        if sort:
            vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(self, vehicle: AVVehicle, lane_index: LaneIndex = None) \
            -> Tuple[AVVehicle | None, AVVehicle | None]:
        """
        Find the preceding and following vehicles of a given vehicle.
        Note that this returns single vehicles (front, rear), not vectors

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear
