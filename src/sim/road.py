from typing import List, Tuple

import numpy as np
from enum import Enum
from highway_env.road.road import Road
from highway_env.vehicle.objects import Landmark

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

from sim.vehicles.highway import IDMVehicle, MetaActionVehicle

AVVehicle = IDMVehicle | MetaActionVehicle


class APPROACHES(Enum):
    AHEAD = "front"
    BEHIND = "behind"
    APPROACHING = "approaching"
    RECEDING = "receding"


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

    @staticmethod
    def determine_approach(ref_v: AVVehicle, other_v: AVVehicle) \
            -> Tuple[APPROACHES, APPROACHES, np.ndarray, np.ndarray]:
        """
        Compare a reference and another vehicle and return if the vehicle is Ahead or Behind, and if
        it is Approaching or Receding
        Also return the relative position and velocity of the other vehicle, each a 2D array
        """
        relative_pos = other_v.position - ref_v.position
        relative_velocity = other_v.velocity - ref_v.velocity

        # if vehicle.av_id == "1":
        #     print(
        #         f"MW DEFENCE COMPARE [{vehicle}, {v}]\n\t pos {vehicle.position} | {v.position} -> {relative_pos}"
        #         f"\n\t dir {vehicle.direction} | {v.direction} -> {v.direction - vehicle.direction}"
        #         f"\n\t vel {vehicle.velocity} | {v.velocity} -> {relative_velocity}"
        #         f"\n\t dist {dist:0.4f} | rel_pos . other_v.direction = {np.dot(relative_pos, other_v.direction):0.4f} |"
        #         f" rel_vel . vehicle.direction = {np.dot(relative_velocity, vehicle.direction):0.4f}"
        #     )

        if np.dot(relative_pos, ref_v.direction) < 0:
            # Behind
            # Compare relative velocity with our velocity to see if its approaching or not
            if np.dot(relative_velocity, ref_v.direction) > 0:
                # Approaching from behind
                return APPROACHES.BEHIND, APPROACHES.APPROACHING, relative_pos, relative_velocity
            else:
                # Moving away from behind
                return APPROACHES.BEHIND, APPROACHES.RECEDING, relative_pos, relative_velocity
        else:
            # In front
            # Compare relative velocity with our velocity to see if its approaching or not
            if np.dot(relative_velocity, ref_v.direction) < 0:
                # Approaching in front
                return APPROACHES.AHEAD, APPROACHES.APPROACHING, relative_pos, relative_velocity
            else:
                # Moving away in front
                return APPROACHES.AHEAD, APPROACHES.RECEDING, relative_pos, relative_velocity

    def alt_neighbour_vehicles(
        self, vehicle: AVVehicle, lane_index: LaneIndex = None
    ) -> Tuple[AVVehicle | None, AVVehicle | None]:
        """
        Custom neighbour_vehicles override.
        This method does a hierarchical search in order:
            - Same road, same lane
                - L2 distance
            - Directly connected road, same lane
                - L2 distance
        That is, for all vehicles on same road, same lane, we take the nearest L2 distance.
        If not vehicle found there, we check the connected road and then take L2 on any of those vehicles

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
            if v is not vehicle and not isinstance(v, Landmark):
                # Check if same road
                same_road = self.network.is_same_road(v.lane_index, lane_index, same_lane=True)
                if same_road:
                    # Determine if in front or behind
                    rel_pos_enum, rel_vel_enum, rel_pos, rel_vel = self.determine_approach(vehicle, v)
                    dist = np.linalg.norm(rel_pos)
                    if vehicle.av_id == "1":
                        print(f"MW SAMEROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                              f" {dist}")
                    if rel_pos_enum == APPROACHES.AHEAD:
                        if not s_front or dist < s_front:
                            if vehicle.av_id == "1":
                                print(f"MW UPDATING FRONT {v.av_id} | {dist}")
                            s_front = dist
                            v_front = v
                    else:
                        # Vehicle is behind
                        if not s_rear or dist < s_rear:
                            if vehicle.av_id == "1":
                                print(f"MW UPDATING REAR {v.av_id} | {dist}")
                            s_rear = dist
                            v_rear = v
                    continue

                # Check upcoming lane
                next_road = self.network.is_leading_to_road(lane_index, v.lane_index, same_lane=True)
                if next_road:
                    rel_pos_enum, rel_vel_enum, rel_pos, rel_vel = self.determine_approach(vehicle, v)
                    dist = np.linalg.norm(rel_pos)
                    # We will assert that this is a front vehicle and ignore the enums
                    if vehicle.av_id == "1":
                        print(f"MW NEXT ROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                              f" {dist}")
                        print(f"MW UPCOMING POSNS -- {vehicle.position} | {v.position} | {rel_pos}")
                        print(f"MW UPCOMING VELS -- {vehicle.velocity} | {v.velocity} | {rel_vel}")
                    if not s_front or dist < s_front:
                        if vehicle.av_id == "1":
                            print(f"MW UPDATING FRONT {v.av_id} | {dist}")
                        s_front = dist
                        v_front = v
                    continue

                # Check previous lane
                prev_road = self.network.is_leading_to_road(v.lane_index, lane_index, same_lane=True)
                if prev_road:
                    rel_pos_enum, rel_vel_enum, rel_pos, rel_vel = self.determine_approach(vehicle, v)
                    dist = np.linalg.norm(rel_pos)
                    # We will assert that this is a rear vehicle and ignore the enums
                    if not s_rear or dist < s_rear:
                        s_rear = dist
                        v_rear = v
                    continue

        if vehicle.av_id == "1":
            print(f"MW FINAL DIST -- {s_rear} | {s_front}")
        return v_front, v_rear

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
            if v is not vehicle and not isinstance(v, Landmark):
                # connected_road = self.network.is_connected_road(v.lane_index, lane_index, same_lane=True)
                # if not connected_road:
                #     continue
                # f"MW I think this line is an issue by only checking the subject vehicle's lane
                s_v, lat_v = lane.local_coordinates(v.position)
                # MW -- HACK for our purposes. Since we are working with Rings,
                # any vehicle on the same line is technically front and behind
                # so we ignore this catch:
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear
