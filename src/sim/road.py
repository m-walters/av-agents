import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from highway_env.road.lane import AbstractLane, lane_from_config, LineType, StraightLane
from highway_env.vehicle.objects import Landmark
from highway_env.road.road import Road

import jax.numpy as jnp
from jax import jit


if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

from sim.vehicles.highway import IDMVehicle, MetaActionVehicle

AVVehicle = IDMVehicle | MetaActionVehicle


class APPROACHES(Enum):
    AHEAD = "front"
    BEHIND = "behind"
    APPROACHING = "approaching"
    RECEDING = "receding"


class AVRoadNetwork(object):
    """
    Override of RoadNetwork
    """
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # Pick next road according to planned route
        if route:
            if (
                    route[0][:2] == current_index[:2]
            ):  # We just finished the first step of the route, drop it.
                route.pop(0)
            if (
                    route and route[0][0] == _to
            ):  # Next road in route is starting at the end of current road.
                _, next_to, next_id = route[0]
            elif route:
                logger.warning(
                    "Route {} does not start after current road {}.".format(
                        route[0], current_index
                    )
                )

        # Compute current projected (desired) position
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        # If next route is not known
        if not next_to:
            # Pick the one with the closest lane to projected target position
            try:
                lanes_dists = [
                    (
                        next_to,
                        *self.next_lane_given_next_road(
                            _from, _to, _id, next_to, next_id, projected_position
                        ),
                    )
                    for next_to in self.graph[_to].keys()
                ]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, _id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id

    def next_lane_given_next_road(
        self,
        _from: str,
        _to: str,
        _id: int,
        next_to: str,
        next_id: int,
        position: np.ndarray,
    ) -> Tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(
                lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position)
            )
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in sorted(
                    [key for key in self.graph[node].keys() if key not in path]
            ):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [
            (lane_index[0], lane_index[1], i)
            for i in range(len(self.graph[lane_index[0]][lane_index[1]]))
        ]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (
                not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (
                not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if AVRoadNetwork.is_same_road(
                lane_index_2, lane_index_1, same_lane
        ) or AVRoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road(
                            (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                        )
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [
            lane for to in self.graph.values() for ids in to.values() for lane in ids
        ]

    def lanes_dict(self) -> Dict[str, AbstractLane]:
        return {
            (from_, to_, i): lane
            for from_, tos in self.graph.items()
            for to_, ids in tos.items()
            for i, lane in enumerate(ids)
        }

    @staticmethod
    def straight_road_network(
        lanes: int = 4,
        start: float = 0,
        length: float = 10000,
        angle: float = 0,
        speed_limit: float = 30,
        nodes_str: Optional[Tuple[str, str]] = None,
        net: Optional["AVRoadNetwork"] = None,
    ) -> "AVRoadNetwork":
        net = net or AVRoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            origin = rotation @ origin
            end = rotation @ end
            line_types = [
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE,
            ]
            net.add_lane(
                *nodes_str,
                StraightLane(
                    origin, end, line_types=line_types, speed_limit=speed_limit
                )
            )
        return net

    def position_heading_along_route(
        self,
        route: Route,
        longitudinal: float,
        lateral: float,
        current_lane_index: LaneIndex,
    ) -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :param current_lane_index: current lane index of the vehicle
        :return: position, heading
        """

        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # We know which road segment will be followed by the vehicle, but not which lane.
                # Hypothesis: the vehicle will keep the same lane_id as the current one.
                id_ = (
                    current_lane_index[2]
                    if current_lane_index[2]
                       < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                    else 0
                )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.integers(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict

import sys
class GPUAVRoad(object):
    """
    Road override

    Optimized with matrices etc.
    Note:
        - We will not have any self.objects
    """
    def __init__(
        self,
        network: AVRoadNetwork = None,
        vehicles: List["kinematics.Vehicle"] = None,
        road_objects: List["objects.RoadObject"] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

        # 2D Array of global vehicle positions
        # self.vehicle_positions = np.zeros((len(self.vehicles), 2))
        self.vehicle_positions = np.array([v.position for v in self.vehicles])
        print(f"MW POSNS -- {self.vehicle_positions}")
        sys.exit()

    def close_objects_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: Optional[int] = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        vehicles = [
            v
            for v in self.vehicles
            if np.linalg.norm(v.position - vehicle.position) < distance
               and v is not vehicle
               and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]
        obstacles = [
            o
            for o in self.objects
            if np.linalg.norm(o.position - vehicle.position) < distance
               and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)
        ]

        objects_ = vehicles if vehicles_only else vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]
        return objects_

    def close_vehicles_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: Optional[int] = None,
        see_behind: bool = True,
        sort: bool = True,
    ) -> object:
        return self.close_objects_to(
            vehicle, distance, count, see_behind, sort, vehicles_only=True
        )

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
            # for other in self.objects:
            #     vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(
        self, vehicle: "AVVehicle", lane_index: LaneIndex = None
    ) -> Tuple[Optional["kinematics.Vehicle"], Optional["kinematics.Vehicle"]]:
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
                        print(
                            f"MW SAMEROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                            f" {dist}"
                        )
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
                        print(
                            f"MW NEXT ROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                            f" {dist}"
                        )
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

    def __repr__(self):
        return self.vehicles.__repr__()

class OldAVRoad(Road):
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
                        print(
                            f"MW SAMEROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                            f" {dist}"
                        )
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
                        print(
                            f"MW NEXT ROAD [{v.av_id}] -- {rel_pos_enum} | {rel_vel_enum} | {rel_pos} | {rel_vel} |"
                            f" {dist}"
                        )
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
