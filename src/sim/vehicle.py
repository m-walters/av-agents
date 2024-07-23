from highway_env.vehicle.kinematics import Vehicle as HEVehicle

import numpy as np

class Vehicle(HEVehicle):
    """
    Vehicle override for our purposes
    """

    """ Vehicle length [m] """
    LENGTH = 5.0
    """ Vehicle width [m] """
    WIDTH = 2.0
    """ Range for random initial speeds [m/s] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Maximum reachable speed [m/s] """
    MAX_SPEED = 40.
    """ Minimum reachable speed [m/s] """
    MIN_SPEED = -40.
    """ Length of the vehicle state history, for trajectory display"""
    HISTORY_SIZE = 30

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
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])

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
