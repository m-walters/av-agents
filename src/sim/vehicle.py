from highway_env.vehicle.kinematics import Vehicle as HEVehicle
from scipy.stats import multivariate_normal
import numpy as np

class Vehicle(HEVehicle):
    """
    Vehicle override with ActInf.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief_state = np.zeros(4)  # [x, y, heading, speed]
        self.precision = np.eye(4)  # [Precision matrix] 
        self.generative_model = self._default_generative_model
        self.free_energy = 0

    
    def step(self, dt: float) -> None:
        """
        Propagate the current vehicle state using ActInf principles.
        """
        
        predicted_state = self.generative_model(self.belief_state, self.action)
        sensory_input = np.array([self.position[0], self.position[1], self.heading, self.speed])
        prediction_error = sensory_input - predicted_state # Updates the belief state using prediction error
        self.belief_state += np.dot(self.precision, prediction_error)

        self.free_energy = self._calculate_free_energy(sensory_input, predicted_state) # Updates Free Energy part
        self.action = self._active_inference_action_selection()
        
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

    def _simple_generative_model(self, state, action):
        """
        Simple generative model for state transition
        """
        new_state = state.copy()
        new_state[0] += np.cos(state[2]) * state[3] * 0.1  # x
        new_state[1] += np.sin(state[2]) * state[3] * 0.1  # y
        new_state[2] += action['steering'] * 0.1  # heading
        new_state[3] += action['acceleration'] * 0.1  # speed
        return new_state

    def _calculate_free_energy(self, sensory_input, predicted_state):
        """
        Calculate Free Energy
        """
        prediction_error = sensory_input - predicted_state
        return 0.5 * np.dot(prediction_error.T, np.dot(self.precision, prediction_error))

    """
    To Add : Action Selection for minimizing the expected free energy, State Update for updating lane, history & belief state ( For Belief State: array with position(0) & position(1)
    """

    def on_state_update(self) -> None:
        """
        For now, just updates the lane and history
        """
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))
