"""
Deprecated classes for vehicle policies
"""
from highway_env.vehicle.controller import ControlledVehicle


class NominalParams:
    """
    From the AggressiveVehicle class
    """
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
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]

    """Desired time gap to the front vehicle."""
    TIME_WANTED = 1.5  # [s]

    # Lateral policy parameters
    # See IDMVehicle.mobil() for what these mainly do
    POLITENESS = 0.2  # in [0, 1]; Is added to a sum for proposed change that is sum(...) < LANE_CHANGE_MIN_ACC_GAIN
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    # Below: [m/s2]; larger means more likely to accept a lane change. Default = 2.0
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


class ConservativeParams(NominalParams):
    """
    From the DefensiveVehicle class
    """
    # Longitudinal policy parameters
    """Maximum acceleration."""
    ACC_MAX = 6.0  # [m/s2]

    """Desired maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]

    """Desired maximum deceleration."""
    COMFORT_ACC_MIN = -3.0  # [m/s2]

    """Exponent of the velocity term."""
    DELTA = 4.0  # []

    """Range of delta when chosen randomly."""
    DELTA_RANGE = [3.5, 4.5]

    """Desired jam distance to the front vehicle."""
    DISTANCE_WANTED = 8.0 + ControlledVehicle.LENGTH  # [m]

    """Desired time gap to the front vehicle."""
    TIME_WANTED = 2.0  # [s]

    # Lateral policy parameters
    POLITENESS = 0.8  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 2.0  # [s]; lower means more frequent lane checks. Default = 1.0

    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]


class HotshotParams(NominalParams):
    """
    Based off the AggressiveVehicle params, but reckless
    """
    # Longitudinal policy parameters
    ACC_MAX = 15.0  # [m/s2]
    COMFORT_ACC_MAX = ACC_MAX  # [m/s2]
    COMFORT_ACC_MIN = -4.0  # [m/s2]

    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH  # [m]
    TIME_WANTED = 0.3

    # Lateral policy parameters
    POLITENESS = 0.
    LANE_CHANGE_MIN_ACC_GAIN = 2.0  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 5.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 0.5  # [s]; lower means more frequent lane checks. Default = 1.0

    # # Lateral policy parameters
    # POLITENESS = 0.  # in [0, 1]
    # LANE_CHANGE_MIN_ACC_GAIN = 2.0  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 5.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    # LANE_CHANGE_DELAY = 2.0  # [s]; lower means more frequent lane checks. Default = 1.0


###########
# Making a suite of small changes for isolated comparison
class PolitenessIncr(NominalParams):
    POLITENESS = 1.0


class PolitenessDecr(NominalParams):
    POLITENESS = 0.0


class TimeDistWantedIncr(NominalParams):
    DISTANCE_WANTED = 7.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 3.0


class TimeDistWantedDecr(NominalParams):
    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 0.3


class AccMaxIncr(NominalParams):
    ACC_MAX = 10.0
    COMFORT_ACC_MAX = 6.0


class AccMaxDecr(NominalParams):
    ACC_MAX = 4.0
    COMFORT_ACC_MAX = 3.0


class ComfBrakingIncr(NominalParams):
    COMFORT_ACC_MIN = -6.0


class ComfBrakingDecr(NominalParams):
    COMFORT_ACC_MIN = -2.0


"""
Results of first trial indicate (relative to Nominal):
- Politeness decrease had a marginal increase in collisions (sample size of only 100 was not enough)
- TimeDist Decrease had a *Significantly* more collisions and earlier.
    - Strangely, TimeDist Increase was marginally worse tha nom, but within noise
- AccMax: Trend showed safer with *increaseing* AccMax -- reasoning might be because of increased control/options.
- Braking Comf had almost no effect

In conclusion, TimeDist had the strongest effect

"""


class ReckMax1(NominalParams):
    """Reckless driver 1"""
    POLITENESS = 0
    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 0.3
    ACC_MAX = 6.0


class ReckMax2(NominalParams):
    """Reckless driver 2: Increased ACC-MAX"""
    POLITENESS = 0
    DISTANCE_WANTED = 1.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 0.3
    ACC_MAX = 10.0


class ReckMax3(ReckMax2):
    """Reckless driver 3: Increased lane-changing"""
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 10.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 1.0  # [s]; lower means more frequent lane checks. Default = 1.0


class DefensiveHE(NominalParams):
    """Defensive driver HE: Based on highway-env class"""
    TIME_WANTED = 2.5

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]


class Defensive1(NominalParams):
    """Defensive driver 2: More intensely safe"""
    POLITENESS = 0.8
    DISTANCE_WANTED = 7.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 3.0


class Defensive2(Defensive1):
    """Defensive driver 2: Like 1 but more lane changing"""
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]; larger means more likely to accept a lane change. Default = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 10.0  # [m/s2]; larger means more likely to accept a lane change. Default = 2.0
    LANE_CHANGE_DELAY = 1.0  # [s]; lower means more frequent lane checks. Default = 1.0
