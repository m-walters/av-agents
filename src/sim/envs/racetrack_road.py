import numpy as np
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


def racetrack_road1(racetrack: "RacetrackEnv") -> Road:

    net = RoadNetwork()
    w = 5
    w2 = 2 * w
    default_speedlimit = racetrack.config['speed_limit']

    # Initialise First Lane
    lane = StraightLane(
        [42, 0],
        [200, 0],
        line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        width=w,
        speed_limit=default_speedlimit,
    )
    racetrack.lane = lane

    # Add Lanes to Road Network - Straight Section
    net.add_lane("a", "b", lane)
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [42, w],
            [200, w],
            line_types=(LineType.STRIPED, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [42, w2],
            [200, w2],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )

    # 2 - Circular Arc #1
    center1 = [200, -20]
    radii1 = 20
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center1,
            radii1,
            np.deg2rad(90),
            np.deg2rad(-1),
            width=w,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center1,
            radii1 + w,
            np.deg2rad(90),
            np.deg2rad(-1),
            width=w,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center1,
            radii1 + w2,
            np.deg2rad(90),
            np.deg2rad(-1),
            width=w,
            clockwise=False,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=default_speedlimit,
        ),
    )

    # 3 - Vertical Straight
    delta_extension = -1.  # Better join
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [220, -20],
            [220, -60 + delta_extension],
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [220 + w, -20],
            [220 + w, -60 + delta_extension],
            line_types=(LineType.STRIPED, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [220 + w2, -20],
            [220 + w2, -60 + delta_extension],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )

    # 4 - Circular Arc #2
    center4 = [205, -60]
    radii4 = 15
    net.add_lane(
        "d",
        "e",
        CircularLane(
            center4,
            radii4,
            np.deg2rad(0),
            np.deg2rad(-181),
            width=w,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "d",
        "e",
        CircularLane(
            center4,
            radii4 + w,
            np.deg2rad(0),
            np.deg2rad(-181),
            width=w,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "d",
        "e",
        CircularLane(
            center4,
            radii4 + w2,
            np.deg2rad(0),
            np.deg2rad(-181),
            width=w,
            clockwise=False,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=default_speedlimit,
        ),
    )

    # 5 - Circular Arc #3
    center5 = [170, -60]
    radii5 = 15
    net.add_lane(
        "e",
        "f",
        CircularLane(
            center5,
            radii5 + 5,
            np.deg2rad(0),
            np.deg2rad(136),
            width=w,
            clockwise=True,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "e",
        "f",
        CircularLane(
            center5,
            radii5,
            np.deg2rad(0),
            np.deg2rad(137),
            width=w,
            clockwise=True,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "e",
        "f",
        CircularLane(
            center5,
            radii5 - w,
            np.deg2rad(0),
            np.deg2rad(137),
            width=w,
            clockwise=True,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=default_speedlimit,
        ),
    )

    # 6 - Slant
    # Extending [-30,-30]
    extend = np.array([-30, -30])
    start6 = np.array([155.7, -45.7])
    end6 = np.array([135.7, -65.7]) + extend
    net.add_lane(
        "f",
        "g",
        StraightLane(
            start6,
            end6,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    start6_2 = np.array([159.3934, -49.2])
    end6_2 = np.array([139.3934, -69.2]) + extend
    net.add_lane(
        "f",
        "g",
        StraightLane(
            start6_2,
            end6_2,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    start6_3 = np.array(
        [
            start6[0] + 2 * (start6_2[0] - start6[0]),
            start6[1] + 2 * (start6_2[1] - start6[1])
        ]
    )
    end6_3 = np.array(
        [
            end6[0] + 2 * (end6_2[0] - end6[0]),
            end6[1] + 2 * (end6_2[1] - end6[1]),
        ]
    )
    net.add_lane(
        "f",
        "g",
        StraightLane(
            start6_3,
            end6_3,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )

    # 7 - Circular Arc #4
    # Reflect it with the slant
    center7 = np.array([118.1, -48.1]) + extend
    radii7 = 25
    theta7 = 317
    # theta7_end = 270 - (theta7 - 270) - 10
    theta7_end = 205
    net.add_lane(
        "g",
        "h",
        CircularLane(
            center7,
            radii7,
            np.deg2rad(theta7),
            np.deg2rad(theta7_end - 3),  # nicer
            width=w,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "g",
        "h",
        CircularLane(
            center7,
            radii7 + 5,
            np.deg2rad(theta7),
            np.deg2rad(theta7_end),
            width=w,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "g",
        "h",
        CircularLane(
            center7,
            radii7 + w2,
            np.deg2rad(theta7),
            np.deg2rad(theta7_end),
            width=w,
            clockwise=False,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=default_speedlimit,
        ),
    )

    # 8 - Next slant
    # Reflected from the last arc's center
    start8 = np.array(
        [
            center7[0] + radii7 * np.cos(np.deg2rad(theta7_end)),
            center7[1] + radii7 * np.sin(np.deg2rad(theta7_end))
        ]
    )
    start8_2 = np.array(
        [
            center7[0] + (radii7 + w) * np.cos(np.deg2rad(theta7_end)),
            center7[1] + (radii7 + w) * np.sin(np.deg2rad(theta7_end))
        ]
    )
    start8_3 = np.array(
        [
            center7[0] + (radii7 + w2) * np.cos(np.deg2rad(theta7_end)),
            center7[1] + (radii7 + w2) * np.sin(np.deg2rad(theta7_end))
        ]
    )

    # We preemptively take section 9's radius to make a nice join.
    radii9 = 15
    rad = np.deg2rad(30)
    end8 = np.array(
        [
            42 - radii9 * np.cos(rad),
            -radii9 - radii9 * np.sin(rad)
        ]
    )
    end8_2 = np.array(
        [
            42 - (radii9 + w) * np.cos(rad),
            -radii9 - (radii9 + w) * np.sin(rad)
        ]
    )
    end8_3 = np.array(
        [
            42 - (radii9 + w2) * np.cos(rad),
            -radii9 - (radii9 + w2) * np.sin(rad)
        ]
    )
    net.add_lane(
        "h",
        "i",
        StraightLane(
            start8,
            end8,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "h",
        "i",
        StraightLane(
            start8_2,
            end8_2,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "h",
        "i",
        StraightLane(
            start8_3,
            end8_3,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            width=w,
            speed_limit=default_speedlimit,
        ),
    )

    # 9 - Circular arc 7, end
    # Since y2 = 0...
    center9 = np.array([42, -radii9])
    net.add_lane(
        "i",
        "a",
        CircularLane(
            center9,
            radii9,
            np.deg2rad(210),
            np.deg2rad(88),
            width=w,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "i",
        "a",
        CircularLane(
            center9,
            radii9 + w,
            np.deg2rad(210),
            np.deg2rad(90),
            width=w,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=default_speedlimit,
        ),
    )
    net.add_lane(
        "i",
        "a",
        CircularLane(
            center9,
            radii9 + w2,
            np.deg2rad(212),
            np.deg2rad(88),  # nicer join
            width=w,
            clockwise=False,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=default_speedlimit,
        ),
    )

    return Road(
        network=net,
        np_random=racetrack.np_random,
        record_history=racetrack.config["show_trajectories"],
    )
