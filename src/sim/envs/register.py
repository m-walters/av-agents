from gymnasium.envs.registration import register


def register_envs():
    register(
        id=f"AVAgents/highway-v0",
        entry_point='sim.envs.highway:AVHighway',
    )
    register(
        id=f"AVAgents/intersection-v0",
        entry_point='sim.envs.intersection:AVIntersection',
    )
