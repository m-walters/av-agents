"""
One-off script for running multiple runs
"""
import os
import subprocess

CONTROL_BEHAVIORS = {
    "nom": "sim.vehicles.highway.NominalParams",
    "cons": "sim.vehicles.highway.ConservativeParams",
    "def": "sim.vehicles.highway.DefensiveParams",
    "hotshot": "sim.vehicles.highway.HotshotParams",
    "polite-incr": "sim.vehicles.highway.PolitenessIncr",
    "polite-decr": "sim.vehicles.highway.PolitenessDecr",
    "timedist-incr": "sim.vehicles.highway.TimeDistWantedIncr",
    "timedist-decr": "sim.vehicles.highway.TimeDistWantedDecr",
    "acc-max-incr": "sim.vehicles.highway.AccMaxIncr",
    "acc-max-decr": "sim.vehicles.highway.AccMaxDecr",
    "comf-brake-incr": "sim.vehicles.highway.ComfBrakingIncr",
    "comf-brake-decr": "sim.vehicles.highway.ComfBrakingDecr",
    "reckmax1": "sim.vehicles.highway.ReckMax1",
    "reckmax2": "sim.vehicles.highway.ReckMax2",
    "reckmax3": "sim.vehicles.highway.ReckMax3",
    "def-HE": "sim.vehicles.highway.DefensiveHE",
    "def-1": "sim.vehicles.highway.Defensive1",
    "def-2": "sim.vehicles.highway.Defensive2",
}


def run_baselines():
    seed = 86777
    script = "ttc.py"

    num_cpus = 16
    env_type = "racetrack-v0"
    any_control_collision = "true"
    world_draws = 500
    duration = 200
    warmup = 1e5
    num_control_vehicles = 8
    num_vehicles_control_speed = 8
    vehicles_count = 24
    # Profiling
    profiling = "false"

    # Name the run, and where it will be saved
    RUN_DIR = "manuscript/av-8-baselines"

    policies = [
        "nom", "def", "hotshot",
    ]

    configs = [
        {
            "name": os.path.join(RUN_DIR, policy),
            "seed": seed,
            "highway_env.default_control_behavior": CONTROL_BEHAVIORS[policy],
            "highway_env.controlled_vehicles": num_control_vehicles,
            "highway_env.num_vehicles_control_speed": num_vehicles_control_speed,
            "highway_env.vehicles_count": vehicles_count,
            "any_control_collision": any_control_collision,
            "multiprocessing_cpus": num_cpus,
            "env_type": env_type,
            "highway_env.duration": duration,
            "world_draws": world_draws,
            "warmup_steps": warmup,
            "profiling": profiling,

        } for policy in policies
    ]

    for config in configs:
        # Convert the config dict into a string
        cfg_args = " ".join([f"{k}={v}" for k, v in config.items()])
        command = f"python {script} {cfg_args}"
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            break


def run_gk():
    """
    We're going to assume time-discounting is active
    """
    seed = 86777
    script = "ttc.py"

    num_cpus = 12
    env_type = "racetrack-v0"
    any_control_collision = "true"
    default_control_behavior = "sim.vehicles.highway.HotshotParams"
    world_draws = 100
    duration = 100
    warmup = 0
    num_control_vehicles = 8
    num_vehicles_control_speed = 8
    vehicles_count = 24
    # GK/MC Stuff
    mc_period = 5
    mc_horizon = 20
    n_montecarlo = 100
    enable_time_discounting = "true"
    # Paradoxically our GK 'nominal' is the risky one
    nominal_class = "sim.vehicles.highway.HotshotParams"
    defensive_class = "sim.vehicles.highway.DefensiveParams"

    # Profiling
    profiling = "false"

    # Name the run, and where it will be saved
    RUN_DIR = "hotshot-gk-sanity-test"
    name = "hotshot-gk"

    configs = [
        {
            "name": os.path.join(RUN_DIR, name),
            "seed": seed,
            "highway_env.default_control_behavior": default_control_behavior,
            "highway_env.controlled_vehicles": num_control_vehicles,
            "highway_env.num_vehicles_control_speed": num_vehicles_control_speed,
            "highway_env.vehicles_count": vehicles_count,
            "any_control_collision": any_control_collision,
            "multiprocessing_cpus": num_cpus,
            "env_type": env_type,
            "highway_env.duration": duration,
            "highway_env.mc_horizon": mc_horizon,
            "highway_env.n_montecarlo": n_montecarlo,
            "world_draws": world_draws,
            "warmup_steps": warmup,
            "mc_period": mc_period,
            "gatekeeper.enable_time_discounting": enable_time_discounting,
            "gatekeeper.behavior_cfg.nominal_class": nominal_class,
            "gatekeeper.behavior_cfg.defensive_class": defensive_class,
            "profiling": profiling,
        }
    ]

    for config in configs:
        # Convert the config dict into a string
        cfg_args = " ".join([f"{k}={v}" for k, v in config.items()])
        command = f"python {script} {cfg_args}"
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            break


if __name__ == '__main__':
    # Accept first argument as function name to be called
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to call")

    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    print(f"Calling {func_name}")
    globals()[func_name]()
