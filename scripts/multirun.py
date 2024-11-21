"""
One-off script for running multiple runs
"""
import os
import subprocess
import time
import multiprocessing

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


def baseline_configs(run_dir: str, tag: str | None = None):
    """
    :param run_dir: Indicate the run dir
    :param tag: Optional tag to prepend the run names
    """
    seed = 867
    script = "ttc.py"

    num_cpus = 12
    threads_per_world = 1
    env_type = "highway-v0"
    default_control_behavior = "sim.vehicles.highway.HotshotParams"
    world_draws = 1000
    duration = 80
    num_control_vehicles = 12  # Total number ego on road, not GK-online though.
    num_vehicles_control_speed = 12
    vehicles_count = 24
    num_collision_watch = 4
    # GK/MC Stuff
    n_online = 12
    warmup = 1e5
    mc_period = 5
    mc_horizon = 10
    n_montecarlo = 20
    enable_time_discounting = "false"
    # Paradoxically our GK 'nominal' is the risky one
    nominal_class = "sim.vehicles.highway.HotshotParams"
    defensive_class = "sim.vehicles.highway.DefensiveParams"
    offline_class = nominal_class

    # Profiling
    profiling = "false"

    runs = {
        # "nom": "sim.vehicles.highway.NominalParams",
        "def": "sim.vehicles.highway.DefensiveParams",
        "hotshot": "sim.vehicles.highway.HotshotParams",
    }

    def get_name(base: str):
        if not tag:
            return base
        return f"{tag}-{base}"

    configs = [
        {
            "name": os.path.join(run_dir, get_name(name)),
            "seed": seed,
            "highway_env.default_control_behavior": BEHAVIOR,
            "highway_env.controlled_vehicles": num_control_vehicles,
            "highway_env.num_vehicles_control_speed": num_vehicles_control_speed,
            "highway_env.vehicles_count": vehicles_count,
            "multiprocessing_cpus": num_cpus,
            "threads_per_world": threads_per_world,
            "env_type": env_type,
            "highway_env.duration": duration,
            "highway_env.mc_horizon": mc_horizon,
            "highway_env.n_montecarlo": n_montecarlo,
            "world_draws": world_draws,
            "warmup_steps": warmup,
            "mc_period": mc_period,
            "num_collision_watch": num_collision_watch,
            "gatekeeper.enable_time_discounting": enable_time_discounting,
            "gatekeeper.behavior_cfg.enable": "false",
            "gatekeeper.n_online": n_online,
            "gatekeeper.behavior_cfg.nominal_class": BEHAVIOR,
            "gatekeeper.behavior_cfg.defensive_class": defensive_class,
            "gatekeeper.behavior_cfg.offline_class": BEHAVIOR,
            "profiling": profiling,
        } for name, BEHAVIOR in runs.items()
    ]

    return script, configs


def online_configs(run_dir: str, tag: str | None = None):
    """
    :param run_dir: Indicate the run dir
    :param tag: Optional tag to prepend the run names
    """
    seed = 867
    script = "ttc.py"

    num_cpus = 12
    threads_per_world = 1
    env_type = "highway-v0"
    default_control_behavior = "sim.vehicles.highway.HotshotParams"
    world_draws = 1
    duration = 80
    num_control_vehicles = 12  # Total number ego on road, not GK-online though.
    num_vehicles_control_speed = 12
    vehicles_count = 24
    num_collision_watch = 4
    # GK/MC Stuff
    warmup = 0
    mc_period = 5
    mc_horizon = 10
    n_montecarlo = 20
    enable_time_discounting = "false"
    # Paradoxically our GK 'nominal' is the risky one
    nominal_class = "sim.vehicles.highway.HotshotParams"
    defensive_class = "sim.vehicles.highway.DefensiveParams"
    offline_class = nominal_class

    # Profiling
    profiling = "false"

    runs = {
        # "online-4": 4,
        "online-12": 12,
    }

    def get_name(base: str):
        if not tag:
            return base
        return f"{tag}-{base}"

    configs = [
        {
            "name": os.path.join(run_dir, get_name(name)),
            "seed": seed,
            "highway_env.default_control_behavior": nominal_class,
            "highway_env.controlled_vehicles": num_control_vehicles,
            "highway_env.num_vehicles_control_speed": num_vehicles_control_speed,
            "highway_env.vehicles_count": vehicles_count,
            "multiprocessing_cpus": num_cpus,
            "threads_per_world": threads_per_world,
            "env_type": env_type,
            "highway_env.duration": duration,
            "highway_env.mc_horizon": mc_horizon,
            "highway_env.n_montecarlo": n_montecarlo,
            "world_draws": world_draws,
            "warmup_steps": warmup,
            "mc_period": mc_period,
            "num_collision_watch": num_collision_watch,
            "gatekeeper.enable_time_discounting": enable_time_discounting,
            "gatekeeper.n_online": n_online,
            "gatekeeper.behavior_cfg.nominal_class": nominal_class,
            "gatekeeper.behavior_cfg.defensive_class": defensive_class,
            "gatekeeper.behavior_cfg.offline_class": offline_class,
            "profiling": profiling,
        } for name, n_online in runs.items()
    ]

    return script, configs


def run_online_exp():
    # Name the run, and where it will be saved
    run_dir = "tmp/crash-4"
    tag = "sample"

    script, configs = online_configs(run_dir, tag)
    _run_configs(script, configs)


def run_baseline():
    # Name the run, and where it will be saved
    run_dir = "tmp/ex2"
    tag = "baselines"

    script, configs = baseline_configs(run_dir, tag)
    _run_configs(script, configs)


def run_hpc_online_exp():
    # Name the run, and where it will be saved
    run_dir = "manuscript/hpc/highway"
    tag = None
    script, configs = online_configs(run_dir, tag)

    # Then just update the number of cores accordingly
    num_cpu = 96
    threads_per_world = 32
    world_draws = 200
    duration = 100
    n_montecarlo = 64

    for config in configs:
        config["multiprocessing_cpus"] = num_cpu
        config["threads_per_world"] = threads_per_world
        config["world_draws"] = world_draws
        config["highway_env.duration"] = duration
        config["highway_env.n_montecarlo"] = n_montecarlo

    _run_configs(script, configs)


def run_hpc_baseline():
    # Name the run, and where it will be saved
    run_dir = "manuscript/hpc/highway"
    tag = "baselines"
    script, configs = baseline_configs(run_dir, tag)

    # Then just update the number of cores accordingly
    num_cpu = 64
    threads_per_world = 1
    world_draws = 1000
    duration = 100
    # n_montecarlo = 100

    for config in configs:
        config["multiprocessing_cpus"] = num_cpu
        config["threads_per_world"] = threads_per_world
        config["world_draws"] = world_draws
        config["highway_env.duration"] = duration
        # config["highway_env.n_montecarlo"] = n_montecarlo

    _run_configs(script, configs)


def _run_configs(script: str, configs: list[dict]):
    for config in configs:
        # Convert the config dict into a string
        cfg_args = " ".join([f"{k}={v}" for k, v in config.items()])
        command = f"python {script} {cfg_args}"
        print(f"Running: {command}")
        print("\n\nCan be pesky to kill all the spawned processes with using `threads_per_core` > 1. \nUse pkill -f "
              "'python ttc.py' \n")
        time.sleep(5)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            break


if __name__ == '__main__':
    # Accept first argument as function name to be called
    import sys
    # multiprocessing.set_start_method("spawn", force=True)

    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to call")

    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    print(f"Calling {func_name}")
    globals()[func_name]()
