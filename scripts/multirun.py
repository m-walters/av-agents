"""
One-off script for running multiple runs
"""
import subprocess
import os


SPEC_BEHAVIORS = {
    "nom": "sim.vehicles.highway.NominalParams",
    # "cons": "sim.vehicles.highway.ConservativeParams",
    # "def": "sim.vehicles.highway.DefensiveParams",
    # "hotshot": "sim.vehicles.highway.HotshotParams",
    # "polite-incr": "sim.vehicles.highway.PolitenessIncr",
    # "polite-decr": "sim.vehicles.highway.PolitenessDecr",
    # "timedist-incr": "sim.vehicles.highway.TimeDistWantedIncr",
    # "timedist-decr": "sim.vehicles.highway.TimeDistWantedDecr",
    # "acc-max-incr": "sim.vehicles.highway.AccMaxIncr",
    # "acc-max-decr": "sim.vehicles.highway.AccMaxDecr",
    # "comf-brake-incr": "sim.vehicles.highway.ComfBrakingIncr",
    # "comf-brake-decr": "sim.vehicles.highway.ComfBrakingDecr",
    # "reckmax1": "sim.vehicles.highway.ReckMax1",
    # "reckmax2": "sim.vehicles.highway.ReckMax2",
    # "reckmax3": "sim.vehicles.highway.ReckMax3",
    # "def-HE": "sim.vehicles.highway.DefensiveHE",
    # "def-1": "sim.vehicles.highway.Defensive1",
    # "def-2": "sim.vehicles.highway.Defensive2",
}

def run_multiagent_sequence():
    seed = 86777
    script = "ttc.py"

    num_cpus = 8
    env_type = "racetrack-v0"
    any_control_collision = "true"
    world_draws = 1
    duration = 20
    warmup = 0
    mc_period = 5
    mc_horizon = 20
    n_montecarlo = 10
    # Discounting
    enable_time_discounting = "true"

    # Name the run, and where it will be saved
    RUN_DIR = "profiling"
    tmp_name = os.path.join(
        RUN_DIR,
        "gk-gamma"
    )

    configs = [
        {
            "name": tmp_name,
            # "name": os.path.join(RUN_DIR, name),
            "seed": seed,
            "highway_env.default_control_behavior": behavior,
            "highway_env.controlled_vehicles": 8,
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
        } for name, behavior in SPEC_BEHAVIORS.items()
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
    run_multiagent_sequence()
