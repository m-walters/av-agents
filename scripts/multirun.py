"""
One-off script for running multiple runs
"""
import subprocess
import os


SPEC_BEHAVIORS = {
    # "nom": "sim.vehicles.highway.NominalParams",
    # "cons": "sim.vehicles.highway.ConservativeParams",
    # "hotshot": "sim.vehicles.highway.HotshotParams",
    "polite-incr": "sim.vehicles.highway.PolitenessIncr",
    "polite-decr": "sim.vehicles.highway.PolitenessDecr",
    "timedist-incr": "sim.vehicles.highway.TimeDistWantedIncr",
    "timedist-decr": "sim.vehicles.highway.TimeDistWantedDecr",
    "acc-max-incr": "sim.vehicles.highway.AccMaxIncr",
    "acc-max-decr": "sim.vehicles.highway.AccMaxDecr",
    "comf-brake-incr": "sim.vehicles.highway.ComfBrakingIncr",
    "comf-brake-decr": "sim.vehicles.highway.ComfBrakingDecr",
}

def run_multiagent_sequence():
    seed = 86777
    script = "ttc.py"

    num_cpus = 8
    duration = 100
    world_draws = 300
    warmup = 1e5
    env_type = "racetrack-v0"
    run_dir = "bigsweep/av-8"
    any_control_collision = "true"

    configs = [
        {
            "name": os.path.join(run_dir, name),
            "seed": seed,
            "highway_env.default_control_behavior": behavior,
            "highway_env.controlled_vehicles": 8,
            "highway_env.duration": duration,
            "any_control_collision": any_control_collision,
            "world_draws": world_draws,
            "warmup_steps": warmup,
            "multiprocessing_cpus": num_cpus,
            "env_type": env_type,
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
