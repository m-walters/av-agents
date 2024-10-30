"""
One-off script for running multiple runs
"""
import subprocess


def run_multiagent_sequence():
    seed = 86777
    script = "ttc.py"

    configs = [
        # {
        #     "name": "quick-ttc-gk-nc-1",
        #     "seed": seed,
        #     "use_mp": True,
        #     "highway_env.controlled_vehicles": 1,
        # },
        {
            "name": "quick-ttc-gk-nc-2",
            "seed": seed,
            "use_mp": True,
            "highway_env.controlled_vehicles": 2,
        },
        {
            "name": "quick-ttc-gk-nc-4",
            "seed": seed,
            "use_mp": True,
            "highway_env.controlled_vehicles": 4,
        },
        {
            "name": "quick-ttc-gk-nc-8",
            "seed": seed,
            "use_mp": True,
            "highway_env.controlled_vehicles": 8,
        },
        # {
        #     "name": "pstar-0p9",
        #     "use_mp": True,
        #     "gatekeeper.preference_prior.p_star": 0.9,
        #     "gatekeeper.preference_prior.l_star": 0.2,
        # },
        # {
        #     "name": "pstar-0p01_lstar-0p1",
        #     "use_mp": True,
        #     "gatekeeper.preference_prior.p_star": 0.01,
        #     "gatekeeper.preference_prior.l_star": 0.1,
        # },
        # {
        #     "name": "pstar-0p1_lstar-0p1",
        #     "use_mp": True,
        #     "gatekeeper.preference_prior.p_star": 0.1,
        #     "gatekeeper.preference_prior.l_star": 0.1,
        # },
        # {
        #     "name": "pstar-0p9_lstar-0p1",
        #     "use_mp": True,
        #     "gatekeeper.preference_prior.p_star": 0.9,
        #     "gatekeeper.preference_prior.l_star": 0.1,
        # },
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
