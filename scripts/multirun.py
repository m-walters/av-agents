"""
One-off script for running multiple runs
"""
import subprocess


def run_multiagent_sequence():
    seed = 86

    configs = [
        {
            "name": "gk-lstar-0p1",
            "seed": seed,
            "use_mp": True,
            "gatekeeper.preference_prior.p_star": 0.1,
            "gatekeeper.preference_prior.l_star": 0.1,
        },
        {
            "name": "gk-lstar-0p2",
            "seed": seed,
            "use_mp": True,
            "gatekeeper.preference_prior.p_star": 0.1,
            "gatekeeper.preference_prior.l_star": 0.2,
        },
        {
            "name": "gk-lstar-0p5",
            "seed": seed,
            "use_mp": True,
            "gatekeeper.preference_prior.p_star": 0.1,
            "gatekeeper.preference_prior.l_star": 0.5,
        },
        {
            "name": "gk-lstar-0p9",
            "seed": seed,
            "use_mp": True,
            "gatekeeper.preference_prior.p_star": 0.1,
            "gatekeeper.preference_prior.l_star": 0.9,
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
        command = f"python multiagent.py {cfg_args}"
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            break


if __name__ == '__main__':
    run_multiagent_sequence()
