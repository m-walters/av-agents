import logging
import os
import shutil

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from omegaconf import DictConfig
from tqdm import tqdm

import sim.params as sim_params
from sim import models, run, utils

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "tmp"

RESULTS_DIR = "../results"
LATEST_DIR = f"{RESULTS_DIR}/latest"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params, gk_cfg = run.init(cfg, LATEST_DIR)
    ds = run.init_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'], run_params['n_montecarlo']
    )
    seed = run_params['seed']
    env_cfg = cfg.highway_env

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    preference_prior = models.SoftmaxPreferencePrior(kappa=1.0)
    risk_model = models.DifferentialEntropyRiskModel(preference_prior=preference_prior)

    # NOTE that e.g. env_cfg['duration'] was unaccepted by AsyncVectorEnv (though it as a declared var was fine..)
    # Also, note using `.unwrapped` here. This is so that the `.call(..)` later doesn't raise a warning.
    # The default wrapper applied to our env is just `OrderEnforcing`, which only raises an error if `step`
    # is called before `reset`, so...hopefully we're fine.
    world_draws = int(run_params['world_draws'])
    duration = int(run_params['duration'])
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                'AVAgents/highway-v0', render_mode='rgb_array', max_episode_steps=duration
            ).unwrapped for _ in range(world_draws)
        ],
        copy=False,
        shared_memory=False,
    )

    # Create a wrapper environment to save episode returns and episode lengths
    # Note that envs_wrapper.unwrapped is our envs AsyncVectorEnv
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=world_draws)

    # Update the env configs
    # For now, all configs are the same
    envs.call("update_config", env_cfg, reset=False)
    # Need separate seeds for each environment
    rkey = utils.NpyRKey(seed)
    seeds = [rkey.next_seed() for _ in range(world_draws)]
    i_mc = 0

    # Providing a single seed or a list of seeds both produce variable world inits
    # If you wanted duplicated worlds, provide a list of the same seed [seed]*world_draws
    observations, infos = envs_wrapper.reset(seed=seeds)

    for step in tqdm(range(duration), desc="Step"):
        if step >= run_params['warmup_steps']:
            if step % run_params['mc_period'] == 0:
                # Get our vectorized losses
                # Each item is a tuple of (losses, log_probs, collisions)
                result = envs_wrapper.unwrapped.call("simulate_mc")
                # Unzip the results and create arrays, shape=[world_draws, n_montecarlo]
                losses = np.array([r[0] for r in result])
                log_probs = np.array([r[1] for r in result])

                # Transpose input to let axis=0 be n_montecarlo
                risk, entropy, energy = risk_model(losses.T, log_probs.T)

                # Record data
                # ds["mc_loss"][:, i_mc, :] = losses
                ds["loss_mean"][:, i_mc] = np.mean(losses, axis=1)
                ds["loss_p5"][:, i_mc] = np.percentile(losses, 5, axis=1)
                ds["loss_p95"][:, i_mc] = np.percentile(losses, 95, axis=1)
                ds["risk"][:, i_mc] = risk
                ds["entropy"][:, i_mc] = entropy
                ds["energy"][:, i_mc] = energy

                i_mc += 1

        # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
        # actions = envs_wrapper.action_space.sample()  # Sample with extra dim of vectorized envs
        # actions = envs_wrapper.call("action_sample")  # Sample from each individually
        actions = [None] * world_draws

        observations, rewards, terminated, truncated, infos = envs_wrapper.step(actions)
        ds["reward"][:, step] = rewards
        # info["rewards"] is a list of {"defensive_reward": <float>, "speed_reward": <float>} items
        ds["defensive_reward"][:, step] = [r["defensive_reward"] for r in infos["rewards"]]
        ds["speed_reward"][:, step] = [r["speed_reward"] for r in infos["rewards"]]
        ds["crashed"][:, step] = infos["crashed"]

    # Close
    envs_wrapper.close()

    # Append an extra data array "real_loss" to our dataset that is the negative of reward
    ds["real_loss"] = -ds["reward"]

    # Automatically save latest
    logger.info("Saving results")
    utils.Results.save_ds(ds, f"{LATEST_DIR}/results.nc")

    # If a name is provided, copy results over
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        logger.info(f"Copying run results to {run_dir}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(LATEST_DIR, run_dir)


if __name__ == '__main__':
    main()
