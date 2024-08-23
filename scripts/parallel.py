import logging
import os, shutil
import random

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import sim.params as sim_params
from sim import models, utils

RESULTS_DIR = "../results"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name="highway")
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    log_level = cfg.get("log_level", "INFO")
    # Get logger *after* setting the level
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)

    # Print our config
    logger.debug(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    # Get and validate the highway-env environment config
    # env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    # Order is first the class defaults
    # overrides = AVHighway.av_default_config()
    # then the yaml overrides
    if cfg.get("env_overrides", None):
        overrides = OmegaConf.to_container(cfg.get("env_overrides", {}), resolve=True)
        OmegaConf.set_struct(cfg, False)
        cfg.env.update(overrides)
        OmegaConf.set_struct(cfg, True)
        # overrides.update(env_overrides)

    env_cfg = cfg.env = utils.validate_env_config(cfg.env)

    # Seed RNG
    seed = cfg.get("seed", 8675309)
    np.random.seed(seed)
    random.seed(seed)

    # Results save dir
    latest_dir = RESULTS_DIR + "/latest"
    if os.path.exists(latest_dir):
        # Clear and write over the latest dir
        for f in os.listdir(latest_dir):
            file_path = os.path.join(latest_dir, f)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        # Create dir
        os.makedirs(latest_dir)

    OmegaConf.save(config=cfg, f=f"{latest_dir}/config.yaml")

    # If a name is provided, save there too
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=f"{run_dir}/config.yaml")

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    preference_prior = models.SoftmaxPreferencePrior(kappa=1.0)
    risk_model = models.DifferentialEntropyRiskModel(preference_prior=preference_prior)

    # Create an xarray Dataset
    duration = env_cfg['duration']
    world_draws = cfg.world_draws
    warmup_steps = cfg.get("warmup_steps", 0)

    ds = xr.Dataset(
        {
            "reward": (("world", "step"), np.zeros((world_draws, duration))),
            "collision_reward": (("world", "step"), np.zeros((world_draws, duration))),
            "speed_reward": (("world", "step"), np.zeros((world_draws, duration))),
            "risk": (("world", "step"), np.zeros((world_draws, duration))),
            "entropy": (("world", "step"), np.zeros((world_draws, duration))),
            "energy": (("world", "step"), np.zeros((world_draws, duration))),
            # Tracking the MC losses
            "mc_loss": (("world", "step", "sample"), np.zeros((world_draws, duration, env_cfg['n_montecarlo']))),
            "loss_mean": (("world", "step"), np.zeros((world_draws, duration))),
            "loss_p5": (("world", "step"), np.zeros((world_draws, duration))),  # 5% percentile
            "loss_p95": (("world", "step"), np.zeros((world_draws, duration))),  # 95% percentile
            # Track if a collision occurred
            "crashed": (("world", "step"), np.zeros((world_draws, duration))),
        },
        coords={
            "world": np.arange(world_draws),
            "step": np.arange(duration),
            "sample": np.arange(env_cfg['n_montecarlo']),
        },
    )

    # NOTE that e.g. env_cfg['duration'] was unaccepted by AsyncVectorEnv (though it as a declared var was fine..)
    # Also, note using `.unwrapped` here. This is so that the `.call(..)` later doesn't raise a warning.
    # The default wrapper applied to our env is just `OrderEnforcing`, which only raises an error if `step`
    # is called before `reset`, so...hopefully we're fine.
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
    rkey = utils.JaxRKey(seed)
    seeds = [rkey.next_seed() for _ in range(world_draws)]

    # Providing a single seed or a list of seeds both produce variable world inits
    # If you wanted duplicated worlds, provide a list of the same seed [seed]*world_draws
    observations, infos = envs_wrapper.reset(seed=seeds)

    for step in tqdm(range(duration), desc="Step"):
        if step >= warmup_steps:
            # Get our vectorized losses
            # Each item is a tuple of (losses, log_probs, collisions)
            result = envs_wrapper.unwrapped.call("simulate_mc")
            # Unzip the results and create arrays, shape=[world_draws, n_montecarlo]
            losses = np.array([r[0] for r in result])
            log_probs = np.array([r[1] for r in result])

            # Transpose input to let axis=0 be n_montecarlo
            risk, entropy, energy = risk_model(losses.T, log_probs.T)

            # Record data
            ds["mc_loss"][:, step, :] = losses
            ds["loss_mean"][:, step] = np.mean(losses, axis=1)
            ds["loss_p5"][:, step] = np.percentile(losses, 5, axis=1)
            ds["loss_p95"][:, step] = np.percentile(losses, 95, axis=1)
            ds["risk"][:, step] = risk
            ds["entropy"][:, step] = entropy
            ds["energy"][:, step] = energy

        # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
        # actions = envs_wrapper.action_space.sample()  # Sample with extra dim of vectorized envs
        # actions = envs_wrapper.call("action_sample")  # Sample from each individually
        actions = [None] * world_draws

        observations, rewards, terminated, truncated, infos = envs_wrapper.step(actions)
        ds["reward"][:, step] = rewards
        # info["rewards"] is a list of {"collision_reward": <float>, "speed_reward": <float>} items
        ds["collision_reward"][:, step] = [r["collision_reward"] for r in infos["rewards"]]
        ds["speed_reward"][:, step] = [r["speed_reward"] for r in infos["rewards"]]
        ds["crashed"][:, step] = infos["crashed"]

    # Close
    envs_wrapper.close()

    # Automatically save latest
    logger.info("Saving results")
    utils.Results.save_ds(ds, f"{latest_dir}/results.nc")

    # If a name is provided, save there too
    if "name" in cfg:
        utils.Results.save_ds(ds, f"{run_dir}/results.nc")


if __name__ == '__main__':
    main()
