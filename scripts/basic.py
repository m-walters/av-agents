import logging
import os
import random

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
import xarray as xr
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import sim.params as sim_params
from sim import models, utils
from sim.envs.highway import AVHighway

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
        # DEPRECATED
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
    os.makedirs(latest_dir, exist_ok=True)
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
        for p in cfg.params:
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    # Control for whether to view the rollout live
    video = cfg.get("video", False)
    video_prefix = "sim"
    # Whether to save the video(s)
    record = cfg.get("record", False)
    if record:
        render_mode = 'rgb_array'
        video_dir = f"{latest_dir}/recordings"
        env = RecordVideo(
            gym.make('AVAgents/highway-v0', render_mode=render_mode), video_dir, name_prefix=video_prefix
        )
    elif video:
        render_mode = 'human'
        env = gym.make('AVAgents/highway-v0', render_mode=render_mode)
    else:
        render_mode = 'rgb_array'
        env = gym.make('AVAgents/highway-v0', render_mode=render_mode)

    uenv: "AVHighway" = env.unwrapped
    uenv.update_config(env_cfg, reset=False)

    preference_prior = models.SoftmaxPreferencePrior(kappa=1.0)
    risk_model = models.DifferentialEntropyRiskModel(preference_prior=preference_prior)

    # Create an xarray Dataset
    duration = env_cfg['duration']
    world_draws = cfg.world_draws
    warmup_steps = cfg.get("warmup_steps", 0)

    ds = xr.Dataset(
        {
            "reward": (("world", "step"), np.zeros((world_draws, duration))),
            "risk": (("world", "step"), np.zeros((world_draws, duration))),
            "entropy": (("world", "step"), np.zeros((world_draws, duration))),
            "energy": (("world", "step"), np.zeros((world_draws, duration))),
            # Tracking the MC losses
            "mc_loss": (("world", "step", "sample"), np.zeros((world_draws, duration, env_cfg['n_montecarlo']))),
            "loss_mean": (("world", "step"), np.zeros((world_draws, duration))),
            "loss_p5": (("world", "step"), np.zeros((world_draws, duration))),  # 5% percentile
            "loss_p95": (("world", "step"), np.zeros((world_draws, duration))),  # 95% percentile
        },
        coords={
            "world": np.arange(world_draws),
            "step": np.arange(duration),
            "sample": np.arange(env_cfg['n_montecarlo']),
        },
    )

    # We have to get seeds based on global seed
    rkey = utils.JaxRKey(seed)

    for wdraw in range(cfg.world_draws):
        # obs, info = env.reset(seed=seed)
        obs, info = env.reset(seed=rkey.next_seed())

        # Run a world simulation
        for step in tqdm(range(duration), desc=f"World {wdraw}"):
            # action = env.action_space.sample()
            # action = env.action_space.sample()
            # spd_reward = env.unwrapped.speed_reward()
            # coll_reward = env.unwrapped.collision_reward()

            if step >= warmup_steps:
                # Run the montecarlo simulation, capturing the risks, losses
                # Returned dimensions are [n_montecarlo]
                losses, loss_log_probs, collisions, mc_rewards = uenv.simulate_mc()

                risk, entropy, energy = risk_model(losses, loss_log_probs)

                # Record data
                ds["mc_loss"][wdraw, step, :] = losses
                ds["loss_mean"][wdraw, step] = np.mean(losses)
                ds["loss_p5"][wdraw, step] = np.percentile(losses, 5)
                ds["loss_p95"][wdraw, step] = np.percentile(losses, 95)
                ds["risk"][wdraw, step] = risk
                ds["entropy"][wdraw, step] = entropy
                ds["energy"][wdraw, step] = energy

            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            action = None

            obs, reward, terminated, truncated, info = env.step(action)
            ds["reward"][wdraw, step] = reward

            logger.debug(f"REWARD: {reward}")
            if terminated or truncated:
                if terminated:
                    logger.info(f"Collision (terminated) at {step}")
                break

    if record:
        env.render()
        env.close()

    # Automatically save latest
    logger.info("Saving results")
    utils.Results.save_ds(ds, f"{latest_dir}/results.nc")

    # If a name is provided, save there too
    if "name" in cfg:
        utils.Results.save_ds(ds, f"{run_dir}/results.nc")


if __name__ == '__main__':
    main()
