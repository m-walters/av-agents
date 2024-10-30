import logging
import os
import shutil

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from tqdm import tqdm

import sim.params as sim_params
from sim import models, run, utils
from sim.envs.highway import AVHighway

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "tmp"

RESULTS_DIR = "../results"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params = run.init(cfg)
    ds = run.init_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'], run_params['n_montecarlo']
    )
    seed = run_params['seed']
    env_cfg = cfg.highway_env
    latest_dir = f"{RESULTS_DIR}/latest"

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
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

    preference_prior = getattr(models, cfg.preference_prior.model)(**cfg.preference_prior, seed=cfg.seed)
    risk_model = getattr(models, cfg.risk.model)(preference_prior=preference_prior, **cfg.risk, seed=cfg.seed)

    # We have to get seeds based on global seed
    rkey = utils.JaxRKey(seed)
    i_mc = 0

    for wdraw in range(run_params['world_draws']):
        obs, info = env.reset(seed=rkey.next_seed())

        # Run a world simulation
        for step in tqdm(range(run_params['duration']), desc=f"World {wdraw}"):
            # action = env.action_space.sample()
            # action = env.action_space.sample()
            # spd_reward = env.unwrapped.speed_reward()
            # def_reward = env.unwrapped.defensive_reward()

            if step >= run_params['warmup_steps']:
                if step % run_params['mc_period'] == 0:
                    # Run the montecarlo simulation, capturing the risks, losses
                    # Returned dimensions are [n_montecarlo]
                    losses, loss_log_probs, collisions = uenv.simulate_mc()

                    risk, entropy, energy = risk_model(losses, loss_log_probs)

                    # Record data
                    ds["mc_loss"][wdraw, i_mc, :] = losses
                    ds["loss_mean"][wdraw, i_mc] = np.mean(losses)
                    ds["loss_p5"][wdraw, i_mc] = np.percentile(losses, 5)
                    ds["loss_p95"][wdraw, i_mc] = np.percentile(losses, 95)
                    ds["risk"][wdraw, i_mc] = risk
                    ds["entropy"][wdraw, i_mc] = entropy
                    ds["energy"][wdraw, i_mc] = energy

            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            action = None

            obs, reward, terminated, truncated, info = env.step(action)
            ds["reward"][wdraw, step] = reward
            ds["defensive_reward"][wdraw, step] = info["rewards"]["defensive_reward"]
            ds["speed_reward"][wdraw, step] = info["rewards"]["speed_reward"]

            logger.debug(f"REWARD: {reward}")
            if terminated or truncated:
                if terminated:
                    logger.info(f"Collision (terminated) at {step}")
                break

    if record:
        env.render()
        env.close()

    # Append an extra data array "real_loss" to our dataset that is the negative of reward
    ds["real_loss"] = -ds["reward"]

    # Automatically save latest
    logger.info("Saving results")
    utils.Results.save_ds(ds, f"{latest_dir}/results.nc")

    # If a name is provided, copy results over
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        logger.info(f"Copying run results to {run_dir}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(latest_dir, run_dir)


if __name__ == '__main__':
    main()
