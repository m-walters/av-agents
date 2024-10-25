"""
Similar to basic.py, but designed for a single run that also creates a video
of the run, and a video of the evolution of the data during the sim.
"""
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
from sim import models, plotting, run, utils
from sim.envs.highway import AVHighway

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "tracker"

RESULTS_DIR = "../results"
LATEST_DIR = f"{RESULTS_DIR}/latest"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params = run.init(cfg, LATEST_DIR)
    ds = run.init_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'], run_params['n_montecarlo']
    )
    seed = run_params['seed']
    env_cfg = cfg.highway_env

    if cfg.world_draws > 1:
        raise ValueError("world_draws > 1 not allowed for tracker.py")

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    preference_prior = getattr(models, cfg.preference_prior.model)(**cfg.preference_prior, seed=cfg.seed)
    risk_model = getattr(models, cfg.risk.model)(preference_prior=preference_prior, **cfg.risk, seed=cfg.seed)

    # Create our gym Env
    render_mode = 'rgb_array'
    video_dir = f"{LATEST_DIR}/recordings"
    video_prefix = "sim"
    env = RecordVideo(
        gym.make('AVAgents/racetrack-v0', render_mode=render_mode), video_dir, name_prefix=video_prefix
    )

    uenv: "AVHighway" = env.unwrapped
    uenv.update_config(env_cfg, reset=False)

    # Run a world simulation
    rkey = utils.JaxRKey(seed)
    obs, info = env.reset(seed=rkey.next_seed())
    i_mc = 0  # Tracking MC steps

    for step in tqdm(range(run_params['duration']), desc="Steps"):
        if step >= run_params['warmup_steps']:
            if step % run_params['mc_period'] == 0:
                # Run the montecarlo simulation, capturing the risks, losses
                # Returned dimensions are [n_montecarlo]
                losses, loss_log_probs, collisions = uenv.simulate_mc()

                risk, entropy, energy = risk_model(losses, loss_log_probs)

                # Record data
                ds["mc_loss"][0, i_mc, :] = losses
                ds["loss_mean"][0, i_mc] = np.mean(losses)
                ds["loss_p5"][0, i_mc] = np.percentile(losses, 5)
                ds["loss_p95"][0, i_mc] = np.percentile(losses, 95)
                ds["risk"][0, i_mc] = risk
                ds["entropy"][0, i_mc] = entropy
                ds["energy"][0, i_mc] = energy

                i_mc += 1

        # We do action after MC sim in case it informs actions
        # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Record the actuals
        ds["reward"][0, step] = reward
        ds["defensive_reward"][0, step] = info["rewards"]["defensive_reward"]
        ds["speed_reward"][0, step] = info["rewards"]["speed_reward"]
        ds["crash_reward"][0, step] = info["rewards"]["crash_reward"]

        logger.debug(f"REWARD: {reward}")
        if terminated or truncated:
            if terminated:
                logger.info(f"Collision (terminated) at {step}")
                # Truncate the data along the duration axis
                ds = ds.isel(step=slice(0, step + 1))
            break

    # Conclude the video
    env.close()

    # Append an extra data array "real_loss" to our dataset that is the negative of reward
    ds["real_loss"] = -ds["reward"]

    # Automatically save latest
    logger.info("Saving results")
    utils.Results.save_ds(ds, f"{LATEST_DIR}/results.nc")

    # Save frames
    np.save(f"{LATEST_DIR}/frames.npy", env.video_recorder.recorded_frames)

    # Create the video with the saved frames and data
    ds_label_map = {
        "R_Def": "defensive_reward",
        "R_Spd": "speed_reward",
        "Actual Loss": "real_loss",
        "E[Loss]": "loss_mean",
        "E[Energy]": "energy",
        "E[Entropy]": "entropy",
        "Risk": "risk",
    }
    if run_params['warmup_steps'] >= run_params['duration']:
        # No Montecarlo data to plot
        _ = ds_label_map.pop("E[Loss]")
        _ = ds_label_map.pop("E[Energy]")
        _ = ds_label_map.pop("E[Entropy]")
        _ = ds_label_map.pop("Risk")

    plotter = plotting.AVPlotter()
    plotter.create_animation(
        f"{LATEST_DIR}/tracker.mp4",
        ds,
        ds_label_map,
        env.video_recorder.recorded_frames,
        fps=10,
    )

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
