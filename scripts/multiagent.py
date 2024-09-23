"""
Multiagent Gatekeeper simulation
"""
import logging
import multiprocessing
import os
import shutil

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from omegaconf import DictConfig
from tqdm import tqdm

import sim.params as sim_params
from sim import gatekeeper, run, utils
from sim.envs.highway import AVHighway

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "multiagent"

RESULTS_DIR = "../results"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params = run.init(cfg)
    if run_params['world_draws'] > 1:
        raise ValueError("world_draws > 1 not configured for multiagent")

    ds = run.init_multiagent_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'],
        run_params['n_montecarlo'], cfg.gatekeeper['n_controlled']
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

    # Create our gym Env
    # render_mode = 'rgb_array'
    render_mode = None  # No visuals because of multiprocessing
    env = gym.make('AVAgents/highway-v0', render_mode=render_mode)

    uenv: "AVHighway" = env.unwrapped
    uenv.update_config(env_cfg, reset=True)

    # Run a world simulation
    rkey = utils.JaxRKey(seed)
    obs, info = env.reset(seed=rkey.next_seed())
    i_mc = 0  # Tracking MC steps

    # Init the gatekeeper
    gk_cmd = gatekeeper.GatekeeperCommand(
        uenv, cfg.gatekeeper, uenv.controlled_vehicles, seed
    )

    with multiprocessing.Pool(cfg.get('multiprocessing_cpus', 8), maxtasksperchild=100) as pool:
        for step in tqdm(range(run_params['duration']), desc="Steps"):
            # We'll use the gatekeeper params for montecarlo control
            if step >= run_params['warmup_steps']:
                if step % gk_cmd.mc_period == 0:
                    # Returned dimensions are [n_controlled]
                    results = gk_cmd.run(pool)

                    # Record data
                    ds["mc_loss"][0, i_mc, :, :] = results["losses"]
                    ds["loss_mean"][0, i_mc, :] = np.mean(results["losses"])
                    ds["loss_p5"][0, i_mc, :] = np.percentile(results["losses"], 5)
                    ds["loss_p95"][0, i_mc, :] = np.percentile(results["losses"], 95)
                    ds["risk"][0, i_mc, :] = results["risk"]
                    ds["entropy"][0, i_mc, :] = results["entropy"]
                    ds["energy"][0, i_mc, :] = results["energy"]
                    i_mc += 1

            # We do action after MC sim in case it informs actions
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            # action = tuple(np.ones_like(action))
            obs, reward, terminated, truncated, info = env.step(action)

            # Record the actuals
            # ds["reward"][0, step] = reward
            # ds["collision_reward"][0, step] = info["rewards"]["collision_reward"]
            # ds["speed_reward"][0, step] = info["rewards"]["speed_reward"]

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
    utils.Results.save_ds(ds, f"{latest_dir}/results.nc")

    # # Save frames
    # np.save(f"{latest_dir}/frames.npy", env.video_recorder.recorded_frames)
    #
    # # Create the video with the saved frames and data
    # ds_label_map = {
    #     "R_Coll": "collision_reward",
    #     "R_Spd": "speed_reward",
    #     "Actual Loss": "real_loss",
    #     "E[Loss]": "loss_mean",
    #     "E[Energy]": "energy",
    #     "E[Entropy]": "entropy",
    #     "Risk": "risk",
    # }
    # plotter = plotting.TrackerPlotter()
    # plotter.create_animation(
    #     f"{latest_dir}/tracker.mp4",
    #     ds,
    #     ds_label_map,
    #     env.video_recorder.recorded_frames,
    #     fps=10,
    # )

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
