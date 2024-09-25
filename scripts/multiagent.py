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
from sim import gatekeeper, recorder, run, utils
from sim.envs.highway import AVHighway

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "multiagent"

RESULTS_DIR = "../results"
LATEST_DIR = f"{RESULTS_DIR}/latest"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params = run.init(cfg, LATEST_DIR)
    if run_params['world_draws'] > 1:
        raise ValueError("world_draws > 1 not configured for multiagent")

    ds = run.init_multiagent_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'],
        run_params['n_montecarlo'], cfg.gatekeeper['n_controlled']
    )
    seed = run_params['seed']
    env_cfg = cfg.highway_env
    use_mp = cfg.get('use_mp', False)

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    # Create our gym Env
    if use_mp:
        render_mode = None  # No visuals because of multiprocessing
        env = gym.make('AVAgents/highway-v0', render_mode=render_mode)
    else:
        render_mode = 'rgb_array'
        video_dir = f"{LATEST_DIR}/recordings"
        video_prefix = "sim"
        env = recorder.AVRecorder(
            gym.make('AVAgents/highway-v0', render_mode=render_mode), video_dir, name_prefix=video_prefix
        )

    uenv: "AVHighway" = env.unwrapped
    uenv.update_config(env_cfg, reset=True)

    # Init the gatekeeper
    gk_cmd = gatekeeper.GatekeeperCommand(
        uenv, cfg.gatekeeper, uenv.controlled_vehicles, seed
    )

    # Run world simulation
    rkey = utils.JaxRKey(seed)
    obs, info = env.reset(seed=rkey.next_seed())
    i_mc = 0  # Tracking MC steps
    crashed_vehicles = set()

    if use_mp:
        with multiprocessing.Pool(cfg.get('multiprocessing_cpus', 8), maxtasksperchild=100) as pool:
            for step in tqdm(range(run_params['duration']), desc="Steps"):
                # We'll use the gatekeeper params for montecarlo control
                if step >= run_params['warmup_steps']:
                    if step % gk_cmd.mc_period == 0:
                        # Returned dimensions are [n_controlled]
                        results = gk_cmd.run(pool)

                        # Record data
                        ds["mc_loss"][0, i_mc, :, :] = results["losses"]
                        ds["loss_mean"][0, i_mc, :] = np.mean(results["losses"], axis=0)
                        ds["loss_p5"][0, i_mc, :] = np.percentile(results["losses"], 5, axis=0)
                        ds["loss_p95"][0, i_mc, :] = np.percentile(results["losses"], 95, axis=0)
                        ds["risk"][0, i_mc, :] = results["risk"]
                        ds["entropy"][0, i_mc, :] = results["entropy"]
                        ds["energy"][0, i_mc, :] = results["energy"]
                        i_mc += 1

                # We do action after MC sim in case it informs actions
                # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
                action = env.action_space.sample()
                # action = tuple(np.ones_like(action))
                obs, reward, crashed, truncated, info = env.step(action)

                # Record the actuals
                ds["reward"][0, step, :] = reward
                ds["crashed"][0, step, :] = crashed
                ds["defensive_reward"][0, step, :] = info["rewards"]["defensive_reward"]
                ds["speed_reward"][0, step, :] = info["rewards"]["speed_reward"]

                # Print which, if any, av-IDs have crashed
                crashed_ids = np.argwhere(crashed)
                if crashed_ids.any():
                    av_ids = np.array(info['av_ids'])
                    crashed_set = set(av_ids[crashed_ids].flatten())
                    if crashed_set - crashed_vehicles:
                        logger.info(f"Crashed vehicles (Step {step}): {crashed_set}")
                        crashed_vehicles.update(crashed_set)

                if truncated:
                    # Times up
                    break

    else:
        # No Multiprocessing
        for step in tqdm(range(run_params['duration']), desc="Steps"):
            # We'll use the gatekeeper params for montecarlo control
            if step >= run_params['warmup_steps']:
                if step % gk_cmd.mc_period == 0:
                    # Returned dimensions are [n_controlled]
                    results = gk_cmd.run(None)

                    # Record data
                    ds["mc_loss"][0, i_mc, :, :] = results["losses"]
                    ds["loss_mean"][0, i_mc, :] = np.mean(results["losses"], axis=0)
                    ds["loss_p5"][0, i_mc, :] = np.percentile(results["losses"], 5, axis=0)
                    ds["loss_p95"][0, i_mc, :] = np.percentile(results["losses"], 95, axis=0)
                    ds["risk"][0, i_mc, :] = results["risk"]
                    ds["entropy"][0, i_mc, :] = results["entropy"]
                    ds["energy"][0, i_mc, :] = results["energy"]
                    i_mc += 1

            # We do action after MC sim in case it informs actions
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            # action = tuple(np.ones_like(action))
            obs, reward, crashed, truncated, info = env.step(action)

            # Record the actuals
            ds["reward"][0, step, :] = reward
            ds["crashed"][0, step, :] = crashed
            ds["defensive_reward"][0, step, :] = info["rewards"]["defensive_reward"]
            ds["speed_reward"][0, step, :] = info["rewards"]["speed_reward"]

            # Print which, if any, av-IDs have crashed
            crashed_ids = np.argwhere(crashed)
            if crashed_ids.any():
                av_ids = np.array(info['av_ids'])
                crashed_set = set(av_ids[crashed_ids].flatten())
                if crashed_set - crashed_vehicles:
                    logger.info(f"Crashed vehicles (Step {step}): {crashed_set}")
                    crashed_vehicles.update(crashed_set)

            if truncated:
                # Times up
                break

        # Conclude the video
        env.close()

        if isinstance(env, recorder.AVRecorder):
            # Save frames
            np.save(f"{LATEST_DIR}/frames.npy", env.video_recorder.recorded_frames)

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
