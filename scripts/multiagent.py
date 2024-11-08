"""
Multiagent Gatekeeper simulation
"""
import logging
import multiprocessing
import os
import shutil
import time

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from omegaconf import DictConfig
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import sim.params as sim_params
from sim import gatekeeper, recorder, run, utils
from sim.envs.highway import AVHighway

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "multiagent"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params, gk_cfg = run.init(cfg)
    run_dir = cfg.run_dir

    if run_params['world_draws'] > 1:
        raise ValueError("world_draws > 1 not configured for multiagent")

    ds, behavior_index = run.init_multiagent_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'],
        run_params['n_montecarlo'], gk_cfg['n_controlled']
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
        env = gym.make(f"AVAgents/{cfg.get('env_type', 'racetrack-v0')}", render_mode=render_mode)
    else:
        render_mode = 'rgb_array'
        video_dir = f"{run_dir}/recordings"
        video_prefix = "sim"
        env = recorder.AVRecorder(
            gym.make(f"AVAgents/{cfg.get('env_type', 'racetrack-v0')}", render_mode=render_mode), video_dir, name_prefix=video_prefix
        )

    uenv: "AVHighway" = env.unwrapped
    uenv.update_config(env_cfg, reset=True)

    # Init the gatekeeper
    gk_cmd = gatekeeper.GatekeeperCommand(
        uenv, gk_cfg, uenv.controlled_vehicles, seed
    )

    # Run world simulation
    rkey = utils.JaxRKey(seed)
    obs, info = env.reset(seed=rkey.next_seed())
    i_mc = 0  # Tracking MC steps
    crashed_vehicles = set()

    def checkpoint_dataset(msg: str = None):
        """
        Save the dataset to disk
        """
        if msg:
            logger.info(msg)
        utils.Results.save_ds(ds, f"{run_dir}/results.nc")
        # Save behavior index
        utils.Results.save_json(behavior_index, f"{run_dir}/behavior_index.json")

    if use_mp:
        with logging_redirect_tqdm():
            with multiprocessing.Pool(cfg.get('multiprocessing_cpus', 8), maxtasksperchild=100) as pool:
                for step in tqdm(range(run_params['duration']), desc="Steps"):
                    # First, record the gatekeeper behavior states
                    ds["behavior_mode"][0, step, :] = gk_cmd.collect_behaviors()

                    # We'll use the gatekeeper params for montecarlo control
                    if step >= run_params['warmup_steps']:
                        if step % gk_cmd.mc_period == 0:
                            # Returned dimensions are [n_controlled]
                            results = gk_cmd.run(pool)

                            # Record data
                            # ds["mc_loss"][0, i_mc, :, :] = results["losses"]
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
                    obs, reward, controlled_crashed, truncated, info = env.step(action)

                    # Record the actuals
                    ds["reward"][0, step, :] = reward
                    ds["crashed"][0, step, :] = controlled_crashed
                    ds["defensive_reward"][0, step, :] = info["rewards"]["defensive_reward"]
                    ds["speed_reward"][0, step, :] = info["rewards"]["speed_reward"]

                    # Print which, if any, av-IDs have crashed
                    crashed_ids = np.argwhere(controlled_crashed)
                    if crashed_ids.any():
                        av_ids = np.array(info['av_ids'])
                        crashed_set = set(av_ids[crashed_ids].flatten())
                        if crashed_set - crashed_vehicles:
                            logger.info(f"Crashed vehicles (Step {step}): {crashed_set}")
                            crashed_vehicles.update(crashed_set)

                    if len(uenv.crashed) >= 6:
                        # That's tooooo many -- probably a jam
                        # ds["time_to_collision"][i_world] = step
                        logger.info(f"Jam occurred (Step {step}). Exiting.")
                        break

                    if truncated:
                        # Times up
                        break

    else:
        # No Multiprocessing
        with logging_redirect_tqdm():
            for step in tqdm(range(run_params['duration']), desc="Steps"):
                # First, record the gatekeeper behavior states
                ds["behavior_mode"][0, step, :] = gk_cmd.collect_behaviors()

                # We'll use the gatekeeper params for montecarlo control
                if step >= run_params['warmup_steps']:
                    if step % gk_cmd.mc_period == 0:
                        # Returned dimensions are [n_controlled]
                        results = gk_cmd.run(None)

                        # Record data
                        # ds["mc_loss"][0, i_mc, :, :] = results["losses"]
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
                obs, reward, controlled_crashed, truncated, info = env.step(action)

                # Record the actuals
                ds["reward"][0, step, :] = reward
                ds["crashed"][0, step, :] = controlled_crashed
                ds["defensive_reward"][0, step, :] = info["rewards"]["defensive_reward"]
                ds["speed_reward"][0, step, :] = info["rewards"]["speed_reward"]

                # Print which, if any, av-IDs have crashed
                crashed_ids = np.argwhere(controlled_crashed)
                if crashed_ids.any():
                    av_ids = np.array(info['av_ids'])
                    crashed_set = set(av_ids[crashed_ids].flatten())
                    if crashed_set - crashed_vehicles:
                        logger.info(f"Crashed Control Vehicles (Step {step}): {crashed_set}")
                        crashed_vehicles.update(crashed_set)

                if len(uenv.crashed) >= 6:
                    # That's tooooo many -- probably a jam
                    # ds["time_to_collision"][i_world] = step
                    logger.info(f"Jam occurred (Step {step}). Exiting.")
                    break

                if truncated:
                    # Times up
                    break

        # Conclude the video
        env.close()

        if isinstance(env, recorder.AVRecorder):
            # Save frames
            np.save(f"{run_dir}/frames.npy", env.video_recorder.recorded_frames)

    # Append an extra data array "real_loss" to our dataset that is the negative of reward
    # Reward is normalized to [0,1]
    ds["real_loss"] = 1 - ds["reward"]

    # Automatically save latest
    checkpoint_dataset("Saving final results")


if __name__ == '__main__':
    main()
