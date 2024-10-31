"""
Multiagent Gatekeeper simulation for a Time-To-Collision mode
"""
import logging
import multiprocessing
import os
import time

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from omegaconf import DictConfig
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import sim.params as sim_params
from sim import gatekeeper, run, utils
from sim.envs.racetrack import AVRacetrack

# Name of file in configs, set this to your liking
DEFAULT_CONFIG = "ttc"

logger = logging.getLogger("av-sim")


@hydra.main(version_base=None, config_path="../configs", config_name=DEFAULT_CONFIG)
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    cfg, run_params, gk_cfg = run.init(cfg)
    run_dir = cfg.run_dir

    ds, behavior_index = run.init_multiagent_results_dataset(
        run_params['world_draws'], run_params['duration'], run_params['mc_steps'],
        run_params['n_montecarlo'], gk_cfg['n_controlled']
    )
    seed = run_params['seed']
    world_draws = run_params['world_draws']
    env_cfg = cfg.highway_env
    use_mp = cfg.get('use_mp', False)
    if not use_mp:
        raise ValueError("Time-To-Collision simulation only configured for multiprocessing")

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(world_draws)

    # Create our gym Env
    render_mode = None  # No visuals because of multiprocessing
    env = gym.make('AVAgents/racetrack-v0', render_mode=render_mode)
    uenv: "AVRacetrack" = env.unwrapped
    uenv.update_config(env_cfg, reset=True)

    # Run world simulation
    rkey = utils.JaxRKey(seed)
    # Generate seeds
    world_seeds = rkey.next_seeds(world_draws)

    def checkpoint_dataset(msg: str = None):
        """
        Save the dataset to disk
        """
        if msg:
            logger.info(msg)
        utils.Results.save_ds(ds, f"{run_dir}/results.nc")
        # Save behavior index
        utils.Results.save_json(behavior_index, f"{run_dir}/behavior_index.json")

    t_last = time.time()
    checkpoint_interval = 1 * 30  # 5 minutes
    chkpt_time = time.time()
    world_loop_times = []

    # If mc_steps is empty, this is a baseline run for collision testing
    # So we don't need to do multiprocessing
    if run_params['mc_steps'].size > 0:
        with logging_redirect_tqdm():
            with multiprocessing.Pool(cfg.get('multiprocessing_cpus', 8), maxtasksperchild=100) as pool:
                i_world = -1  # tqdm doesn't handle the enumerate tuples well
                for w_seed in tqdm(world_seeds, desc="Worlds"):
                    # Seed world
                    i_world += 1
                    obs, info = env.reset(seed=w_seed)
                    uenv: "AVRacetrack" = env.unwrapped
                    i_mc = 0  # Tracking MC steps

                    # Init the gatekeeper
                    gk_cmd = gatekeeper.GatekeeperCommand(
                        uenv, gk_cfg, uenv.controlled_vehicles, w_seed
                    )

                    for step in tqdm(range(run_params['duration']), desc="Steps", leave=False):
                        # First, record the gatekeeper behavior states
                        ds["behavior_mode"][i_world, step, :] = gk_cmd.collect_behaviors()

                        # We'll use the gatekeeper params for montecarlo control
                        if step >= run_params['warmup_steps']:
                            if step % gk_cmd.mc_period == 0:
                                # Returned dimensions are [n_controlled]
                                results = gk_cmd.run(pool)

                                # Record data
                                ds["mc_loss"][i_world, i_mc, :, :] = results["losses"]
                                ds["loss_mean"][i_world, i_mc, :] = np.mean(results["losses"], axis=0)
                                ds["loss_p5"][i_world, i_mc, :] = np.percentile(results["losses"], 5, axis=0)
                                ds["loss_p95"][i_world, i_mc, :] = np.percentile(results["losses"], 95, axis=0)
                                ds["risk"][i_world, i_mc, :] = results["risk"]
                                ds["entropy"][i_world, i_mc, :] = results["entropy"]
                                ds["energy"][i_world, i_mc, :] = results["energy"]
                                i_mc += 1

                        # We do action after MC sim in case it informs actions
                        # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
                        action = env.action_space.sample()
                        # action = tuple(np.ones_like(action))
                        obs, reward, controlled_crashed, truncated, info = env.step(action)

                        # Record the actuals
                        ds["reward"][i_world, step, :] = reward
                        # Reward is normalized to [0,1]
                        ds["real_loss"][i_world, step, :] = 1 - reward
                        ds["crashed"][i_world, step, :] = controlled_crashed
                        ds["defensive_reward"][i_world, step, :] = info["rewards"]["defensive_reward"]
                        ds["speed_reward"][i_world, step, :] = info["rewards"]["speed_reward"]

                        # We are concerned with any collisions on the map so
                        if uenv.any_crashed():
                            # Record the TTC crash and exit
                            ds["time_to_collision"][i_world] = step
                            logger.info(f"Crashed vehicles (Step {step}). Exiting.")
                            break

                        if truncated:
                            # Times up
                            break

                    t_lap = time.time()
                    world_loop_times.append(t_lap - t_last)
                    chkpt_time += t_lap - t_last
                    if chkpt_time > checkpoint_interval:
                        checkpoint_dataset(f"Checkpointing World {i_world}")
                        chkpt_time = 0

                    logger.info(f"World Loop: {t_lap - t_last:.2f}s (Avg: {np.mean(world_loop_times):.2f}s)")
                    t_last = t_lap


    else:
        # No Gatekeeper MC
        i_world = -1
        with logging_redirect_tqdm():
            for w_seed in tqdm(world_seeds, desc="Worlds"):
                i_world += 1
                # Seed world
                obs, info = env.reset(seed=w_seed)
                uenv: "AVRacetrack" = env.unwrapped

                for step in tqdm(range(run_params['duration']), desc="Steps", leave=False):
                    # We do action after MC sim in case it informs actions
                    # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
                    action = env.action_space.sample()
                    # action = tuple(np.ones_like(action))
                    obs, reward, crashed, truncated, info = env.step(action)

                    # Record the actuals
                    ds["reward"][i_world, step, :] = reward
                    # Reward is normalized to [0,1]
                    ds["real_loss"][i_world, step, :] = 1 - reward
                    ds["crashed"][i_world, step, :] = crashed
                    ds["defensive_reward"][i_world, step, :] = info["rewards"]["defensive_reward"]
                    ds["speed_reward"][i_world, step, :] = info["rewards"]["speed_reward"]

                    # We are concerned with any collisions on the map so
                    if uenv.any_crashed():
                        # Record the TTC crash and exit
                        ds["time_to_collision"][i_world] = step
                        logger.info(f"Crashed vehicles (Step {step}). Exiting.")
                        break

                    if truncated:
                        # Times up
                        break

                t_lap = time.time()
                world_loop_times.append(t_lap - t_last)
                chkpt_time += t_lap - t_last
                if chkpt_time > checkpoint_interval:
                    checkpoint_dataset(f"Checkpointing World {i_world}")
                    chkpt_time = 0

                logger.info(f"World Loop: {t_lap - t_last:.2f}s (Avg: {np.mean(world_loop_times):.2f}s)")
                t_last = t_lap

    # Automatically save latest
    checkpoint_dataset("Saving final results")


if __name__ == '__main__':
    main()
