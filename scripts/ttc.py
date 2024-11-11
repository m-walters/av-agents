"""
Multiagent Gatekeeper simulation for a Time-To-Collision mode
"""
import cProfile
import logging
import multiprocessing
import os
import time
import warnings

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

# Suppress specific warnings around our Env method overrides.
warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.utils\.passive_env_checker")


def non_mc_worldsim(
    world_idx: int, world_seed: int, env: AVRacetrack, any_control_collision: bool, profiler: cProfile.Profile | None
) -> tuple[int, dict]:
    """
    Run a single world simulation without MC
    """
    # Seed world
    obs, info = env.reset(seed=world_seed)
    uenv: "AVRacetrack" = env.unwrapped

    duration, n_controlled = uenv.config['duration'], uenv.config['controlled_vehicles']

    result = {
        "losses": np.full((duration, n_controlled), np.nan),
        "reward": np.full((duration, n_controlled), np.nan),
        "real_loss": np.full((duration, n_controlled), np.nan),
        "crashed": np.full((duration, n_controlled), np.nan),
        "defensive_reward": np.full((duration, n_controlled), np.nan),
        "speed_reward": np.full((duration, n_controlled), np.nan),
        "time_to_collision": np.nan,  # Or int if found later
    }

    with logging_redirect_tqdm():
        for step in tqdm(range(duration), desc="Steps", leave=False, disable=bool(profiler)):
            # We do action after MC sim in case it informs actions
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            # action = tuple(np.ones_like(action))
            obs, reward, crashed, truncated, info = env.step(action)

            # Record the actuals
            result["reward"][step, :] = reward
            # Reward is normalized to [0,1]
            result["real_loss"][step, :] = 1 - reward
            result["crashed"][step, :] = crashed
            result["defensive_reward"][step, :] = info["rewards"]["defensive_reward"]
            result["speed_reward"][step, :] = info["rewards"]["speed_reward"]

            # Check if any of control vehicle crashed
            if any_control_collision and any(crashed):
                # if crash_tagged_id in [v.av_id for v in uenv.crashed]:
                # Record the step which saw the first vehicle collision
                result["time_to_collision"] = step
                logger.info(f"Control vehicle collision (Step {step}): {crashed}\nExiting.")
                break
            elif not any_control_collision and uenv.controlled_vehicles[0].crashed:
                # We are tracking the first controlled vehicle and it crashed
                result["time_to_collision"] = step
                logger.info(f"Control vehicle collision (Step {step}): {crashed}\nExiting.")
                break

            if len(uenv.crashed) >= 6:
                # That's tooooo many -- probably a jam
                logger.info(f"Jam occurred (Step {step}). Exiting.")
                break

            if truncated:
                # Times up
                break

    return world_idx, result


def mc_worldsim(
    world_idx: int, world_seed: int, env: AVRacetrack, run_params: dict, gk_cfg: dict, any_control_collision: bool,
    profiler: cProfile.Profile | None
) -> tuple[int, dict]:
    """
    Gatekeep world sim for world-multiprocessing
    """
    # Seed world
    obs, info = env.reset(seed=world_seed)
    uenv: "AVRacetrack" = env.unwrapped

    duration, n_controlled = run_params['duration'], uenv.config['controlled_vehicles']
    num_mc_sweeps = len(run_params["mc_steps"])
    warmup_steps = run_params['warmup_steps']

    result = {
        "losses": np.full((duration, n_controlled), np.nan),
        "reward": np.full((duration, n_controlled), np.nan),
        "real_loss": np.full((duration, n_controlled), np.nan),
        "crashed": np.full((duration, n_controlled), np.nan),
        "defensive_reward": np.full((duration, n_controlled), np.nan),
        "speed_reward": np.full((duration, n_controlled), np.nan),
        "behavior_mode": np.full((duration, n_controlled), np.nan),
        "time_to_collision": np.nan,  # Or int if found later
        # MC data
        # "mc_loss": np.full((num_mc_sweeps, n_controlled), np.nan),
        "loss_mean": np.full((num_mc_sweeps, n_controlled), np.nan),
        "loss_p5": np.full((num_mc_sweeps, n_controlled), np.nan),
        "loss_p95": np.full((num_mc_sweeps, n_controlled), np.nan),
        "risk": np.full((num_mc_sweeps, n_controlled), np.nan),
        "entropy": np.full((num_mc_sweeps, n_controlled), np.nan),
        "energy": np.full((num_mc_sweeps, n_controlled), np.nan),
    }

    # Init the gatekeeper
    gk_cmd = gatekeeper.GatekeeperCommand(
        uenv, gk_cfg, uenv.controlled_vehicles, world_seed
    )

    with logging_redirect_tqdm():
        i_mc = 0  # Tracking MC steps
        for step in tqdm(range(duration), desc="Steps", leave=False, disable=bool(profiler)):
            # First, record the gatekeeper behavior states
            result["behavior_mode"][step, :] = gk_cmd.collect_behaviors()

            # We'll use the gatekeeper params for montecarlo control
            if step >= warmup_steps:
                if step % gk_cmd.mc_period == 0:
                    # Returned dimensions are [n_controlled]
                    results = gk_cmd.run()

                    # Record data
                    # result["mc_loss"][i_mc, :, :] = results["losses"]
                    result["loss_mean"][i_mc, :] = np.mean(results["losses"], axis=0)
                    result["loss_p5"][i_mc, :] = np.percentile(results["losses"], 5, axis=0)
                    result["loss_p95"][i_mc, :] = np.percentile(results["losses"], 95, axis=0)
                    result["risk"][i_mc, :] = results["risk"]
                    result["entropy"][i_mc, :] = results["entropy"]
                    result["energy"][i_mc, :] = results["energy"]
                    i_mc += 1

            # We do action after MC sim in case it informs actions
            # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
            action = env.action_space.sample()
            # action = tuple(np.ones_like(action))
            obs, reward, controlled_crashed, truncated, info = env.step(action)

            # Record the actuals
            result["reward"][step, :] = reward
            # Reward is normalized to [0,1]
            result["real_loss"][step, :] = 1 - reward
            result["crashed"][step, :] = controlled_crashed
            result["defensive_reward"][step, :] = info["rewards"]["defensive_reward"]
            result["speed_reward"][step, :] = info["rewards"]["speed_reward"]

            # Check if any of control vehicle crashed
            if any_control_collision and any(controlled_crashed):  # type: ignore
                # Record the step which saw the first vehicle collision
                result["time_to_collision"] = step
                logger.info(f"Control vehicle collision (Step {step}): {controlled_crashed}\nExiting.")
                break

            elif not any_control_collision and uenv.controlled_vehicles[0].crashed:
                # We are tracking the first controlled vehicle and it crashed
                result["time_to_collision"] = step
                logger.info(f"Control vehicle collision (Step {step}): {controlled_crashed}\nExiting.")
                break

            if len(uenv.crashed) >= 6:
                # That's tooooo many -- probably a jam
                logger.info(f"Jam occurred (Step {step}). Exiting.")
                break

            if truncated:
                # Times up
                break

    return world_idx, result


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

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.get('params', []):
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(world_draws)

    # Create our gym Env
    render_mode = None  # No visuals because of multiprocessing
    env = gym.make(f"AVAgents/{cfg.get('env_type', 'racetrack-v0')}", render_mode=render_mode)

    uenv: "AVRacetrack" = env.unwrapped
    uenv.update_config(env_cfg, reset=True)

    # Run world simulation
    rkey = utils.NpyRKey(seed)
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
    chkpt_time = time.time()
    world_loop_times = []

    # We track a single controlled vehicle for crash so that we can compare its TTC across runs
    any_control_collision = cfg.get("any_control_collision", True)

    if cfg.get("profiling", False):
        # Create profiler
        profiler = cProfile.Profile()
    else:
        profiler = None

    num_cpu = cfg.get('multiprocessing_cpus', 1)

    # If mc_steps is empty, this is a baseline run for collision testing
    if run_params['mc_steps'].size > 0:
        with logging_redirect_tqdm():
            with multiprocessing.Pool(num_cpu, maxtasksperchild=4) as pool:
                # i_world = -1  # tqdm doesn't handle the enumerate tuples well
                if profiler:
                    profiler.enable()

                for i_world in tqdm(
                        range(0, world_draws, num_cpu), desc="Worlds", unit_scale=num_cpu, disable=bool(profiler)
                ):

                    # Chunk the worlds by number of processes
                    end_world = min(i_world + num_cpu, world_draws)
                    pool_args = [
                        (
                            world_idx,
                            world_seeds[world_idx],
                            env,
                            run_params,
                            gk_cfg,
                            any_control_collision,
                            profiler
                        ) for world_idx in range(i_world, end_world)
                    ]

                    results = pool.starmap(
                        mc_worldsim,
                        pool_args,
                    )

                    for world_idx, result_dict in results:
                        ds["reward"][world_idx, :, :] = result_dict["reward"]
                        ds["real_loss"][world_idx, :, :] = result_dict["real_loss"]
                        ds["crashed"][world_idx, :, :] = result_dict["crashed"]
                        ds["defensive_reward"][world_idx, :, :] = result_dict["defensive_reward"]
                        ds["speed_reward"][world_idx, :, :] = result_dict["speed_reward"]
                        ds["time_to_collision"][world_idx] = result_dict["time_to_collision"]
                        # MC data
                        ds["loss_mean"][world_idx, :, :] = result_dict["loss_mean"]
                        ds["loss_p5"][world_idx, :, :] = result_dict["loss_p5"]
                        ds["loss_p95"][world_idx, :, :] = result_dict["loss_p95"]
                        ds["risk"][world_idx, :, :] = result_dict["risk"]
                        ds["entropy"][world_idx, :, :] = result_dict["entropy"]
                        ds["energy"][world_idx, :, :] = result_dict["energy"]

                #
                # for w_seed in tqdm(world_seeds, desc="Worlds", disable=bool(profiler)):
                #     # Seed world
                #     i_world += 1
                #     obs, info = env.reset(seed=w_seed)
                #     uenv: "AVRacetrack" = env.unwrapped
                #
                #     # Init the gatekeeper
                #     gk_cmd = gatekeeper.GatekeeperCommand(
                #         uenv, gk_cfg, uenv.controlled_vehicles, w_seed
                #     )
                #
                #     i_mc = 0  # Tracking MC steps
                #     for step in tqdm(range(run_params['duration']), desc="Steps", leave=False):
                #         # First, record the gatekeeper behavior states
                #         ds["behavior_mode"][i_world, step, :] = gk_cmd.collect_behaviors()
                #
                #         # We'll use the gatekeeper params for montecarlo control
                #         if step >= run_params['warmup_steps']:
                #             if step % gk_cmd.mc_period == 0:
                #                 # Returned dimensions are [n_controlled]
                #                 results = gk_cmd.run(pool)
                #
                #                 # Record data
                #                 # ds["mc_loss"][i_world, i_mc, :, :] = results["losses"]
                #                 ds["loss_mean"][i_world, i_mc, :] = np.mean(results["losses"], axis=0)
                #                 ds["loss_p5"][i_world, i_mc, :] = np.percentile(results["losses"], 5, axis=0)
                #                 ds["loss_p95"][i_world, i_mc, :] = np.percentile(results["losses"], 95, axis=0)
                #                 ds["risk"][i_world, i_mc, :] = results["risk"]
                #                 ds["entropy"][i_world, i_mc, :] = results["entropy"]
                #                 ds["energy"][i_world, i_mc, :] = results["energy"]
                #                 i_mc += 1
                #
                #         # We do action after MC sim in case it informs actions
                #         # For IDM-type vehicles, this doesn't really mean anything -- they do what they want
                #         action = env.action_space.sample()
                #         # action = tuple(np.ones_like(action))
                #         obs, reward, controlled_crashed, truncated, info = env.step(action)
                #
                #         # Record the actuals
                #         ds["reward"][i_world, step, :] = reward
                #         # Reward is normalized to [0,1]
                #         ds["real_loss"][i_world, step, :] = 1 - reward
                #         ds["crashed"][i_world, step, :] = controlled_crashed
                #         ds["defensive_reward"][i_world, step, :] = info["rewards"]["defensive_reward"]
                #         ds["speed_reward"][i_world, step, :] = info["rewards"]["speed_reward"]
                #
                #         # Check if any of control vehicle crashed
                #         if any_control_collision and any(controlled_crashed):  # type: ignore
                #             # Record the step which saw the first vehicle collision
                #             ds["time_to_collision"] = step
                #             logger.info(f"Control vehicle collision (Step {step}): {controlled_crashed}\nExiting.")
                #             break
                #
                #         elif not any_control_collision and uenv.controlled_vehicles[0].crashed:
                #             # We are tracking the first controlled vehicle and it crashed
                #             ds["time_to_collision"] = step
                #             logger.info(f"Control vehicle collision (Step {step}): {controlled_crashed}\nExiting.")
                #             break
                #
                #         if len(uenv.crashed) >= 6:
                #             # That's tooooo many -- probably a jam
                #             logger.info(f"Jam occurred (Step {step}). Exiting.")
                #             break
                #
                #         if truncated:
                #             # Times up
                #             break
                #
                #     # Checkpoint world draws
                #     if i_world + 1 < world_draws:
                #         checkpoint_dataset(f"Checkpointing World {i_world}")
                #
                #     t_lap = time.time()
                #     world_loop_times.append(t_lap - t_last)
                #     # chkpt_time += t_lap - t_last
                #     # if chkpt_time > checkpoint_interval:
                #     #     chkpt_time = 0
                #
                #     logger.info(f"World Loop: {t_lap - t_last:.2f}s (Avg: {np.mean(world_loop_times):.2f}s)")
                #     t_last = t_lap

    else:
        # No Gatekeeper MC
        # We are instead going to using the multiprocess pooling across the world draws, since no MC
        num_cpu = cfg.get('multiprocessing_cpus', 1)
        with logging_redirect_tqdm():
            with multiprocessing.Pool(num_cpu, maxtasksperchild=10) as pool:
                if profiler:
                    profiler.enable()
                    
                for i_world in tqdm(
                        range(0, world_draws, num_cpu), desc="Worlds", unit_scale=num_cpu, disable=bool(profiler)
                ):
                    # Chunk the worlds by number of processes
                    end_world = min(i_world + num_cpu, world_draws)
                    pool_args = [
                        (
                            world_idx,
                            world_seeds[world_idx],
                            env,
                            any_control_collision,
                            profiler
                        ) for world_idx in range(i_world, end_world)
                    ]

                    results = pool.starmap(
                        non_mc_worldsim,
                        pool_args,
                    )

                    for world_idx, result_dict in results:
                        ds["reward"][world_idx, :, :] = result_dict["reward"]
                        ds["real_loss"][world_idx, :, :] = result_dict["real_loss"]
                        ds["crashed"][world_idx, :, :] = result_dict["crashed"]
                        ds["defensive_reward"][world_idx, :, :] = result_dict["defensive_reward"]
                        ds["speed_reward"][world_idx, :, :] = result_dict["speed_reward"]
                        ds["time_to_collision"][world_idx] = result_dict["time_to_collision"]

    # Automatically save latest
    checkpoint_dataset("Saving final results")

    if profiler:
        profiler.disable()
        profiler.dump_stats(os.path.join(run_dir, "profiling.prof"))


if __name__ == '__main__':
    main()
