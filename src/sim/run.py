"""
Utils for running simulations
"""
import logging
import os
import random
import shutil
from typing import Tuple, TypedDict

import numpy as np
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from sim import utils

RESULTS_DIR = "../../results"

logger = logging.getLogger("av-sim")


class RunParams(TypedDict):
    seed: int
    world_draws: int
    duration: int
    warmup_steps: int
    mc_period: int
    mc_steps: np.ndarray
    n_montecarlo: int


def validate_env_config(cfg: DictConfig) -> DictConfig:
    """
    Validate the highway-env Environment config
    """
    return cfg


def init_results_dataset(
    world_draws: int,
    duration: int,
    mc_steps: utils.Array,
    n_montecarlo: int,
) -> xr.Dataset:
    num_mc_sweeps = len(mc_steps)
    return xr.Dataset(
        {
            ### Data recorded every world step
            # Rewards
            "reward": (("world", "step"), np.full((world_draws, duration), np.nan)),
            "collision_reward": (("world", "step"), np.full((world_draws, duration), np.nan)),
            "speed_reward": (("world", "step"), np.full((world_draws, duration), np.nan)),
            "crashed": (("world", "step"), np.full((world_draws, duration), np.nan)),
            ### Data recorded from MC Sweeps
            "risk": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),
            "entropy": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),
            "energy": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),
            # Tracking the MC losses -- These are predicted losses
            "mc_loss": (
                ("world", "mc_step", "sample"), np.full((world_draws, num_mc_sweeps, n_montecarlo), np.nan)),
            "loss_mean": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),
            "loss_p5": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),  # 5% percentile
            "loss_p95": (("world", "mc_step"), np.full((world_draws, num_mc_sweeps), np.nan)),  # 95% percentile
        },
        coords={
            "world": np.arange(world_draws),
            "step": np.arange(duration),
            "mc_step": mc_steps,
            "sample": np.arange(n_montecarlo),
        },
    )


def init(cfg: DictConfig) -> Tuple[DictConfig, "RunParams", xr.Dataset]:
    """
    Process the config, set up some objects etc.
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

    env_cfg = cfg.highway_env = validate_env_config(cfg.highway_env)

    # Seed RNG
    seed = cfg.get("seed", None) or np.random.randint(1e6.__int__())
    logger.warning(f"No seed provided. Using random seed: {seed}")
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

    # Create an xarray Dataset
    duration = int(env_cfg.duration)
    world_draws = int(cfg.world_draws)
    warmup_steps = cfg.get("warmup_steps", 0)

    # Monte Carlo logistics
    mc_period = env_cfg.get("mc_period", 5)
    mc_steps = np.arange(warmup_steps, duration, mc_period)
    n_montecarlo = env_cfg['n_montecarlo']

    ds = init_results_dataset(
        world_draws, duration, mc_steps, n_montecarlo
    )

    # Convert to py-dict so we can record the seed in case this is a random run
    py_cfg = OmegaConf.to_container(cfg, resolve=True)
    py_cfg['seed'] = seed
    OmegaConf.save(config=py_cfg, f=f"{latest_dir}/config.yaml")

    run_params = {
        "seed": seed,
        "world_draws": world_draws,
        "duration": duration,
        "warmup_steps": warmup_steps,
        "mc_period": mc_period,
        "mc_steps": mc_steps,
        "n_montecarlo": n_montecarlo,
    }

    return cfg, run_params, ds
