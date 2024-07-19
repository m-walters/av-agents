import logging
import os
import shutil

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import sim.params as sim_params
from sim import models, plotting, utils, vehicle

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
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    # Get and validate the highway-env environment config
    env_cfg = cfg.env = utils.validate_env_config(cfg.env)

    # Initiate the pymc model and load the parameters
    params = []
    with pm.Model() as pm_model:
        for p in cfg.params:
            params.append(sim_params.load_param(p))

        param_collection = sim_params.ParamCollection(params)
        param_collection.draw(cfg.world_draws)

    v_l = cfg.v_l
    alpha, beta = cfg.loss.alpha, cfg.loss.beta
    p_star = cfg.preference_prior.p_star
    l_star = - alpha * np.exp(-(0.1 * v_l) ** 2 / alpha)

    loss_model = models.LossModel(alpha=alpha, beta=beta, v_l=v_l, seed=cfg.seed)
    preference_prior = models.ExponentialPreferencePrior(p_star=p_star, l_star=l_star, seed=cfg.seed)

    # For now we just look at a single ego vehicle and use the default policy
    policy = models.DefaultPolicy(seed=cfg.seed)
    ego = vehicle.Vehicle(
        policy=policy,
    )

    risk_model = getattr(models, cfg.risk.model)(
        preference_prior=preference_prior, **cfg.risk, seed=cfg.seed
    )

    env = gym.make('highway-v0', render_mode='rgb_array')
    env.reset()

    world_model = models.WorldModel(
        param_collection,
        cfg.world_draws,
        cfg.n_montecarlo,
        cfg.env.duration,
        cfg.plan_duration,
        ego,
        loss_model,
        risk_model,
    )
    output = world_model()

    return

    # Store the outputs from this run as xArray Dataset
    omega_results = utils.OmegaResults(omegas, outputs, cfg.duration, cfg.world_draws)
    ds = omega_results.to_dataset()
    # Automatically save latest
    latest_dir = RESULTS_DIR + "/latest"
    # Clear latest_dir if it exists
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)

    os.makedirs(latest_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=f"{latest_dir}/config.yaml")
    highway_results.save_ds(ds, f"{latest_dir}/highway_results.nc")

    # If a name is provided, save there too
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=f"{run_dir}/config.yaml")
        omega_results.save_ds(ds, f"{run_dir}/highway_results.nc")

    # Display a plot of results
    plotter = plotting.Plotter(ds)
    plotter.omega_quad_plot(save_path=f"{latest_dir}/highway_results.png")
    plt.show()


if __name__ == '__main__':
    main()
