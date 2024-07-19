import logging
from typing import TYPE_CHECKING

import gymnasium as gym
import hydra
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import sim.params as sim_params
from sim import models, utils, vehicle

if TYPE_CHECKING:
    from gymnasium import Env

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

    env: "Env" = gym.make('highway-v0', render_mode='rgb_array')
    env.reset()

    world_model = models.WorldModel(
        env,
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

    # Test the display is working etc.
    for _ in range(cfg.env.duration):
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    plt.imshow(env.render())
    plt.show()

    return


if __name__ == '__main__':
    main()
