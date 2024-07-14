import logging
import os

import hydra
import jax.numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from sim import models, plotting, utils

RESULTS_DIR = "../results"


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    """
    Here, we simulate an evolving preference model through omega.
    Omega is sampled from a distribution (mu, sigma) at each episode.
    """
    log_level = cfg.get("log_level", "INFO")
    if log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)

    # Get logger *after* setting the level
    logger = logging.getLogger(__name__)
    # Print our config
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    dynamics = getattr(models, cfg.DE_dynamics.model)(**cfg.DE_dynamics, seed=cfg.seed)
    revenue_model = getattr(models, cfg.revenue.model)(**cfg.revenue, seed=cfg.seed)
    cost_model = getattr(models, cfg.cost.model)(**cfg.cost, seed=cfg.seed)
    policy = getattr(models, cfg.policy.model)(
        revenue_model=revenue_model, cost_model=cost_model, **cfg.policy, seed=cfg.seed
    )
    loss_model = getattr(models, cfg.loss.model)(**cfg.loss, seed=cfg.seed)

    preference_prior = getattr(models, cfg.preference_prior.model)(
        **cfg.preference_prior,
        seed=cfg.seed,
    )

    risk_model = getattr(models, cfg.risk.model)(preference_prior=preference_prior, **cfg.risk, seed=cfg.seed)

    # Override omega to have evolving values
    omega_params = cfg.run_params.omega
    if isinstance(omega_params, (dict, DictConfig)):
        if "evolution" in omega_params:
            omega_iter = utils.ParamIterator(**omega_params)
            # Init with first value
            omegas = omega_iter() * jnp.ones((1, cfg.run_params.num_param_batches))
        else:
            # Not implemented yet
            raise NotImplementedError("Undefined omega params")
    else:
        omega_iter = utils.ParamIterator(evolution="constant", x_0=omega_params)
        omegas = omega_iter() * jnp.ones((1, cfg.run_params.num_param_batches))

    # Expand
    params = utils.init_run_params(cfg, w=omegas)

    world_model = models.PreferenceEvolveWorldModel(
        params,
        cfg.run_params.num_param_batches,
        cfg.world_sim.n_montecarlo,
        cfg.world_sim.real_horizon,
        cfg.world_sim.plan_horizon,
        dynamics,
        policy,
        revenue_model,
        cost_model,
        loss_model,
        risk_model,
        omega_iter=omega_iter,
    )
    output = world_model()

    # Store the outputs from this run as xArray Dataset
    results = utils.EvolvePreferenceResults(output, cfg.world_sim.real_horizon, cfg.run_params.num_param_batches)
    ds = results.to_dataset()
    # Automatically save latest
    latest_dir = RESULTS_DIR + "/latest"
    os.makedirs(latest_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=f"{latest_dir}/config.yaml")
    results.save_ds(ds, f"{latest_dir}/pref_evolve_results.nc")

    # If a name is provided, save there too
    if "name" in cfg:
        save_dir: str = cfg.get("save_dir", RESULTS_DIR)
        run_dir: str = os.path.join(save_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=f"{run_dir}/config.yaml")
        results.save_ds(ds, f"{run_dir}/pref_evolve_results.nc")

    # Display a plot of results
    plotter = plotting.Plotter(ds)
    plotter.pref_evolve_plot(save_path=f"{latest_dir}/pref_evolve_results.png")
    plt.show()


if __name__ == '__main__':
    main()
