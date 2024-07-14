import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sim import models, plotting, utils

RESULTS_DIR = "../results"


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    log_level = cfg.get("log_level", "INFO")
    if log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)

    # Get logger *after* setting the level
    logger = logging.getLogger("main")
    # Print our config
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    dynamics = getattr(models, cfg.DE_dynamics.model)(**cfg.DE_dynamics, seed=cfg.seed)
    revenue_model = getattr(models, cfg.revenue.model)(**cfg.revenue, seed=cfg.seed)
    cost_model = getattr(models, cfg.cost.model)(**cfg.cost, seed=cfg.seed)
    policy = getattr(models, cfg.policy.model)(
        revenue_model=revenue_model, cost_model=cost_model, **cfg.policy, seed=cfg.seed
    )
    loss_model = getattr(models, cfg.loss.model)(**cfg.loss, seed=cfg.seed)
    preference_prior = getattr(models, cfg.preference_prior.model)(**cfg.preference_prior, seed=cfg.seed)
    risk_model = getattr(models, cfg.risk.model)(preference_prior=preference_prior, **cfg.risk, seed=cfg.seed)

    # Linear spacing
    lambdas = np.arange(
        cfg.run_params.lmbda.min[0],
        cfg.run_params.lmbda.max[0],
        cfg.run_params.lmbda.step[0]
    )
    lambdas = np.round(lambdas, 2)

    lambda_to_qE_map = {}
    current_step = 0
    num_steps_per_iteration = int(1 / cfg.fish_params.qE.step)
    lambda_idx = 0
    current_lmbda = lambdas[lambda_idx]
    rts = []
    qEs = []
    while np.abs(current_lmbda - lambdas[-1]) > 1e-5:
        print('\nLooking for qE such that Risk(qE) = {}\n'.format(current_lmbda))

        current_step += 1
        qE_val = (current_step % num_steps_per_iteration) * cfg.fish_params.qE.step
        num_ones = current_step // num_steps_per_iteration
        qE = np.zeros(cfg.world_sim.plan_horizon)
        qE[:num_ones] = 1.
        qE[num_ones] = qE_val
        qEs.append(qE)

        p = utils.Params(
            cfg.fish_params.B0 * np.ones((1, cfg.run_params.num_param_batches)),
            cfg.fish_params.r * np.ones((1, cfg.run_params.num_param_batches)),
            cfg.fish_params.k * np.ones((1, cfg.run_params.num_param_batches)),
            # np.random.rand(1, cfg.run_params.num_param_batches),
            np.ones((1, cfg.run_params.num_param_batches)),
            qE
        )

        experimental_model = models.ConstrainedPolicyWorldModel(
            p,
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
        )
        experimental_model.policy.reset_apriori_policy(qE)

        es, rt = experimental_model()
        rts.append(rt)

        while rt > current_lmbda:
            lambda_to_qE_map[current_lmbda] = qE
            lambda_idx += 1
            if lambda_idx < len(lambdas):
                current_lmbda = lambdas[lambda_idx]
            else:
                break

    rts = np.array(rts)
    final_qE_value = qE.sum()
    qE_sums = np.arange(0., final_qE_value, cfg.fish_params.qE.step)

    # Store the outputs from this run as xArray Dataset
    qE_array = np.stack(qEs)
    lambda_results = utils.LambdaResults(qE_array, rts)
    ds = lambda_results.to_dataset()
    # # Automatically save latest
    latest_dir = f"{RESULTS_DIR}/latest"
    # os.makedirs(latest_dir, exist_ok=True)
    # OmegaConf.save(config=cfg, f=f"{latest_dir}/config.yaml")
    # lambda_results.save_ds(ds, f"{latest_dir}/lambdas_latest.nc")

    # # If a name is provided, save there too
    # if "name" in cfg:
    #     save_dir: str = cfg.get("save_dir", RESULTS_DIR)
    #     run_dir: str = os.path.join(save_dir, cfg.name)
    #     os.makedirs(run_dir, exist_ok=True)
    #     OmegaConf.save(config=cfg, f=f"{run_dir}/config.yaml")
    #     lambda_results.save_ds(ds, f"{run_dir}/lambdas_results.nc")

    # # Display a plot of results
    plotter = plotting.LambdaPlotter(ds)
    # plotter.lambda_plot(save_path=f"{latest_dir}/lambdas_latest.png")
    plotter.policy_plot(qE_array, rts, save_path=f"{latest_dir}/policy_latest.png")
    # plt.show()


if __name__ == '__main__':
    main()
