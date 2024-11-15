import os

import numpy as np
import xarray as xr
import seaborn as sns

from sim import plotting

LABEL_TO_METRIC = {
    "R_Def": "defensive_reward",
    "R_Spd": "speed_reward",
    "Actual Loss": "real_loss",
    "E[Loss]": "loss_mean",
    "E[Energy]": "energy",
    "E[Entropy]": "entropy",
    "Risk": "risk",
    "Fraction Ego Crashed": "crashed",
    "Crashed": "crashed",
    "Fraction Defensive": "behavior_mode",
}


def animate_single():
    # Create the video with the saved frames and data
    # Load the data
    RESULTS_DIR = "../results"
    run_dir = RESULTS_DIR + "/multiagent"

    dataset = xr.open_dataset(f"{run_dir}/results.nc")

    ds_label_map = {
        "R_Def": "defensive_reward",
        "R_Spd": "speed_reward",
        "Actual Loss": "real_loss",
        "E[Loss]": "loss_mean",
        "E[Energy]": "energy",
        "E[Entropy]": "entropy",
        "Risk": "risk",
    }

    avplot = plotting.AVPlotter()
    frames = np.load(f"{run_dir}/frames.npy")
    avplot.create_animation(
        f"{run_dir}/anim_plot.mp4",
        dataset,
        ds_label_map,
        frames,
        fps=10,
    )


def multiagent_plot():
    # Create the video with the saved frames and data
    # Load the data
    RESULTS_DIR = "../results"
    run_dir = RESULTS_DIR + "/freezer/quick-ttc-gk-nc-8"

    dataset = xr.open_dataset(f"{run_dir}/results.nc")

    ds_label_map = {
        "R_Def": "defensive_reward",
        "R_Spd": "speed_reward",
        "Actual Loss": "real_loss",
        "E[Loss]": "loss_mean",
        "E[Energy]": "energy",
        "E[Entropy]": "entropy",
        "Risk": "risk",
    }

    avplot = plotting.AVPlotter()
    avplot.multiagent_plot(
        f"{run_dir}/multiagent.png",
        dataset,
        ds_label_map,
        # max_av = 3,
        # truncate = 0,
    )


def compare_plot():
    """
    Generic comparison plot
    """
    title = None

    truncate = None

    # save_dir = RESULTS_DIR + "/tmp"
    # data_tups = [
    #     (xr.open_dataset(RESULTS_DIR + "/tmp/results.nc"), r'$\beta = 10$'),
    #     (xr.open_dataset(RESULTS_DIR + "/tmp/results.nc"), r'$\beta = 1$'),
    # ]

    labels = [
        "R_Def",
        "R_Spd",
        "Actual Loss",
        "E[Loss]",
        "E[Energy]",
        "E[Entropy]",
        "Risk",
        "Crashed",
        "Fraction Defensive",
    ]
    # 4 rows, 2 columns
    axes_layout = [
        ["R_Def", "Actual Loss"],
        ["R_Spd", "E[Loss]"],
        ["E[Energy]", "Risk"],
        ["E[Entropy]", "Crashed"],
        [None, "Fraction Defensive"],
    ]

    # labels = [
    #     "R_Def",
    #     "R_Spd",
    #     "Actual Loss",
    #     "Crashed",
    # ]
    # # 4 rows, 2 columns
    # axes_layout = [
    #     ["R_Def", "Actual Loss"],
    #     ["R_Spd", "Crashed"],
    # ]
    metric_label_map = {k: LABEL_TO_METRIC[k] for k in labels}


    # RESULTS_DIR = "../results/manuscript/freezer/crash-4"
    RESULTS_DIR = "../results/tmp/"
    save_path = os.path.join(RESULTS_DIR, "combined-metrics.png")
    data_tups = [
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-1/results.nc')), "NGK-1"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-2/results.nc')), "NGK-2"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-4/results.nc')), "NGK-4"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-8/results.nc')), "NGK-8"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-12/results.nc')), "NGK-12"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-nom/results.nc')), "Nom"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-def/results.nc')), "Def"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-hotshot/results.nc')), "Hotshot"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-nom/results.nc')), "Nom"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-def/results.nc')), "Def"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-hotshot/results.nc')), "Hotshot"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'ttc/results.nc')), "tmp"),
    ]
    styles = [
        "-",
        "-",
        "--",
        "--",
    ]
    avplot = plotting.AVPlotter(
        sns_context="notebook"
    )

    colors = [
        avplot.color_map["online-4"],
        avplot.color_map["online-12"],
        avplot.color_map["defensive"],
        avplot.color_map["hotshot"],
    ]

    avplot.comparison_plot(
        save_path,
        data_tups,
        metric_label_map,
        styles=styles,
        colors=colors,
        axes_layout=axes_layout,
        truncate=truncate,
        title=title,
    )


def ttc_hist():


    RESULTS_DIR = "../results/freezer/crash-4"
    save_path = os.path.join(RESULTS_DIR, "online-exp-ttc.png")
    data_tups = [
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-1/results.nc')), "NGK-1"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-2/results.nc')), "NGK-2"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-4/results.nc')), "NGK-4"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-8/results.nc')), "NGK-8"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-12/results.nc')), "NGK-12"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'multi/ttc-16/results.nc')), "NGK-16"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'multi/ttc-4/results.nc')), "NGK-4"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-nom/results.nc')), "Nom"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-def/results.nc')), "Def"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-hotshot/results.nc')), "Hotshot"),
    ]

    # bin_range = (0, 20)
    bin_range = None
    kde = False

    avplot = plotting.AVPlotter()
    avplot.ttc_hist(
        save_path,
        data_tups,
        bin_range=bin_range,
        kde=kde,
    )


def ttc_vs_online():
    """
    Time-to-Collision vs Number controlled by GK
    """
    title = None

    RESULTS_DIR = "../results/manuscript/freezer/crash-4"
    save_path = os.path.join(RESULTS_DIR, "ttc-violin.png")
    datasets = [
        xr.open_dataset(os.path.join(RESULTS_DIR, 'online-4/results.nc')),
        xr.open_dataset(os.path.join(RESULTS_DIR, 'online-12/results.nc')),
    ]

    avplot = plotting.AVPlotter()
    avplot.ttc_vs_online(
        save_path,
        datasets
    )


if __name__ == '__main__':
    # Accept first argument as function name to be called
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to call")

    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    print(f"Calling {func_name}")
    globals()[func_name]()
