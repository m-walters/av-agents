import os

import numpy as np
import xarray as xr
import seaborn as sns

from sim import plotting

LABEL_TO_METRIC = {
    r'$R_D$': "defensive_reward",
    r'$R_S$': "speed_reward",
    "Loss": "real_loss",
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
    RESULTS_DIR = "../results/tmp/sample-baseline"
    frames = np.load(os.path.join(RESULTS_DIR, "frames.npy"))
    save_path = os.path.join(RESULTS_DIR, "anim.mp4")


    ds_label_map = {
        r'$R_D$': "defensive_reward",
        r'$R_S$': "speed_reward",
        "Loss": "real_loss",
        "E[Loss]": "loss_mean",
        "E[Energy]": "energy",
        "Risk": "risk",
    }

    datasets = [
        xr.open_dataset(os.path.join(RESULTS_DIR, "results.nc")),
    ]

    avplot = plotting.AVPlotter()
    colors = [
        (0.55, 0.66, 0.7),
        avplot.color_map["hotshot"],
    ]

    avplot.create_animation(
        save_path,
        datasets,
        ds_label_map,
        frames,
        fps=10,
        colors=colors,
    )


def double_animation():
    # Create the video with the saved frames and data
    # Load the data
    RESULTS_DIR = "../results/tmp"
    save_path = os.path.join(RESULTS_DIR, "double-anim.mp4")


    ds_label_map = {
        r'$R_D$': "defensive_reward",
        r'$R_S$': "speed_reward",
        "Loss": "real_loss",
        # "E[Loss]": "loss_mean",
        # "E[Energy]": "energy",
        # "Risk": "risk",
    }

    run1 = "sample-online"
    run2 = "sample-baseline"
    datasets = [
        xr.open_dataset(os.path.join(RESULTS_DIR, f"{run1}/results.nc")),
        xr.open_dataset(os.path.join(RESULTS_DIR, f"{run2}/results.nc")),
    ]
    frames = [
        np.load(os.path.join(RESULTS_DIR, f"{run1}/frames.npy")),
        np.load(os.path.join(RESULTS_DIR, f"{run2}/frames.npy")),
    ]
    video_labels = [
        "Online",
        "Hotshot",
    ]

    avplot = plotting.AVPlotter()
    colors = [
        (0.55, 0.66, 0.7),
        avplot.color_map["hotshot"],
    ]

    avplot.double_animation(
        save_path,
        datasets,
        ds_label_map,
        frames[0],
        frames[1],
        video_labels,
        fps=10,
        colors=colors,
    )


def multiagent_plot():
    # Create the video with the saved frames and data
    # Load the data
    RESULTS_DIR = "../results"
    run_dir = RESULTS_DIR + "/freezer/quick-ttc-gk-nc-8"

    dataset = xr.open_dataset(f"{run_dir}/results.nc")

    ds_label_map = {
        r'$R_D$': "defensive_reward",
        r'$R_S$': "speed_reward",
        "Loss": "real_loss",
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

    labels = [
        r'$R_D$',
        r'$R_S$',
        "Loss",
        # "E[Loss]",
        "E[Energy]",
        # "E[Entropy]",
        "Risk",
        "Crashed",
        "Fraction Defensive",
    ]
    # 4 rows, 2 columns
    axes_layout = [
        [r'$R_D$', None],
        [r'$R_S$', "Loss"],
        ["E[Energy]", "Risk"],
        ["Fraction Defensive", "Crashed"],
    ]

    # labels = [
    #     r'$R_D$',
    #     r'$R_S$',
    #     "Loss",
    #     "Crashed",
    # ]
    # # 4 rows, 2 columns
    # axes_layout = [
    #     [r'$R_D$', "Loss"],
    #     [r'$R_S$', "Crashed"],
    # ]

    metric_label_map = {k: LABEL_TO_METRIC[k] for k in labels}


    RESULTS_DIR = "../results/manuscript/freezer/crash-4-lrg"
    save_path = os.path.join(RESULTS_DIR, "figs/combined-metrics-66.pdf")
    data_tups = [
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-1/results.nc')), "NGK-1"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-2/results.nc')), "NGK-2"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-4/results.nc')), "4 Online"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-8/results.nc')), "NGK-8"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-12/results.nc')), "12 Online"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-nom/results.nc')), "Nom"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-def/results.nc')), "Def"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-latest-hotshot/results.nc')), "Hotshot"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-nom/results.nc')), "Nom"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-def/results.nc')), "Def."),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-hotshot/results.nc')), "Hotshot"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'ttc/results.nc')), "tmp"),
    ]
    styles = [
        "-",
        "-",
        "--",
        "--",
    ]
    avplot = plotting.AVPlotter(
        sns_context="paper", font_scale=1.3
    )
    skip_error = [
        False,
        False,
        True,
        True
    ]

    colors = [
        (0.66, 0.74, 0.66),
        (0.55, 0.66, 0.7),
        # (0.75, 0.75, 0.75),
        # (0.5, 0.5, 0.5),
        # avplot.color_map["online-4"],
        # avplot.color_map["online-12"],
        avplot.color_map["defensive"],
        avplot.color_map["hotshot"],
    ]

    avplot.comparison_plot(
        save_path,
        data_tups,
        metric_label_map,
        styles=styles,
        colors=colors,
        skip_error=skip_error,
        axes_layout=axes_layout,
        truncate=truncate,
        title=title,
    )


def ttc_hist():


    RESULTS_DIR = "../results/manuscript/freezer/crash-4-lrg"
    save_path = os.path.join(RESULTS_DIR, "baselines-ttc.png")
    data_tups = [
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-1/results.nc')), "NGK-1"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-2/results.nc')), "NGK-2"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-4/results.nc')), "NGK-4"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-8/results.nc')), "NGK-8"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'online-12/results.nc')), "NGK-12"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'multi/ttc-16/results.nc')), "NGK-16"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'multi/ttc-4/results.nc')), "NGK-4"),
        # (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-nom/results.nc')), "Nom"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-def/results.nc')), "Def"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'baselines-hotshot/results.nc')), "Hotshot"),
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
