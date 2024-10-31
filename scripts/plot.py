import numpy as np
import xarray as xr

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
    "Fraction Conservative": "behavior_mode",
    "Number in Conservative": "behavior_mode",
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


def gatekeep_compare():
    # Gatekeep comparing
    RESULTS_DIR = "../results"
    save_dir = RESULTS_DIR + "/freezer/quick-ttc-gk-nc-8"
    no_gk_dir = RESULTS_DIR + "/freezer/quick-ttc-gk-nc-8"
    gk_dir = RESULTS_DIR + "/freezer/quick-ttc-gk-nc-8"

    data_tups = [
        (xr.open_dataset(no_gk_dir + "/results.nc"), "No GK"),
        (xr.open_dataset(gk_dir + "/results.nc"), "GK"),
    ]

    labels = [
        "R_Def",
        "R_Spd",
        "Actual Loss",
        "E[Loss]",
        "E[Energy]",
        "E[Entropy]",
        "Risk",
        "Crashed",
        "Number in Conservative",
    ]
    metric_label_map = {k: LABEL_TO_METRIC[k] for k in labels}

    # 4 rows, 2 columns
    axes_layout = [
        ["R_Def", "Actual Loss"],
        ["R_Spd", "E[Loss]"],
        ["E[Energy]", "Risk"],
        ["E[Entropy]", "Crashed"],
        ["Number in Conservative", None],
    ]

    avplot = plotting.AVPlotter()
    avplot.multiagent_comparison_plot(
        f"{save_dir}/gatekeep_compare.png",
        data_tups,
        metric_label_map,
        axes_layout=axes_layout,
        # truncate = 0,
    )


def compare_plot():
    """
    Generic comparison plot
    """
    title = None

    RESULTS_DIR = "../results/freezer/seeds/seed-86777"
    save_dir = RESULTS_DIR + "/"
    data_tups = [
        (xr.open_dataset(RESULTS_DIR + "/gk-lstar-0p1/results.nc"), r'$L^*=0.1$'),
        (xr.open_dataset(RESULTS_DIR + "/gk-lstar-0p2/results.nc"), r'$L^*=0.2$'),
        (xr.open_dataset(RESULTS_DIR + "/gk-lstar-0p5/results.nc"), r'$L^*=0.5$'),
        (xr.open_dataset(RESULTS_DIR + "/gk-lstar-0p9/results.nc"), r'$L^*=0.9$'),
    ]

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
        "Number in Conservative",
    ]
    metric_label_map = {k: LABEL_TO_METRIC[k] for k in labels}

    # 4 rows, 2 columns
    axes_layout = [
        ["R_Def", "Actual Loss"],
        ["R_Spd", "E[Loss]"],
        ["E[Energy]", "Risk"],
        ["E[Entropy]", "Crashed"],
        [None, "Number in Conservative"],
    ]

    avplot = plotting.AVPlotter()
    avplot.multiagent_comparison_plot(
        f"{save_dir}/temp_compare.png",
        data_tups,
        metric_label_map,
        axes_layout=axes_layout,
        truncate=truncate,
    )


def baseline_hist():
    """
    Plot the baseline histogram
    """
    RESULTS_DIR = "../results/freezer/"
    save_dir = RESULTS_DIR + "/"

    avplot = plotting.AVPlotter()
    avplot.ttc_baselines_hist(
        f"{save_dir}/ttc_baseline_hist.png",
        nominal_ds=xr.open_dataset(RESULTS_DIR + "/ttc_nom_baseline/results.nc"),
        conservative_ds=xr.open_dataset(RESULTS_DIR + "/ttc_conservative_baseline/results.nc"),
    )


def ttc_vs_gk():
    """
    Time-to-Collision vs Number controlled by GK
    """
    title = None

    RESULTS_DIR = "../results/freezer"
    save_dir = RESULTS_DIR + "/"
    data = [
        xr.open_dataset(RESULTS_DIR + "/quick-ttc-gk-nc-1/results.nc"),
        xr.open_dataset(RESULTS_DIR + "/quick-ttc-gk-nc-2/results.nc"),
        xr.open_dataset(RESULTS_DIR + "/quick-ttc-gk-nc-4/results.nc"),
        # xr.open_dataset(RESULTS_DIR + "/quick-ttc-gk-nc-8/results.nc"),
    ]

    avplot = plotting.AVPlotter()
    avplot.ttc_vs_num_gk(
        f"{save_dir}/ttc-vs-gk.png",
        data
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
