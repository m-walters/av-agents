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
    no_gk_dir = RESULTS_DIR + "/freezer/multiagent"
    gk_dir = RESULTS_DIR + "/freezer/multiagent_gk"

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
        f"{gk_dir}/gatekeep_compare.png",
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

    RESULTS_DIR = "../results"
    save_dir = RESULTS_DIR + "/"
    data_tups = [
        # (xr.open_dataset(RESULTS_DIR + "/beta_10/results.nc"), r'$\beta = 10$'),
        # (xr.open_dataset(RESULTS_DIR + "/beta_2/results.nc"), r'$\beta = 2$'),
        # (xr.open_dataset(RESULTS_DIR + "/beta_1/results.nc"), r'$\beta = 1$'),
        # (xr.open_dataset(RESULTS_DIR + "/beta_0p5/results.nc"), r'$\beta = 0.5$'),
        # (xr.open_dataset(RESULTS_DIR + "/beta_0p1/results.nc"), r'$\beta = 0.1$'),
        (xr.open_dataset(RESULTS_DIR + "/pstar-0p01/results.nc"), r'$p^* = 0.01, L^*=0.9$'),
        (xr.open_dataset(RESULTS_DIR + "/pstar-0p1/results.nc"), r'$p^* = 0.1, L^*=0.9$'),
        (xr.open_dataset(RESULTS_DIR + "/pstar-0p9/results.nc"), r'$p^* = 0.9, L^*=0.9$'),
        # (xr.open_dataset(RESULTS_DIR + "/pstar-0p01_lstar-0p1/results.nc"), r'$p^* = 0.01, L^*=0.1$'),
        # (xr.open_dataset(RESULTS_DIR + "/pstar-0p1_lstar-0p1/results.nc"), r'$p^* = 0.01, L^*=0.1$'),
        # (xr.open_dataset(RESULTS_DIR + "/pstar-0p9_lstar-0p1/results.nc"), r'$p^* = 0.01, L^*=0.1$'),

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
