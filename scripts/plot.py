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
    run_dir = RESULTS_DIR + "/multiagent"
    run_dir2 = RESULTS_DIR + "/multiagent_gk"

    ds1 = xr.open_dataset(run_dir + "/results.nc")
    ds1_label = "No GK"
    ds2 = xr.open_dataset(run_dir2 + "/results.nc")
    ds2_label = "GK"

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
        f"{run_dir}/gatekeep_compare.png",
        ds1,
        ds1_label,
        ds2,
        ds2_label,
        metric_label_map,
        axes_layout=axes_layout,
        # truncate = 0,
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
