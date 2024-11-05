import numpy as np
import xarray as xr
import os

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


def comparison_plot():
    # Comparison plot
    RESULTS_DIR = "../results"
    save_path = RESULTS_DIR + "/freezer/nc8/behavior_compare.png"

    data_tups = [
        (xr.open_dataset(f"{RESULTS_DIR}/freezer/nc8/ttc-baseline-cons/results.nc"), "Cons."),
        (xr.open_dataset(f"{RESULTS_DIR}/freezer/nc8/ttc-baseline-nom/results.nc"), "Nom."),
        (xr.open_dataset(f"{RESULTS_DIR}/freezer/nc8/ttc-baseline-hotshot/results.nc"), "Hotshot"),
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
    avplot.comparison_plot(
        save_path,
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

    RESULTS_DIR = "../results/manuscript/av-8"
    save_path = os.path.join(RESULTS_DIR, "trajectory_metrics.png")

    data_tups = [
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'nom/results.nc')), "Nominal"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'def/results.nc')), "Defensive"),
        (xr.open_dataset(os.path.join(RESULTS_DIR, 'hotshot/results.nc')), "Hotshot"),
    ]

    truncate = None

    # save_dir = RESULTS_DIR + "/tmp"
    # data_tups = [
    #     (xr.open_dataset(RESULTS_DIR + "/tmp/results.nc"), r'$\beta = 10$'),
    #     (xr.open_dataset(RESULTS_DIR + "/tmp/results.nc"), r'$\beta = 1$'),
    # ]

    # labels = [
    #     "R_Def",
    #     "R_Spd",
    #     "Actual Loss",
    #     "E[Loss]",
    #     "E[Energy]",
    #     "E[Entropy]",
    #     "Risk",
    #     "Crashed",
    #     "Number in Conservative",
    # ]
    # # 4 rows, 2 columns
    # axes_layout = [
    #     ["R_Def", "Actual Loss"],
    #     ["R_Spd", "E[Loss]"],
    #     ["E[Energy]", "Risk"],
    #     ["E[Entropy]", "Crashed"],
    #     [None, "Number in Conservative"],
    # ]

    labels = [
        "R_Def",
        "R_Spd",
        "Actual Loss",
        # "Crashed",
    ]
    # 4 rows, 2 columns
    axes_layout = [
        ["R_Def", "Actual Loss"],
        ["R_Spd", None],
    ]
    metric_label_map = {k: LABEL_TO_METRIC[k] for k in labels}

    avplot = plotting.AVPlotter()
    avplot.comparison_plot(
        save_path,
        data_tups,
        metric_label_map,
        axes_layout=axes_layout,
        truncate=truncate,
        title=title,
    )


def ttc_baseline_hist():
    """
    Plot the baseline histogram
    """
    RESULTS_DIR = "../results/"
    run_dir = os.path.join(RESULTS_DIR, "tmp/nc8/oval-av1-crash/")
    save_path = os.path.join(run_dir, "ttc-baseline-hist.png")

    avplot = plotting.AVPlotter()
    avplot.ttc_baselines_hist(
        save_path,
        nominal_ds=xr.open_dataset(os.path.join(run_dir, "ttc-baseline-nom/results.nc")),
        conservative_ds=xr.open_dataset(os.path.join(run_dir, "ttc-baseline-cons/results.nc")),
        hotshot_ds=xr.open_dataset(os.path.join(run_dir, "ttc-baseline-hotshot/results.nc")),
    )


def ttc_spec_plot():

    SPEC_BEHAVIORS = {
        "nom": "sim.vehicles.highway.NominalParams",
        # "cons": "sim.vehicles.highway.ConservativeParams",
        "def": "sim.vehicles.highway.DefensiveParams",
        "hotshot": "sim.vehicles.highway.HotshotParams",
        # "polite-incr": "sim.vehicles.highway.PolitenessIncr",
        # "polite-decr": "sim.vehicles.highway.PolitenessDecr",
        # "timedist-incr": "sim.vehicles.highway.TimeDistWantedIncr",
        # "timedist-decr": "sim.vehicles.highway.TimeDistWantedDecr",
        # "acc-max-incr": "sim.vehicles.highway.AccMaxIncr",
        # "acc-max-decr": "sim.vehicles.highway.AccMaxDecr",
        # "comf-brake-incr": "sim.vehicles.highway.ComfBrakingIncr",
        # "comf-brake-decr": "sim.vehicles.highway.ComfBrakingDecr",
        # "reckmax1": "sim.vehicles.highway.ReckMax1",
        # "reckmax2": "sim.vehicles.highway.ReckMax2",
        # "reckmax3": "sim.vehicles.highway.ReckMax3",
        # "def-HE": "sim.vehicles.highway.DefensiveHE",
        # "def-1": "sim.vehicles.highway.Defensive1",
        # "def-2": "sim.vehicles.highway.Defensive2",
    }

    RESULTS_DIR = "../results"
    run_dir = os.path.join(RESULTS_DIR, "manuscript/av-8")
    save_path = os.path.join(run_dir, "ttc-baseline-hist.png")

    avplot = plotting.AVPlotter()
    avplot.spec_baselines_hist(
        save_path,
        [
            (xr.open_dataset(os.path.join(run_dir, f"{name}/results.nc")), name) for name in SPEC_BEHAVIORS
        ]
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
        xr.open_dataset(RESULTS_DIR + "/quick-ttc-gk-nc-8/results.nc"),
    ]

    avplot = plotting.AVPlotter()
    avplot.ttc_vs_gk(
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
