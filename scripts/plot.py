import numpy as np
import xarray as xr

from sim import plotting

# Create the video with the saved frames and data
# Load the data
RESULTS_DIR = "../results"
run_dir = RESULTS_DIR + "/latest"

dataset = xr.open_dataset(f"{run_dir}/results.nc")

ds_label_map = {
    "R_Def": "defensive_reward",
    "R_Spd": "speed_reward",
    "Actual Loss": "real_loss",
    # "E[Loss]": "loss_mean",
    # "E[Energy]": "energy",
    # "E[Entropy]": "entropy",
    # "Risk": "risk",
}

avplot = plotting.TrackerPlotter()
#
# avplot.multiagent_plot(
#     f"{run_dir}/multiagent.png",
#     dataset,
#     ds_label_map,
#     # max_av = 3,
#     # truncate = 0,
# )

# Also time the animation creation
frames = np.load(f"{run_dir}/frames.npy")
avplot.create_animation(
    f"{run_dir}/tracker.mp4",
    dataset,
    ds_label_map,
    frames,
    fps=10,
)

# plt.show()
