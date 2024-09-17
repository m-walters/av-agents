import timeit

import numpy as np
import xarray as xr

from sim import plotting

# Create the video with the saved frames and data
# Load the data
RESULTS_DIR = "../results"
run_dir = RESULTS_DIR + "/mc-10period"

frames = np.load(f"{run_dir}/frames.npy")
dataset = xr.open_dataset(f"{run_dir}/results.nc")

ds_label_map = {
    "R_Coll": "collision_reward",
    "R_Spd": "speed_reward",
    "Actual Loss": "real_loss",
    "E[Loss]": "loss_mean",
    "E[Energy]": "energy",
    "E[Entropy]": "entropy",
    "Risk": "risk",
}

# Plot the first frame as an image
# fig, axs = plt.subplots(4, 1)
# ax1, data_axs = axs[0], axs[1:]
# for ax in data_axs[:-1]:
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_xlabel('')
#
# ax1.axis('off')
# r = ax1.imshow(frames[0])
#


avplot = plotting.TrackerPlotter()

# Also time the animation creation
t = timeit.timeit(
    lambda: avplot.create_animation(
        f"{run_dir}/tracker.mp4",
        dataset,
        ds_label_map,
        frames,
        fps=10,
    ),
    number=1,

)
print(f"Time to create animation: {t:.2f} s")

# plt.show()
