from matplotlib import pyplot as plt
from sim.plotting import AVPlotter

plotter = AVPlotter(
    "../results/latest/results.nc",
    sns_context="paper"
)

save_path = "../results/latest/fig.png"

fig, axs = plotter.quad_plot(
    # plot_kwargs={
    #     "figsize": (8, 4),
    # },
)

plt.savefig(save_path)

plt.show()
