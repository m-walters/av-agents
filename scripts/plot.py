from matplotlib import pyplot as plt

from sim.plotting import Plotter

plotter = Plotter(
    "../results/latest/omega_results.nc",
    sns_context="talk"
)

save_path = "../results/latest/fig.png"

fig, axs = plotter.omega_quad_plot(
    # plot_kwargs={
    #     "figsize": (8, 4),
    # },
)
# fig, axs = plotter.pref_evolve_plot_2(
#     "omega",
#     "w",
#     # plot_kwargs={
#     #     "figsize": (8, 4),
#     # },
# )

# axs[1,1].set_ylim(bottom=0)

plt.savefig(save_path)

plt.show()
