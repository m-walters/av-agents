from matplotlib import pyplot as plt

from sim.plotting import Plotter

plotter = Plotter(
    "../results/latest/highway.nc",
    sns_context="talk"
)

save_path = "../results/latest/fig.png"

plt.savefig(save_path)

plt.show()
