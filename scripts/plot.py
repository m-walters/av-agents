from matplotlib import pyplot as plt
from sim.plotting import Plotter
import gym

def plot_highway_env(env, plotter):
    plt.figure(figsize=(10, 5)) #took random size for this 
    plt.imshow(env.render())
    plt.title("Highway Environment")
    plt.axis('off') 

plt.tight_layout() # layout adjustment acc to the dimensions 10 / 5
    plt.savefig("../results/highway_env_plot.png")
    plt.close()

plotter = Plotter(      
    "../results/latest/highway.nc",
    sns_context="talk"
)

save_path = "../results/latest/fig.png"

plt.savefig(save_path)

plt.show()

env.close()
