from typing import Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

"""
Note https://docs.xarray.dev/en/stable/user-guide/plotting.html
for plotting with xarray objects
"""


class Plotter:
    """
    Plotting util for xArray Datasets
    """

    def __init__(
        self,
        ds_or_path: Union[xr.Dataset, str],
        sns_context: str = "notebook",  # or "paper", "talk", "poster"
    ):
        if isinstance(ds_or_path, xr.Dataset):
            self.ds = ds_or_path
        else:
            self.ds = xr.open_dataset(ds_or_path)

        # Initialize seaborn
        sns.set()
        sns.set_context(sns_context)
        sns.set_palette("colorblind")

    @staticmethod
    def get_color_wheel():
        """
        Return a color generator for the current seaborn palette
        """
        return iter(sns.color_palette())

    @staticmethod
    def subplots(nrow, ncol, **kwargs):
        return plt.subplots(nrow, ncol, **kwargs)


class AVPlotter(Plotter):
    """
    Plotter for run results
    """

    def quad_plot(
        self, fig=None, axs=None, save_path=None, plot_kwargs: dict | None = None,
        shadow: bool = False
    ):
        """
        Generate a 2x2 plot with Energy, Entropy, Loss, Risk
        In each plot, we reduce across the world draw axis

        'shadow' param toggles whether the plot does shadow error bars, or individual lines across world draws
        """
        plot_kwargs = plot_kwargs or {}

        if axs is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 6))
            fig, axs = self.subplots(2, 2, sharex=True, **plot_kwargs)
            plt.subplots_adjust(wspace=0.4)

        # Note: match keys the dataset variable names
        ax_map = {
            'energy': axs[0, 0],
            'entropy': axs[0, 1],
            'loss_mean': axs[1, 0],
            'risk': axs[1, 1],
        }
        labels = {
            'energy': 'E',
            'entropy': 'H',
            'loss_mean': 'Loss',
            'risk': 'Risk',
        }

        colors = self.get_color_wheel()
        color = next(colors)
        ls = "-"
        for i, var in enumerate(ax_map):
            # This produces a pivot table with time as index and batch as columns
            pivot = self.ds[var].to_pandas()
            # label = f"w={np.round(omega.values, 2)}"

            if shadow:
                # Melt it to go from wide to long form, with world as a variable, and our var as value
                melted = pivot.T.melt(var_name='world', value_name=var, ignore_index=False)
                sns.lineplot(
                    x="step", y=var, data=melted, color=color, ls=ls, ax=ax_map[var], legend=False,
                    label=None,
                )
            else:
                # Plot each world draw as a separate line; same color though
                data = pivot.T
                for world in self.ds.world:
                    # ax_map[var].plot(pivot.index, pivot[world], color=color, alpha=0.3, label=f'World {world}')
                    sns.lineplot(
                        x=data.index.values, y=data[int(world)], color=color, alpha=0.3, ls=ls, ax=ax_map[var],
                        legend=False, label=None,
                    )

            # Set ax title
            ax_map[var].set_ylabel(labels[var])

        # Title
        # fig.suptitle('')

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel("Step")
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, axs


class TrackerPlotter:
    """
    For plotting data in tandem during a sim recording
    """

    def __init__(
        self,
        sns_context: str = "paper"
    ):
        # Initialize seaborn
        sns.set()
        sns.set_context(sns_context)
        sns.set_palette("colorblind")

    @staticmethod
    def get_color_wheel():
        """
        Return a color generator for the current seaborn palette
        """
        return iter(sns.color_palette())

    @staticmethod
    def subplots(nrow, ncol, **kwargs):
        return plt.subplots(nrow, ncol, **kwargs)

    def create_animation(
        self,
        save_path: str,
        ds_or_path: Union[xr.Dataset, str],
        ds_label_map: dict,
        sim_frames: list[np.ndarray],
        fps: int = 30,
    ):
        """
        Use pyplot animation to create video + data animation
        'ds_label_map' maps the figure labels to keys in the dataset
        """
        if isinstance(ds_or_path, xr.Dataset):
            ds = ds_or_path
        else:
            ds = xr.open_dataset(ds_or_path)

        # There is one more frame than recorded data, so we'll drop the first frame
        frames = sim_frames[1:]
        nframe = len(frames)
        nrow = len(ds_label_map) + 1
        y_labels = list(ds_label_map)

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))
        # fig, axs = self.subplots(nrow, 1)
        sim_ax, data_axes = axs[0], axs[1:]
        sim_ax.axis('off')

        steps = np.arange(0, nframe)
        # Make ticks nice
        if nframe < 20:
            xticks = np.arange(0, nframe + 1)
        elif nframe < 100:
            xticks = np.arange(0, nframe + 1, 5)
        elif nframe <= 200:
            # Round to nearest 10
            r = 10 - nframe % 10
            xticks = np.arange(0, nframe + r + 1, 10, dtype=int)
        else:
            # Do at most 20 of some multiple of 10
            size = nframe // 20
            # We want our size to be the next multiple of 10
            size = size + (-size%10)
            # Remainder to final bin
            r = size - nframe % size
            # Tick to the next bin above nframe
            xticks = np.arange(0, nframe + r + 1, size, dtype=int)

        # Tighten up
        plt.subplots_adjust(hspace=0.1, bottom=0.05, right=0.95, top=0.98)

        # We initialize our plots to get limits etc.
        color = next(self.get_color_wheel())
        lines = []  # Line2D objects
        y_values = []

        for i, label in enumerate(y_labels):
            data_axes[i].set_ylabel(label)

            var = ds_label_map[label]
            # This produces a pivot table with time as index and batch as columns
            pivot = ds[var].to_pandas()

            # Plot each world draw as a separate line; same color though
            data = pivot.T
            world = 0  # We should only have one world
            y_values.append(data[world].to_numpy())
            # Plot the whole line
            sns.lineplot(
                x=steps, y=y_values[-1], color=color, ax=data_axes[i],
                legend=False, label=None,
            )

            # data_axes[i].set_ylim(-1.5, 1.5)
            # data_axes[i].set_yticks(np.linspace(-1, 1, 3))
            data_axes[i].set_autoscaley_on(True)

            # Append the Line2D object
            lines.append(data_axes[i].lines[0])

            # All have the tick lines
            data_axes[i].set_xticks(xticks)
            data_axes[i].set_xlim(-0.95, nframe + 0.95)

            if i < len(y_labels) - 1:
                # Remove x-label
                data_axes[i].set_xticklabels([])
                data_axes[i].set_xlabel('')
            else:
                # Set the final plot's label
                data_axes[i].set_xlabel("Step")

        # Init the frames ax
        sim_img = sim_ax.imshow(frames[0])

        def f_init():
            for ln in lines:
                ln.set_data([], [])
            sim_img.set_data(frames[0])
            return axs

        def f_animate(frame_idx):
            x = steps[:frame_idx + 1]
            for j, ln in enumerate(lines):
                ln.set_data(x, y_values[j][:frame_idx + 1])
                # Re-lim y-axis
                data_axes[j].relim()
                data_axes[j].autoscale_view(scalex=False)

            sim_img.set_data(frames[frame_idx])
            # sim_ax.imshow(frames[i], animated=True)
            return axs

        anim = animation.FuncAnimation(
            fig,
            f_animate,
            frames=steps,
            init_func=f_init,
            interval=50,
            repeat_delay=3000,
            blit=False,  # blitting can't be used with Figure artists
        )

        if save_path:
            print(f"Saving animation {save_path}")
            anim.save(save_path, writer='ffmpeg', fps=fps)

        return anim
