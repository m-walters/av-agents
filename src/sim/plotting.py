from typing import List, Union

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
        try:
            mc_steps = ds.coords['mc_step'].values
        except AttributeError:
            # Older data may not have the mc-steps
            mc_steps = np.arange(0, nframe)

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))
        # fig, axs = self.subplots(nrow, 1)
        sim_ax, data_axes = axs[0], axs[1:]
        sim_ax.axis('off')

        steps = ds.coords['step'].values
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
            size = size + (-size % 10)
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
        x_values = []
        mc_axes = []  # Mask for MC graphs

        for i, label in enumerate(y_labels):
            data_axes[i].set_ylabel(label)

            var = ds_label_map[label]
            # This produces a pivot table with time as index and batch as columns
            pivot = ds[var].to_pandas()

            # Plot each world draw as a separate line; same color though
            data = pivot.T
            world = 0  # We should only have one world
            y_values.append(data[world].to_numpy())

            # Some plots use world steps, others mc-steps
            if 'mc_step' in ds[var].dims:
                mc_axes.append(True)
                x_values.append(mc_steps.copy())
            else:
                mc_axes.append(False)
                x_values.append(steps.copy())

            # Plot the whole line
            sns.lineplot(
                x=x_values[-1], y=y_values[-1], color=color, ax=data_axes[i],
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
            i_step = steps[frame_idx]
            world_xs = steps[:frame_idx + 1]
            do_mc = i_step in mc_steps
            if do_mc:
                # Get mc xs
                i_mc = np.argwhere(mc_steps == i_step).flatten()[0]
                mc_xs = mc_steps[:i_mc + 1]

            for j, ln in enumerate(lines):
                ax_is_mc = mc_axes[j]
                if ax_is_mc and do_mc:
                    ln.set_data(mc_xs, y_values[j][:i_mc + 1])
                    # Re-lim y-axis
                    data_axes[j].relim()
                    data_axes[j].autoscale_view(scalex=False)
                elif not ax_is_mc:
                    ln.set_data(world_xs, y_values[j][:frame_idx + 1])
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

    def create_multiagent_animation(
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
        try:
            mc_steps = ds.coords['mc_step'].values
        except AttributeError:
            # Older data may not have the mc-steps
            mc_steps = np.arange(0, nframe)

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))
        # fig, axs = self.subplots(nrow, 1)
        sim_ax, data_axes = axs[0], axs[1:]
        sim_ax.axis('off')

        steps = ds.coords['step'].values
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
            size = size + (-size % 10)
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
        x_values = []
        mc_axes = []  # Mask for MC graphs

        for i, label in enumerate(y_labels):
            data_axes[i].set_ylabel(label)

            var = ds_label_map[label]
            # This produces a pivot table with time as index and batch as columns
            pivot = ds[var].to_pandas()

            # Plot each world draw as a separate line; same color though
            data = pivot.T
            world = 0  # We should only have one world
            y_values.append(data[world].to_numpy())

            # Some plots use world steps, others mc-steps
            if 'mc_step' in ds[var].dims:
                mc_axes.append(True)
                x_values.append(mc_steps.copy())
            else:
                mc_axes.append(False)
                x_values.append(steps.copy())

            # Plot the whole line
            sns.lineplot(
                x=x_values[-1], y=y_values[-1], color=color, ax=data_axes[i],
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
            i_step = steps[frame_idx]
            world_xs = steps[:frame_idx + 1]
            do_mc = i_step in mc_steps
            if do_mc:
                # Get mc xs
                i_mc = np.argwhere(mc_steps == i_step).flatten()[0]
                mc_xs = mc_steps[:i_mc + 1]

            for j, ln in enumerate(lines):
                ax_is_mc = mc_axes[j]
                if ax_is_mc and do_mc:
                    ln.set_data(mc_xs, y_values[j][:i_mc + 1])
                    # Re-lim y-axis
                    data_axes[j].relim()
                    data_axes[j].autoscale_view(scalex=False)
                elif not ax_is_mc:
                    ln.set_data(world_xs, y_values[j][:frame_idx + 1])
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

    def multiagent_plot(
        self,
        save_path: str,
        ds_or_path: Union[xr.Dataset, str],
        ds_label_map: dict,
        max_av: int | None = None,
        truncate: Union[int, range, list] | None = None,
    ):
        """
        PLot data from multiagent dataset

        :param save_path: Save path
        :param ds_or_path: Dataset or path to one
        :param ds_label_map: Map of plot labels to Dataset keys
        :param max_av: Optionally limit the number of egos plotted
        :param truncate: Optionally limit the plotting range by simulation step.
            Provide an int for an end limit or an inclusive window range
        """
        if isinstance(ds_or_path, xr.Dataset):
            ds = ds_or_path
        else:
            ds = xr.open_dataset(ds_or_path)

        nrow = len(ds_label_map)
        y_labels = list(ds_label_map)
        egos = ds.coords['ego'].values
        # Cap at max AV if provided
        n_ego = max_av or len(egos)
        egos = egos[:n_ego]

        steps = ds.coords['step'].values
        mc_steps = ds.coords['mc_step'].values
        if isinstance(truncate, int):
            steps = steps[steps <= truncate]
            mc_steps = mc_steps[mc_steps <= truncate]
        elif isinstance(truncate, (range, list)):
            steps = steps[truncate[0] <= steps <= truncate[1]]
            mc_steps = mc_steps[truncate[0] <= mc_steps <= truncate[1]]

        # Determine these indices for filtering our y-value data for plotting
        _steps = ds.coords['step'].values
        _mc_steps = ds.coords['mc_step'].values
        left_steps_idx = int(np.argwhere(_steps == steps[0])[0][0])
        right_steps_idx = int(np.argwhere(_steps == steps[-1])[0][0])
        left_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[0])[0][0])
        right_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[-1])[0][0])

        duration = steps[-1] - steps[0] + 1
        col_wheel = self.get_color_wheel()
        colors = {ego: next(col_wheel) for ego in egos}

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))

        # Make ticks nice
        if duration < 20:
            xticks = np.arange(steps[0], duration + 1)
        elif duration < 100:
            xticks = np.arange(steps[0], duration + 1, 5)
        elif duration <= 200:
            # Round to nearest 10
            r = 10 - duration % 10
            xticks = np.arange(steps[0], duration + r + 1, 10, dtype=int)
        else:
            # Do at most 20 of some multiple of 10
            size = duration // 20
            # We want our size to be the next multiple of 10
            size = size + (-size % 10)
            # Remainder to final bin
            r = size - duration % size
            # Tick to the next bin above duration
            xticks = np.arange(steps[0], duration + r + 1, size, dtype=int)

        # Tighten up
        plt.subplots_adjust(hspace=0.1, bottom=0.05, right=0.95, top=0.98)

        for i, label in enumerate(y_labels):
            var = ds_label_map[label]
            data = ds[var].sel(world=0)
            y_lines = {}
            # Some plots use world steps, others mc-steps
            if 'mc_step' in ds[var].dims:
                x = mc_steps
                for ego in egos:
                    y = data.sel(ego=ego).values
                    y_lines[ego] = y[left_mc_steps_idx:right_mc_steps_idx + 1]
            else:
                x = steps
                for ego in egos:
                    y = data.sel(ego=ego).values
                    y_lines[ego] = y[left_steps_idx:right_steps_idx + 1]

            # Plot lines for each ego
            for ego in egos:
                sns.lineplot(
                    x=x, y=y_lines[ego], color=colors[ego], ax=axs[i],
                    legend=False, label=f"AV-{ego}",
                )

            axs[i].set_ylabel(label)
            # All have the tick lines
            axs[i].set_xticks(xticks)
            axs[i].set_xlim(steps[0] - 0.95, steps[-1] + 0.95)

            if i < len(y_labels) - 1:
                # Remove x-label
                axs[i].set_xticklabels([])
                axs[i].set_xlabel('')
            else:
                # Set the final plot's label
                axs[i].set_xlabel("Step")

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def multiagent_comparison_plot(
        self,
        save_path: str,
        ds1: Union[xr.Dataset, str],
        ds1_name: str,
        ds2: Union[xr.Dataset, str],
        ds2_name: str,
        metric_label_map: dict,
        axes_layout: List[List[str]] | None = None,
        truncate: Union[int, range, list] | None = None,
    ):
        """
        Plot to compare two datasets
        Datasets should have the same egos and steps
        Plotted values are the averages across egos

        :param save_path: Save path
        :param ds1: Dataset or Dataset path
        :param ds1_name: Name for first dataset
        :param ds2: Dataset or Dataset path
        :param ds2_name: Name for second dataset
        :param metric_label_map: Map of plot labels to Dataset keys
        :param axes_layout: Optional 2D layout of axes labels
        :param truncate: Optionally limit the plotting range by simulation step.
            Provide an int for an end limit or an inclusive window range
        """
        if isinstance(ds1, xr.Dataset):
            ds1 = ds1
        else:
            ds1 = xr.open_dataset(ds1)
        if isinstance(ds2, xr.Dataset):
            ds2 = ds2
        else:
            ds2 = xr.open_dataset(ds2)

        y_labels = list(metric_label_map)
        egos = ds1.coords['ego'].values
        # Check that these datasets have the same egos
        assert np.array_equal(egos, ds2.coords['ego'].values)

        steps = ds1.coords['step'].values
        mc_steps = ds1.coords['mc_step'].values
        # Assert that ds1 and ds2 have the same steps/mc_steps
        assert np.array_equal(steps, ds2.coords['step'].values)
        assert np.array_equal(mc_steps, ds2.coords['mc_step'].values)

        if isinstance(truncate, int):
            steps = steps[steps <= truncate]
            mc_steps = mc_steps[mc_steps <= truncate]
        elif isinstance(truncate, (range, list)):
            steps = steps[truncate[0] <= steps <= truncate[1]]
            mc_steps = mc_steps[truncate[0] <= mc_steps <= truncate[1]]

        # Determine these indices for filtering our y-value data for plotting
        _steps = ds1.coords['step'].values
        _mc_steps = ds1.coords['mc_step'].values
        left_steps_idx = int(np.argwhere(_steps == steps[0])[0][0])
        right_steps_idx = int(np.argwhere(_steps == steps[-1])[0][0])
        left_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[0])[0][0])
        right_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[-1])[0][0])

        duration = steps[-1] - steps[0] + 1
        col_wheel = self.get_color_wheel()
        cols = [next(col_wheel), next(col_wheel)]

        num_ax = len(y_labels)
        if axes_layout is None:
            # Default to 2 columns
            if num_ax % 2 == 0:
                axes_shape = (num_ax // 2, 2)
                axes_layout = [[y_labels[i], y_labels[i + 1]] for i in range(0, num_ax, 2)]
                # Bottom two plots have ticks
            else:
                # Odd number of axes
                axes_shape = ((num_ax + 1) // 2, 2)
                axes_layout = [[y_labels[i], y_labels[i + 1]] for i in range(0, num_ax - 1, 2)]
                axes_layout.append([y_labels[-1], None])
        else:
            # Get the axes_shape from the layout
            axes_shape = (len(axes_layout), 2)

        nrow, ncol = axes_shape
        tick_plots = [None] * ncol  # Those which show the ticks
        # Determine the tick plot labels
        # for i, row in enumerate(axes_layout):
        #     for j, label in enumerate(row):
        #         if label is not None:
        #             tick_plots[j] = label
        tick_plots = ["E[Entropy]", "Risk"]

        fig, axs = self.subplots(nrow, ncol, figsize=(6.4 * ncol, nrow * 1.6))
        # Disable/clear [-1, -1] axis
        axs[-1, -1].axis('off')

        # Create label-to-axis map
        def label2axis(lbl):
            for i, row in enumerate(axes_layout):
                if lbl in row:
                    return axs[i, row.index(lbl)]
            raise ValueError(f"Label {lbl} not found in axes_layout")

        # Make ticks nice
        if duration < 20:
            xticks = np.arange(steps[0], duration + 1)
        elif duration < 100:
            xticks = np.arange(steps[0], duration + 1, 5)
        elif duration <= 200:
            # Round to nearest 10
            r = 10 - duration % 10
            xticks = np.arange(steps[0], duration + r + 1, 10, dtype=int)
        else:
            # Do at most 20 of some multiple of 10
            size = duration // 20
            # We want our size to be the next multiple of 10
            size = size + (-size % 10)
            # Remainder to final bin
            r = size - duration % size
            # Tick to the next bin above duration
            xticks = np.arange(steps[0], duration + r + 1, size, dtype=int)

        # Tighten up
        plt.subplots_adjust(hspace=0.1, bottom=0.05, right=0.95, top=0.98)

        for i, label in enumerate(y_labels):
            var = metric_label_map[label]
            data1 = ds1[var].sel(world=0)
            data2 = ds2[var].sel(world=0)

            y_lines = {}
            # Some plots use world steps, others mc-steps
            if 'mc_step' in ds1[var].dims:
                x = mc_steps
                y_mean1 = data1.mean(dim='ego').values[left_mc_steps_idx:right_mc_steps_idx + 1]
                y_mean2 = data2.mean(dim='ego').values[left_mc_steps_idx:right_mc_steps_idx + 1]
            else:
                x = steps
                y_mean1 = data1.mean(dim='ego').values[left_steps_idx:right_steps_idx + 1]
                y_mean2 = data2.mean(dim='ego').values[left_steps_idx:right_steps_idx + 1]

            ax = label2axis(label)
            sns.lineplot(
                x=x, y=y_mean1, color=cols[0], ax=ax,
                legend=True, label=ds1_name,
            )
            sns.lineplot(
                x=x, y=y_mean2, color=cols[1], ax=ax,
                legend=True, label=ds2_name,
            )

            ax.set_ylabel(label)
            # All have the tick lines
            ax.set_xticks(xticks)
            ax.set_xlim(steps[0] - 0.95, steps[-1] + 0.95)

            if label not in tick_plots:
                # Remove x-label
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                # Set the final plot's label
                ax.set_xlabel("Step")

        if save_path:
            plt.savefig(save_path)

        plt.show()
