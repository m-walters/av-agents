from typing import Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

"""
Note https://docs.xarray.dev/en/stable/user-guide/plotting.html
for plotting with xarray objects
"""


class AVPlotter:
    """
    Plotting class for standard plots, multiagent plots, animation plots, etc.
    """

    def __init__(
        self,
        sns_context: str = "paper"
    ):
        # Initialize seaborn
        sns.set()
        sns.set_context(sns_context)
        sns.set_palette("colorblind")

        # Custom metric processing override methods
        # Keys are the plot labels, permitting Dataset metric keys to not be confined to a single processor
        # We have different sets based on different contexts
        self.GK_METRIC_PROCESSORS = {
            "Crashed": self.process_multiagent_crashed,
            "Number in Conservative": self.process_multiagent_behaviors,
        }

    @staticmethod
    def get_color_wheel():
        """
        Return a color generator for the current seaborn palette
        """
        return iter(sns.color_palette())

    @staticmethod
    def subplots(nrow, ncol, **kwargs):
        return plt.subplots(nrow, ncol, **kwargs)

    @classmethod
    def process_multiagent_crashed(cls, var_dataset: xr.Dataset) -> np.ndarray:
        """
        Specialized function for the 'crashed' multiagent Dataset variable

        :param var_dataset: The filtered dataset on the 'crashed' variable
        """
        # return var_dataset.sum(dim='ego').values
        return var_dataset.sum(dim='ego').values  # Pycharm flags as wrong return type but it's correct

    @classmethod
    def process_multiagent_behaviors(cls, var_dataset: xr.Dataset) -> np.ndarray:
        """
        Specialized function for the 'behavior_mode' multiagent Dataset variable

        :param var_dataset: The filtered dataset on the 'behavior_mode' variable
        """
        return var_dataset.sum(dim='ego').values  # Pycharm flags as wrong return type but it's correct

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
        datasets: list[tuple[Union[xr.Dataset, str], str]],
        metric_label_map: dict,
        axes_layout: list[list[str]] | None = None,
        ylog_plots: list[str] | None = None,
        truncate: Union[int, range, list] | None = None,
    ):
        """
        Plot to compare multiple datasets.
        Datasets should have the same egos and steps.
        Plotted values are the averages across egos.

        :param save_path: Save path
        :param datasets: List of (Dataset or Dataset path, Name) tuples
        :param metric_label_map: Map of plot labels to Dataset keys
        :param axes_layout: Optional 2D layout of axes labels
        :param ylog_plots: Optional list of labels to plot on a log scale
        :param truncate: Optionally limit the plotting range by simulation step.
            Provide an int for an end limit or an inclusive window range
        """
        # Compile datasets
        datasets = [(xr.open_dataset(ds) if isinstance(ds, str) else ds, name) for ds, name in datasets]

        # Extract the first dataset to use as a reference
        ref_ds = datasets[0][0]

        y_labels = list(metric_label_map)
        egos = ref_ds.coords['ego'].values
        steps = ref_ds.coords['step'].values
        mc_steps = ref_ds.coords['mc_step'].values

        # Check that all datasets have the same egos, steps, and mc_steps
        for ds, name in datasets:
            assert np.array_equal(egos, ds.coords['ego'].values), \
                f"Egos do not match for dataset {name}"
            assert np.array_equal(steps, ds.coords['step'].values), \
                f"Steps do not match for dataset {name}"
            assert np.array_equal(mc_steps, ds.coords['mc_step'].values), \
                f"MC steps do not match for dataset {name}"

        # Handle truncation
        if isinstance(truncate, int):
            steps = steps[steps <= truncate]
            mc_steps = mc_steps[mc_steps <= truncate]
        elif isinstance(truncate, (range, list)):
            steps = steps[truncate[0] <= steps <= truncate[1]]
            mc_steps = mc_steps[truncate[0] <= mc_steps <= truncate[1]]

        # Determine these indices for filtering our y-value data for plotting
        _steps = ref_ds.coords['step'].values
        _mc_steps = ref_ds.coords['mc_step'].values
        left_steps_idx = int(np.argwhere(_steps == steps[0])[0][0])
        right_steps_idx = int(np.argwhere(_steps == steps[-1])[0][0])
        left_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[0])[0][0])
        right_mc_steps_idx = int(np.argwhere(_mc_steps == mc_steps[-1])[0][0])

        duration = steps[-1] - steps[0] + 1
        col_wheel = self.get_color_wheel()
        cols = [next(col_wheel) for _ in datasets]

        num_ax = len(y_labels)
        if axes_layout is None:
            # Default to 2 columns
            if num_ax % 2 == 0:
                axes_shape = (num_ax // 2, 2)
                axes_layout = [[y_labels[i], y_labels[i + 1]] for i in range(0, num_ax, 2)]
            else:
                axes_shape = ((num_ax + 1) // 2, 2)
                axes_layout = [[y_labels[i], y_labels[i + 1]] for i in range(0, num_ax - 1, 2)]
                axes_layout.append([y_labels[-1], None])
        else:
            axes_shape = (len(axes_layout), 2)

        nrow, ncol = axes_shape
        tick_plots = [None] * ncol  # Those which show the ticks
        # The bottom plots of each column have the ticks
        # Work backwards from the bottom
        i = 0
        while not all(tick_plots):
            i -= 1
            for c, lbl in enumerate(axes_layout[i]):
                if tick_plots[c] is None and lbl:
                    tick_plots[c] = lbl

        fig, axs = self.subplots(nrow, ncol, figsize=(6.4 * ncol, nrow * 1.6))

        # Disable/clear any 'None' axes
        for i, row in enumerate(axes_layout):
            for j, lbl in enumerate(row):
                if not lbl:
                    axs[i, j].axis('off')

        # Create label-to-axis map
        def label2axis(_lbl):
            for i, _row in enumerate(axes_layout):
                if _lbl in _row:
                    return axs[i, _row.index(_lbl)]
            raise ValueError(f"Label {_lbl} not found in axes_layout")

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

        for label in y_labels:
            var = metric_label_map[label]
            ax = label2axis(label)

            for idx, (ds, ds_name) in enumerate(datasets):
                data = ds[var].sel(world=0)

                # Call custom methods if they exist
                if label in self.GK_METRIC_PROCESSORS:
                    y = self.GK_METRIC_PROCESSORS[label](data)
                else:
                    # Default to mean
                    y = data.mean(dim='ego').values

                if 'mc_step' in ds[var].dims:
                    x = mc_steps
                    y = y[left_mc_steps_idx:right_mc_steps_idx + 1]
                else:
                    x = steps
                    y = y[left_steps_idx:right_steps_idx + 1]

                sns.lineplot(
                    x=x, y=y, color=cols[idx], ax=ax,
                    legend=False, label=ds_name,
                )

            ax.set_ylabel(label)
            ax.set_xticks(xticks)
            ax.set_xlim(steps[0] - 0.95, steps[-1] + 0.95)
            if ylog_plots:
                if label in ylog_plots:
                    ax.set_yscale('log')

            if label not in tick_plots:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                ax.set_xlabel("Step")

        # Add legend to the top row
        for ax in axs[0]:
            ax.legend(
                # bbox_to_anchor=(1.05, 1),
                # bbox_transform=axs[1, 1].transAxes,
                # ncol=len(self.ds.omega),
                # loc='center left',
                title=None
            )

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

    def ttc_baselines_hist(
        self,
        save_path: str,
        nominal_ds: xr.Dataset,
        conservative_ds: xr.Dataset,
    ):
        """
        Histogram of TTCs for the two baselines
        """
        nom_values: np.ndarray = nominal_ds['time_to_collision'].values
        cons_values: np.ndarray = conservative_ds['time_to_collision'].values
        # When no collision occurred, values are `inf`
        # Filter these out
        nom_values = nom_values[nom_values != np.inf]
        cons_values = cons_values[cons_values != np.inf]

        col_wheel = self.get_color_wheel()
        nom_color = next(col_wheel)
        cons_color = next(col_wheel)

        # Plotting the distributions
        sns.histplot(nom_values, bins=30, kde=True, label='Nominal', color=nom_color)
        sns.histplot(cons_values, bins=30, kde=True, label='Cons.', color=cons_color)

        # Adding vertical lines at the means
        nom_mean = np.mean(nom_values)
        cons_mean = np.mean(cons_values)
        plt.axvline(nom_mean, color=nom_color, linestyle='dashed', linewidth=1)
        plt.axvline(cons_mean, color=cons_color, linestyle='dashed', linewidth=1)

        # Adding titles and labels
        plt.title("")
        plt.xlabel("Step")
        plt.ylabel("")
        plt.legend()

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

    def ttc_vs_num_gk(
        self,
        save_path: str,
        datasets: list[Union[xr.Dataset, str]],
    ):
        """
        Time-To-Collision as function of number of GK-controlled vehicles.

        :param save_path: Save path
        :param datasets: List of (Dataset or Dataset path, Label) tuples
        """
        # Compile datasets
        datasets = [xr.open_dataset(ds) if isinstance(ds, str) else ds for ds in datasets]

        df = pd.DataFrame([], columns=['num_gk', 'ttc'], dtype=int)
        for i_ds, ds in enumerate(datasets):
            n_controlled = ds.ego.size
            ttc = ds.time_to_collision.values
            # Filter out the inf values
            ttc = ttc[ttc != np.inf].astype(int)
            if len(ttc) == 0:
                print(f"No TTC values for dataset: {i_ds}")


            # Add to dataframe
            df = pd.concat([df, pd.DataFrame({'num_gk': n_controlled, 'ttc': ttc})])

        # Set placeholder categories to make the x-axis numerical
        all_x_values = np.arange(0, df['num_gk'].max() + 1)
        df['num_gk'] = pd.Categorical(df['num_gk'], categories=all_x_values)

        col_wheel = self.get_color_wheel()
        fig = sns.violinplot(
            data=df, x='num_gk', y='ttc', color=next(col_wheel),
            cut=0, inner='point',
            # If inner is 'box'
            # inner_kws=dict(box_width=15, whis_width=2, color=".8")
        )

        # Set ylims from 0 to 100 or the max TTC
        plt.ylim(0, min(100, df['ttc'].max()) + 5)
        plt.ylabel("TTC")
        plt.xlabel("Number of GK-controlled vehicles")

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

