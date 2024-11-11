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
        self.PLOT_OVERRIDES = {
            "Crashed": self.multiagent_crash_override,
            "Number in Conservative": self.multiagent_behaviors_override,
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

    @staticmethod
    def multiagent_crash_override(
        ds: xr.Dataset, var, steps, mc_steps, world_avg: bool = False, **plot_kwargs,
    ) -> None:
        """
        Specialized function for the 'crashed' multiagent Dataset variable
        """
        if not world_avg:
            y = ds[var].sum(dim='ego').values
            sns.lineplot(
                x=steps, y=y, **plot_kwargs
            )
        else:
            # Shape of 'data' will be [duration]
            data = ds[var].sum(['ego', 'world'], skipna=True)

            # Dataframe with columns ['step', var]
            df = data.to_dataframe().reset_index()
            _ = plot_kwargs.pop("errorbar", None)
            sns.lineplot(
                data=df, x='step', y=var, errorbar='sd', **plot_kwargs
            )
            # bin_range = (0, df[var].max())
            # sns.histplot(
            #     x='step', y=var, bins=30, kde=False,
            #     binrange=bin_range, **plot_kwargs
            # )

    @staticmethod
    def multiagent_behaviors_override(
        ds: xr.Dataset, var, steps, mc_steps, world_avg: bool=False, **plot_kwargs,
    ) -> None:
        """
        Specialized function for the 'behavior_mode' multiagent Dataset variable
        """
        if not world_avg:
            y = ds[var].sum(dim='ego').values
            sns.lineplot(
                x=steps, y=y, **plot_kwargs
            )
        else:
            # Shape of 'data' will be [duration]
            data = ds[var].sum(['ego', 'world'], skipna=True)

            # Dataframe with columns ['step', var]
            df = data.to_dataframe().reset_index()
            _ = plot_kwargs.pop("errorbar", None)
            sns.lineplot(
                data=df, x='step', y=var, errorbar='sd', **plot_kwargs
            )
            # bin_range = (0, df[var].max())
            # sns.histplot(
            #     x='step', y=var, bins=30, kde=False,
            #     binrange=bin_range, **plot_kwargs
            # )

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

    @staticmethod
    def world_stats(
        ds: xr.Dataset,
        var: str,
        as_df: bool = True,
    ) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
        """
        Return the mean and 5/95 percentiles for a dataset on var across worlds

        :param ds: Dataset
        :param var: Variable to extract
        :param as_df: Return as a DataFrame averaged across ego, but containing each world_draw and duration measure
        """
        # First get a mask of the NaNs for this variable
        # mask = ds[var].isnull()
        mask = ds[var] == 0

        # Shape of 'data' will be [world_draws, duration]
        # Set masked 0s to nan and then skip them in the mean calculation
        data = ds[var].where(~mask, np.nan).mean('ego', skipna=True)

        if as_df:
            # Convert to a DataFrame
            # data has world_draws and step dimensions that we need to make columns for
            # Return a dataframe with columns ['world', 'step', var]
            return data.to_dataframe().reset_index()
        else:
            # NaN values may still be present if they were for all egos
            data = data.values
            mean = np.nanmean(data, axis=0)
            p5 = np.nanpercentile(data, 5, axis=0)
            p95 = np.nanpercentile(data, 95, axis=0)

            return mean, p5, p95

    def comparison_plot(
        self,
        save_path: str,
        datasets: list[tuple[Union[xr.Dataset, str], str]],
        metric_label_map: dict,
        axes_layout: list[list[str]] | None = None,
        ylog_plots: list[str] | None = None,
        title: str | None = None,
        truncate: Union[int, range, list] | None = None,
        world_avg: bool = True,
    ):
        """
        Plot to compare multiple datasets.
        Datasets should have the same egos and steps.
        Plotted values are the averages across egos.
        Filters out data after a 'time_to_collision' is encountered

        :param save_path: Save path
        :param datasets: List of (Dataset or Dataset path, Name) tuples
        :param metric_label_map: Map of plot labels to Dataset keys
        :param axes_layout: Optional 2D layout of axes labels
        :param ylog_plots: Optional list of labels to plot on a log scale
        :param title: Optional title for the plot
        :param truncate: Optionally limit the plotting range by simulation step.
            Provide an int for an end limit or an inclusive window range
        :param world_avg: Average across worlds
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

        for label in y_labels:
            var = metric_label_map[label]
            ax = label2axis(label)

            for idx, (ds, ds_name) in enumerate(datasets):
                if world_avg:
                    if label in self.PLOT_OVERRIDES:
                        self.PLOT_OVERRIDES[label](
                            ds, var=var, steps=steps, mc_steps=mc_steps, world_avg=True, color=cols[idx], ax=ax,
                            errorbar='sd', legend=False, label=ds_name
                        )
                    else:
                        # Default to mean
                        df = self.world_stats(ds, var, as_df=True)

                        if 'mc_step' in ds[var].dims:
                            x = mc_steps
                            # Std-dev error bars. See https://seaborn.pydata.org/tutorial/error_bars.html
                            sns.lineplot(
                                data=df, x='mc_step', y=var, color=cols[idx], ax=ax,
                                errorbar='sd', legend=False, label=ds_name,
                            )
                        else:
                            x = steps
                            sns.lineplot(
                                data=df, x='step', y=var, color=cols[idx], ax=ax,
                                errorbar='sd', legend=False, label=ds_name,
                            )

                else:
                    # Call custom methods if they exist
                    if label in self.PLOT_OVERRIDES:
                        self.PLOT_OVERRIDES[label](
                            ds, var=var, steps=steps, mc_steps=mc_steps, world_avg=False,
                            color=cols[idx], ax=ax, legend=False, label=ds_name,
                        )
                    else:
                        data = ds[var].sel(world=0)
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

        # Tighten up
        if not title:
            plt.subplots_adjust(hspace=0.1, bottom=0.15, right=0.95, top=0.95)
        else:
            fig.suptitle(title)
            plt.subplots_adjust(hspace=0.1, bottom=0.15, right=0.95, top=0.9)

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

    def ttc_hist(
        self,
        save_path: str,
        _datasets: list[tuple[xr.Dataset, str]],
    ):
        # Get num world draws from ref
        ref_ds = _datasets[0][0]
        world_draws = ref_ds.coords['world'].values[-1]
        duration = ref_ds.coords['step'].values[-1]

        # Compile datasets
        datasets = [
            ds['time_to_collision'].values for ds, lbl in _datasets
        ]
        # And again
        datasets = [
            ds[~np.isnan(ds)] for ds in datasets
        ]

        col_wheel = self.get_color_wheel()
        # Plotting the distributions
        # Get the bin range as the 'duration' of the simulation
        bin_range = (0, duration)
        # bin_range = (0, 30)

        for i_ds, ds in enumerate(datasets):
            lbl = _datasets[i_ds][1]
            col = next(col_wheel)
            sns.histplot(ds, bins=30, kde=True, label=lbl, color=col, binrange=bin_range)
            print(f"{lbl}: {np.mean(ds)} ({len(ds)}/{world_draws} crashed)")
            # plt.axvline(np.mean(ds), color=col, linestyle='dashed', linewidth=1)

        # Adding titles and labels
        plt.title("")
        plt.xlabel("Step")
        plt.ylabel("")
        plt.legend()

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

    def collisions(
        self,
        save_path: str,
        nominal_ds: xr.Dataset,
        defensive_ds: xr.Dataset,
        hotshot_ds: xr.Dataset | None = None,
    ):
        """
        Investigate the collision/crashed dataset.
        """
        ...


    def ttc_vs_gk(
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
            ttc = ttc[ttc != np.nan].astype(int)
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
        plt.ylim(0, min(200, df['ttc'].max()) + 5)
        plt.ylabel("TTC")
        plt.xlabel("Number of GK-controlled vehicles")

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()

