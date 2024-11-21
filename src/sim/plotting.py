from typing import Any, Union, Optional

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

# To my best colorblind-ass visual determination
palette = sns.color_palette("colorblind")
muted_palette = sns.color_palette("muted")
deep_palette = sns.color_palette("deep")
paired_pal = sns.color_palette("Paired")
# brewer = sns.color_palette("Set2")
# online_pal = sns.color_palette("crest", n_colors=5)
# green_pal = sns.light_palette("green", n_colors=5)
# spectral_pal = sns.color_palette("Spectral", n_colors=8)

# Color some common plot entities
AV_COLORS = {
    "blue": palette[0],
    "orange": palette[1],
    "green": palette[2],
    "red": palette[3],
    "pink": palette[4],
    "tan": palette[5],
    #####
    # "nominal": deep_palette[2],  # greensh
    "nominal": palette[3],  # greensh
    "alter": palette[5],  # tansh
    "hotshot": palette[3],  # redsh
    "defensive": deep_palette[0],  # bluesh
    "online-4": (0.66, 0.74, 0.66),  # less greysh
    "online-12": (0.55, 0.66, 0.7),  # greysh
}


class AVPlotter:
    """
    Plotting class for standard plots, multiagent plots, animation plots, etc.
    """

    def __init__(
        self,
        sns_context: str = "notebook",
        font_scale: float = 1.0,
    ):
        # Initialize seaborn
        sns.set()

        # See all plot context with:
        # print(sns.plotting_context())
        rc = {
            "lines.linewidth": 1.4, # Default: 1.2
        }
        sns.set_context(sns_context, font_scale=font_scale, rc=rc)
        sns.set_palette("colorblind")

        # Custom metric processing override methods
        # Keys are the plot labels, permitting Dataset metric keys to not be confined to a single processor
        # We have different sets based on different contexts
        self.PLOT_OVERRIDES = {
            "Crashed": self.multiagent_crash_override,
            "Fraction Defensive": self.multiagent_behaviors_override,
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

        The crashed metric observes all ego vehicles (both online and offline).
        For a given world run, for regular vehicles if they collide they will keep trying to drive.
        However, that makes the following crash information a bit sketchy (they may be crashing on-and-off a fair bit
        afterwards). To manage this, we treat each ego's crash status as just a boolean if they crashed at all
        during the world sim (instead of counting how many times).
        """
        # Get number that were online
        n_online = ds.attrs["n_online"]

        if not world_avg:
            # y = ds[var].sum(dim='ego').values
            # sns.lineplot(
            #     x=steps, y=y, **plot_kwargs
            # )
            raise NotImplementedError("Not implemented for non-world averaging mode")
        else:
            y = np.zeros_like(steps)
            x = steps
            data = ds[var].fillna(0).any(['ego'])
            for i in x:
                # Calculate number of worlds that have had at least one of there egos crash to this timestep
                y[i] = data.sel(step=range(i+1)).any('step').sum(['world']).values

            # No err bars
            _ = plot_kwargs.pop("errorbar", None)
            sns.lineplot(
                y=y, x=x, **plot_kwargs
            )
            ax = plot_kwargs['ax']
            ylim = ax.get_ylim()
            ymax = max(ylim[1], len(ds.coords['world']))
            plot_kwargs['ax'].set_ylim(top=ymax, auto=True)

            # data = ds[var].fillna(0).any(['ego']).sum('world')

            # Dataframe with columns ['step', var]
            # df = data.to_dataframe().reset_index()
            # _ = plot_kwargs.pop("errorbar", None)
            # sns.lineplot(
            #     data=df, x='step', y=var, errorbar=('pi', 90), **plot_kwargs
            # )
            # bin_range = (0, df[var].max())
            # sns.histplot(
            #     x='step', y=var, bins=30, kde=False,
            #     binrange=bin_range, **plot_kwargs
            # )

    @staticmethod
    def multiagent_behaviors_override(
        ds: xr.Dataset, var, steps, mc_steps, world_avg: bool = False, **plot_kwargs,
    ) -> None:
        """
        Specialized function for the 'behavior_mode' multiagent Dataset variable
        """
        # Get number that were online
        n_online = ds.attrs["n_online"]

        if not world_avg:
            raise NotImplementedError("Behavior mode is not implemented for non-world averaged data")
        else:
            # Take the mean across the world and online egos
            data = ds[var].sel(ego=range(n_online)).mean(['ego'], skipna=True)

            # Dataframe with columns ['step', var]
            df = data.to_dataframe().reset_index()
            # _ = plot_kwargs.pop("errorbar", None)
            sns.lineplot(
                data=df, x='step', y=var, **plot_kwargs
            )
            # sns.lineplot(
            #     data=df, x='step', y=var, color=cols[idx], ax=ax,
            #     errorbar=('pi', 90), legend=False, label=ds_name,
            # )
            # bin_range = (0, df[var].max())
            # sns.histplot(
            #     x='step', y=var, bins=30, kde=False,
            #     binrange=bin_range, **plot_kwargs
            # )

    def create_animation(
        self,
        save_path: str,
        datasets: list[xr.Dataset],
        ds_label_map: dict,
        sim_frames: list[np.ndarray],
        fps: int = 30,
        colors: list[Any] | None = None,
    ):
        """
        Use pyplot animation to create video + data animation
        'ds_label_map' maps the figure labels to keys in the dataset
        """
        # First dataset is reference
        ref_ds = datasets[0]

        # There is one more frame than recorded data, so we'll drop the first frame
        frames = sim_frames[1:]
        nframe = len(frames)
        nrow = len(ds_label_map) + 1
        y_labels = list(ds_label_map)
        try:
            mc_steps = ref_ds.coords['mc_step'].values
        except AttributeError:
            # Older data may not have the mc-steps
            mc_steps = np.arange(0, nframe)

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))
        # fig, axs = self.subplots(nrow, 1)
        sim_ax, data_axes = axs[0], axs[1:]
        sim_ax.axis('off')

        steps = ref_ds.coords['step'].values
        # Make ticks nice
        if nframe < 20:
            xticks = np.arange(0, nframe + 1)
        elif nframe < 50:
            xticks = np.arange(0, nframe + 1, 10)
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
        plt.subplots_adjust(hspace=0.1, bottom=0.05, left=0.15, right=0.95, top=0.98)

        # We initialize our plots to get limits etc.
        whl = self.get_color_wheel()
        colors = colors or [next(whl) for _ in datasets]
        line_groups = []  # Line2D objects
        y_values = []
        x_values = []
        mc_axes = []  # Mask for MC graphs

        for i, label in enumerate(y_labels):
            print(f"MW VAR -- {label}")
            data_axes[i].set_ylabel(label)
            var = ds_label_map[label]

            # Some plots use world steps, others mc-steps
            if 'mc_step' in ref_ds[var].dims:
                mc_axes.append(True)
                x_values.append(mc_steps.copy())
            else:
                mc_axes.append(False)
                x_values.append(steps.copy())

            y_values.append([])  # store the y values of this var from each dataset
            for i_ds, ds in enumerate(datasets):
                y_values[-1].append(
                    ds[var].sel(world=0, ego=0).values
                )
                if not any(y_values[-1][-1]):
                    print(f"MW NO YVALS")
                    y_values[-1][-1] = []
                    # Plot empty line
                    sns.lineplot(
                        x=[], y=[], ax=data_axes[i], legend=False, label=None,
                    )
                else:
                    # Plot the line
                    sns.lineplot(
                        x=x_values[-1], y=y_values[-1][-1], color=colors[i_ds], ax=data_axes[i],
                        legend=False, label=None,
                    )

            # data_axes[i].set_yticks(np.linspace(-1, 1, 3))
            # data_axes[i].set_ylim(0, 1)
            data_axes[i].set_autoscaley_on(True)

            # Append the Line2D object
            # lines.append(data_axes[i].lines[0])
            line_groups.append(data_axes[i].lines)

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
            for group in line_groups:
                for ln in group:
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

            for j, ln_grp in enumerate(line_groups):
                ax_is_mc = mc_axes[j]
                for k, ln in enumerate(ln_grp):
                    if ax_is_mc and do_mc:
                        ln.set_data(mc_xs, y_values[j][k][:i_mc + 1])
                        # Re-lim y-axis
                        data_axes[j].relim()
                        data_axes[j].autoscale_view(scalex=False)
                    elif not ax_is_mc:
                        ln.set_data(world_xs, y_values[j][k][:frame_idx + 1])
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

    def double_animation(
        self,
        save_path: str,
        ds_label_map: dict,
        mc_ds: xr.Dataset,
        ds2: xr.Dataset,
        sim_frames1: list[np.ndarray],
        sim_frames2: list[np.ndarray],
        sim_labels: list[str],
        fps: int = 30,
        colors: list[Any] | None = None,
    ):
        """
        Compare a MC-dataset with another (ie baseline)
        """
        # First dataset is reference
        ref_ds = mc_ds

        # There is one more frame than recorded data, so we'll drop the first frame
        frames1 = sim_frames1[1:]
        frames2 = sim_frames2[1:]
        nframe = len(frames1)
        nrow = len(ds_label_map) + 2
        y_labels = list(ds_label_map)
        try:
            mc_steps = ref_ds.coords['mc_step'].values
        except AttributeError:
            # Older data may not have the mc-steps
            mc_steps = np.arange(0, nframe)

        fig, axs = self.subplots(nrow, 1, figsize=(6.4, nrow * 1.6))
        # fig, axs = self.subplots(nrow, 1)
        sim_axes, data_axes = axs[:2], axs[2:]
        [sax.axis('off') for sax in sim_axes]

        steps = ref_ds.coords['step'].values
        # Make ticks nice
        if nframe < 20:
            xticks = np.arange(0, nframe + 1)
        elif nframe < 50:
            xticks = np.arange(0, nframe + 1, 10)
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
        plt.subplots_adjust(hspace=0.1, bottom=0.05, left=0.15, right=0.95, top=0.98)

        # We initialize our plots to get limits etc.
        whl = self.get_color_wheel()
        colors = colors or [next(whl) for _ in range(2)]
        line_groups = []  # Line2D objects
        y_values = []
        x_values = []
        mc_axes = []  # Mask for MC graphs

        for i, label in enumerate(y_labels):
            print(f"MW VAR -- {label}")
            data_axes[i].set_ylabel(label)
            var = ds_label_map[label]

            # Some plots use world steps, others mc-steps
            if 'mc_step' in ref_ds[var].dims:
                mc_axes.append(True)
                x_values.append(mc_steps.copy())
            else:
                mc_axes.append(False)
                x_values.append(steps.copy())

            y_values.append([])  # store the y values of this var from each dataset
            # First grab and plot the MC data
            y_values[-1].append(
                mc_ds[var].sel(world=0, ego=0).values
            )
            sns.lineplot(
                x=x_values[-1], y=y_values[-1][-1], color=colors[0], ax=data_axes[i],
                legend=False, label=None,
            )
            # Check if partner dataset has values for this var
            ds2_vals = ds2[var].sel(world=0, ego=0).values
            if np.isnan(ds2_vals).all():
                # Don't add this line
                ...
            else:
                y_values[-1].append(ds2_vals)
                sns.lineplot(
                    x=x_values[-1], y=ds2_vals, color=colors[1], ax=data_axes[i],
                    legend=False, label=None,
                )


            # data_axes[i].set_yticks(np.linspace(-1, 1, 3))
            # data_axes[i].set_ylim(0, 1)
            data_axes[i].set_autoscaley_on(True)

            # Append the Line2D object
            # lines.append(data_axes[i].lines[0])
            line_groups.append(data_axes[i].lines)

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
        sim_img1 = sim_axes[0].imshow(frames1[0])
        sim_img2 = sim_axes[1].imshow(frames2[0])

        def f_init():
            for group in line_groups:
                for ln in group:
                    ln.set_data([], [])
            sim_img1.set_data(frames1[0])
            sim_img2.set_data(frames2[0])
            return axs

        def f_animate(frame_idx):
            i_step = steps[frame_idx]
            world_xs = steps[:frame_idx + 1]
            do_mc = i_step in mc_steps
            if do_mc:
                # Get mc xs
                i_mc = np.argwhere(mc_steps == i_step).flatten()[0]
                mc_xs = mc_steps[:i_mc + 1]

            for j, ln_grp in enumerate(line_groups):
                ax_is_mc = mc_axes[j]
                for k, ln in enumerate(ln_grp):
                    if ax_is_mc and do_mc:
                        ln.set_data(mc_xs, y_values[j][k][:i_mc + 1])
                        # Re-lim y-axis
                        data_axes[j].relim()
                        data_axes[j].autoscale_view(scalex=False)
                    elif not ax_is_mc:
                        ln.set_data(world_xs, y_values[j][k][:frame_idx + 1])
                        # Re-lim y-axis
                        data_axes[j].relim()
                        data_axes[j].autoscale_view(scalex=False)

            sim_img1.set_data(frames1[frame_idx])
            sim_img2.set_data(frames2[frame_idx])
            # sim_ax.imshow(frames1[i], animated=True)
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
        styles: Optional[list[Any]] = None,
        colors: Optional[list[Any]] = None,
        skip_error: Optional[list[bool]] = None,
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
        :param styles: List of seaborn line styles
        :param skip_error: Optional list of booleans indicating error plotting
        :param axes_layout: Optional 2D layout of axes labels
        :param ylog_plots: Optional list of labels to plot on a log scale
        :param title: Optional title for the plot
        :param truncate: Optionally limit the plotting range by simulation step.
            Provide an int for an end limit or an inclusive window range
        :param world_avg: Average across worlds
        """
        # Default styles
        styles = styles or ["-"] * len(datasets)

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
            # assert np.array_equal(steps, ds.coords['step'].values), \
            #     f"Steps do not match for dataset {name}"
            # assert np.array_equal(mc_steps, ds.coords['mc_step'].values), \
            #     f"MC steps do not match for dataset {name}"
            ...

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
        if not colors:
            col_wheel = self.get_color_wheel()
            cols = [next(col_wheel) for _ in datasets]
        else:
            cols = colors

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

        fig, axs = self.subplots(nrow, ncol, figsize=(6.4 * ncol, nrow * 1.8))

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
        elif duration < 50:
            xticks = np.arange(steps[0], duration + 1, 10)
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
                err = ('pi', 66)
                if skip_error and skip_error[idx]:
                    # Indication to skip
                    err = None

                if world_avg:
                    if label in self.PLOT_OVERRIDES:
                        self.PLOT_OVERRIDES[label](
                            ds, var=var, steps=steps, mc_steps=mc_steps, world_avg=True, color=cols[idx], ax=ax,
                            errorbar=err, legend=False, label=ds_name, ls=styles[idx]
                        )
                    else:
                        # Default to mean
                        df = self.world_stats(ds, var, as_df=True)

                        if 'mc_step' in ds[var].dims:
                            # Std-dev error bars. See https://seaborn.pydata.org/tutorial/error_bars.html
                            sns.lineplot(
                                data=df, x='mc_step', y=var, color=cols[idx], ax=ax,
                                errorbar=err, legend=False, label=ds_name, ls=styles[idx],
                            )
                        else:
                            sns.lineplot(
                                data=df, x='step', y=var, color=cols[idx], ax=ax,
                                errorbar=err, legend=False, label=ds_name, ls=styles[idx],
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

        # For baselines
        # axs[0, 1].legend(
        #     title=None
        # )
        # For compare-plot.pdf
        axs[0, 0].legend(
            bbox_to_anchor=(1.15, 0.25),
            ncol=4,
            title=None
        )

        # Tighten up
        if not title:
            plt.subplots_adjust(hspace=0.1, bottom=0.15, left=0.06, right=0.97, top=0.95, wspace=0.16)
        else:
            fig.suptitle(title)
            plt.subplots_adjust(hspace=0.1, bottom=0.15, right=0.95, top=0.9)

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    def ttc_hist(
        self,
        save_path: str,
        _datasets: list[tuple[xr.Dataset, str]],
        colors: Optional[list[Any]] = None,
        bin_range: tuple[int, int] | None = None,
        kde: bool = True,
    ):
        # Get num world draws from ref
        ref_ds = _datasets[0][0]
        world_draws = ref_ds.coords['world'].values[-1] + 1  # zero-based
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
        colors = colors or [next(col_wheel) for _ in datasets]
        # Plotting the distributions
        # Get the bin range as the 'duration' of the simulation
        if not bin_range:
            bin_range = (0, duration)
            # bin_range = (0, 30)

        nbins = min(bin_range[1] - bin_range[0], 30)

        for i_ds, ds in enumerate(datasets):
            lbl = _datasets[i_ds][1]
            col = colors[i_ds]
            sns.histplot(ds, bins=nbins, kde=kde, label=lbl, color=col, binrange=bin_range)
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

    def ttc_vs_online(
        self,
        save_path: str,
        datasets: list[Union[xr.Dataset, str]],
    ):
        """
        Time-To-Collision as function of number of online GK vehicles

        :param save_path: Save path
        :param datasets: List of (Dataset or Dataset path, Label) tuples
        """
        # Compile datasets
        datasets: list[xr.Dataset] = [xr.open_dataset(ds) if isinstance(ds, str) else ds for ds in datasets]

        df = pd.DataFrame([], columns=['n_online', 'ttc'], dtype=int)
        for i_ds, ds in enumerate(datasets):
            n_online = ds.attrs["n_online"]
            ttc = ds["time_to_collision"]
            # Filter out the inf values
            ttc = ttc[~ttc.isnull()].astype(int)
            print(f"MW MAX -- {ttc.max()}")
            if len(ttc) == 0:
                print(f"No TTC values for dataset: {i_ds}")

            # Add to dataframe
            df = pd.concat([df, pd.DataFrame({'n_online': n_online, 'ttc': ttc})])

        # Set placeholder categories to make the x-axis numerical
        # all_x_values = np.arange(0, df['n_online'].max() + 1)
        # df['n_online'] = pd.Categorical(df['n_online'], categories=all_x_values)

        print(df[:100])

        col_wheel = self.get_color_wheel()
        fig = sns.violinplot(
            data=df, x='n_online', y='ttc', color=next(col_wheel),
            cut=0, inner='point',
            # If inner is 'box'
            # inner_kws=dict(box_width=15, whis_width=2, color=".8")
        )

        # Set ylims from 0 to 100 or the max TTCe
        # plt.ylim(0, min(200, df['ttc'].max()) + 5)
        plt.ylabel("TTC")
        plt.xlabel("Number of GK-online vehicles")

        if save_path:
            print(f"Saving to {save_path}")
            plt.savefig(save_path)

        plt.show()
