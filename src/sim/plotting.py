from typing import Optional, Union

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

    def omega_quad_plot(self, fig=None, axs=None, save_path=None, plot_kwargs: Optional[dict] = None):
        """
        For an OmegaResults dataset
        Generate a 2x2 plot with Biomass, Profit, Risk, and E*
        In each plot, we reduce across the batch axis, and color by omega
        """
        plot_kwargs = plot_kwargs or {}

        if axs is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 6))
            fig, axs = self.subplots(2, 2, sharex=True, **plot_kwargs)
            plt.subplots_adjust(wspace=0.4)

        colors = self.get_color_wheel()
        for omega in self.ds.omega:
            c = next(colors)
            for i, var in enumerate(['B', 'V', 'E', 'Rt']):
                # This produces a pivot table with time as index and batch as columns
                pivot = self.ds[var].sel(omega=omega).to_pandas()
                # Melt it to go from wide to long form, with batch as a variable, and our var as value
                melted = pivot.melt(var_name='batch', value_name=var, ignore_index=False)
                label = f"w={np.round(omega.values, 2)}"
                sns.lineplot(
                    x="time", y=var, data=melted, color=c, label=label, ax=axs[i // 2, i % 2],
                    legend=False
                )

        # Title
        # fig.suptitle('')
        # Set Biomass y-scale to be log and min at 0
        # axs[0, 0].set_yscale('log')
        axs[0, 0].set_ylim(bottom=0)

        axs[0, 0].set_ylabel('Biomass')
        axs[0, 1].set_ylabel('Profit')
        axs[1, 0].set_ylabel('E')
        axs[1, 1].set_ylabel('Risk')

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel('Horizon')
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        axs[0, 1].legend(
            # bbox_to_anchor=(0.7925, 2.51),
            # bbox_transform=axs[1, 1].transAxes,
            # ncol=len(self.ds.omega),
            # loc='center left',
            # title='Omega'
        )

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, axs

    def projection_plot(self, fig=None, axs=None, save_path=None, plot_kwargs: Optional[dict] = None):
        """
        For a ProjectionResults dataset
        Generate two plots that show risk projections for different qE values, and biomass evolution
        In each plot, we reduce across the batch axis, and color by E
        """
        plot_kwargs = plot_kwargs or {}

        if axs is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 6))
            fig, axs = self.subplots(2, 2, sharex=True, **plot_kwargs)
            plt.subplots_adjust(wspace=0.4)

        colors = self.get_color_wheel()
        for E in self.ds.E:
            c = next(colors)
            for i, var in enumerate(['B', 'V', 'Rt']):
                # This produces a pivot table with time as index and batch as columns
                pivot = self.ds[var].sel(E=E).to_pandas()
                # Melt it to go from wide to long form, with batch as a variable, and our var as value
                melted = pivot.melt(var_name='batch', value_name=var, ignore_index=False)
                label = f"E={np.round(E.values, 2)}"
                sns.lineplot(
                    # x="time", y=var, data=melted, color=c, label=label, ax=axs[i],
                    x="time", y=var, data=melted, color=c, label=label, ax=axs[i // 2, i % 2],
                    legend=False
                )

        # # Title
        # # fig.suptitle('')
        # # Set Biomass y-scale to be log and min at 0
        # # axs[0, 0].set_yscale('log')
        # axs[0].set_ylim(bottom=0)
        #
        # axs[0].set_ylabel('Biomass')
        # axs[1].set_ylabel('Risk')
        # axs[1].set_xlabel('Horizon')
        # axs[1].xaxis.get_major_locator().set_params(integer=True)
        #
        # axs[0].legend(
        #     # bbox_to_anchor=(0.7925, 2.51),
        #     # bbox_transform=axs[1, 1].transAxes,
        #     # ncol=len(self.ds.omega),
        #     # loc='center left',
        #     # title='E'
        # )

        axs[0, 0].set_ylim(bottom=0)

        axs[0, 0].set_ylabel('Biomass')
        axs[0, 1].set_ylabel('Profit')
        axs[1, 0].set_ylabel('Risk')

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel('Horizon')
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        axs[0, 1].legend(
            # bbox_to_anchor=(0.7925, 2.51),
            # bbox_transform=axs[1, 1].transAxes,
            # ncol=len(self.ds.omega),
            # loc='center left',
            # title='Omega'
        )

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, axs

    def pref_evolve_plot(
        self, fig=None, axs=None, save_path=None, plot_kwargs: dict | None = None,
    ):
        """
        For an EvolvePreferenceResults dataset
        Generate a 2x2 plot with Biomass, Profit, Risk, and E
        In each plot, we reduce across the batch axis
        """
        plot_kwargs = plot_kwargs or {}

        if axs is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 6))
            fig, axs = self.subplots(2, 2, sharex=True, **plot_kwargs)
            plt.subplots_adjust(wspace=0.4)

        ax_map = {
            'B': axs[0, 0],
            'V': axs[0, 1],
            'Lt': axs[0, 1].twinx(),  # Share with Profit V
            'Rt': axs[1, 0],
            'E': axs[1, 1],
        }

        colors = self.get_color_wheel()
        c = next(colors)
        lt_c = next(colors)
        for i, var in enumerate(['B', 'V', 'Lt', 'Rt', 'E']):
            # This produces a pivot table with time as index and batch as columns
            pivot = self.ds[var].to_pandas()
            # Melt it to go from wide to long form, with batch as a variable, and our var as value
            melted = pivot.melt(var_name='batch', value_name=var, ignore_index=False)
            # label = f"w={np.round(omega.values, 2)}"
            if var == 'Lt':
                ls = ':'  # Dotted line for Loss
                color = lt_c
            else:
                ls = '-'
                color = c

            sns.lineplot(
                x="time", y=var, data=melted, color=color, ls=ls, ax=ax_map[var], legend=False,
                label=None,
            )

        # Title
        # fig.suptitle('')
        ax_map['B'].set_ylabel('Biomass')
        ax_map['B'].set_ylim(bottom=0)

        ax_map['V'].set_ylabel('Profit')
        ax_map['Lt'].set_ylabel('Loss', color=lt_c)
        ax_map['Lt'].grid(False)
        ax_map['Lt'].tick_params(axis='y', labelcolor=lt_c)

        ax_map['Rt'].set_ylabel('Risk')
        ax_map['E'].set_ylabel('E')

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel('Horizon')
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, axs

    def pref_evolve_plot_2(
        self, param_name, param_label=None, fig=None, axs=None, save_path=None, plot_kwargs: dict | None = None,
    ):
        """
        Similar to pref_evolve_plot, but with E excluded
        Instead we plot the param evolution
        """
        param_label = param_label or param_name
        plot_kwargs = plot_kwargs or {}

        if axs is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 6))
            fig, axs = self.subplots(2, 2, sharex=True, **plot_kwargs)
            plt.subplots_adjust(wspace=0.4)

        ax_map = {
            'B': axs[0, 0],
            'E': axs[1, 0],
            'V': axs[0, 1],
            'Lt': axs[0, 1].twinx(),  # Share with Profit V
            'Rt': axs[1, 1],
            param_name: axs[1, 1].twinx(),
        }

        colors = self.get_color_wheel()
        c = next(colors)
        c2 = next(colors)
        ls = '-'
        ls2 = ':'
        for i, var in enumerate(['B', 'V', 'Lt', 'Rt', 'E']):
            # This produces a pivot table with time as index and batch as columns
            pivot = self.ds[var].to_pandas()
            # Melt it to go from wide to long form, with batch as a variable, and our var as value
            melted = pivot.melt(var_name='batch', value_name=var, ignore_index=False)
            ls_ = ls2 if var == 'Lt' else ls
            c_ = c2 if var == 'Lt' else c

            sns.lineplot(
                x="time", y=var, data=melted, color=c_, ls=ls_, ax=ax_map[var], legend=False,
                label=None,
            )

        # Plot param evolution
        param_y = self.ds[param_name].to_numpy()
        sns.lineplot(param_y, color=c2, ls=ls2, ax=ax_map[param_name], legend=False, label=param_label)

        # Title
        # fig.suptitle('')
        ax_map['B'].set_ylabel('Biomass')
        ax_map['B'].set_ylim(bottom=0)
        ax_map['E'].set_ylabel('E')

        ax_map['V'].set_ylabel('Profit')
        ax_map['Lt'].set_ylabel('Loss', color=c2)
        ax_map['Lt'].grid(False)
        ax_map['Lt'].tick_params(axis='y', labelcolor=c2)

        ax_map['Rt'].set_ylabel('Risk')
        # ax_map['Rt'].ticklabel_format(style='scientific')

        ax_map[param_name].set_ylabel(param_label, color=c2)
        ax_map[param_name].grid(False)
        ax_map[param_name].tick_params(axis='y', labelcolor=c2)

        for i in range(2):
            lower_ax = axs[1, i]
            lower_ax.set_xlabel('Horizon')
            lower_ax.xaxis.get_major_locator().set_params(integer=True)

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, axs


class AVPlotter(Plotter):
    """
    Plotter for run results
    """

    def __init__(
        self,
        ds_or_path: Union[xr.Dataset, str],
        sns_context: str = "notebook",
    ):
        super().__init__(ds_or_path, sns_context)

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


class RoadVisualizer(Plotter):
    """
    Plotting utils for visualizing some event data
    """

    def __init__(
        self,
        ds_or_path: Union[xr.Dataset, str],
        sns_context: str = "notebook",
    ):
        super().__init__(ds_or_path, sns_context)

    def trajectory_plot(
        self, fig=None, ax=None, save_path="results/plots/trajectory_plot.png", plot_kwargs: Optional[dict] = None
    ):
        """
        Plot the trajectory of vehicles in the highway environment.
        """
        plot_kwargs = plot_kwargs or {}

        if ax is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (10, 6))  # Random Size
            fig, ax = plt.subplots(**plot_kwargs)

        vehicles = self.ds.vehicles
        for vehicle in vehicles:
            x = self.ds.x.sel(vehicle=vehicle)
            y = self.ds.y.sel(vehicle=vehicle)
            ax.plot(x, y, label=f'Vehicle {vehicle.item()}')

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Vehicle Trajectories')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)  # results/plot

        return fig, ax

    def speed_plot(
        self, fig=None, ax=None, save_path="results/plots/speed_plot.png", plot_kwargs: Optional[dict] = None
    ):
        """
        Plot the speed of vehicles over time a specific duration.
        """
        plot_kwargs = plot_kwargs or {}

        if ax is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (10, 6))  # Random Size
            fig, ax = plt.subplots(**plot_kwargs)

        vehicles = self.ds.vehicles
        for vehicle in vehicles:
            speed = self.ds.speed.sel(vehicle=vehicle)
            time = self.ds.time
            ax.plot(time, speed, label=f'Vehicle {vehicle.item()}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Speed')
        ax.set_title('Vehicle Speeds Over Time')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, ax

    def safety_distance_plot(
        self, fig=None, ax=None, save_path="results/plots/safety_distance_plot.png", plot_kwargs: Optional[dict] = None
    ):
        """
        Plot the safety distance between vehicles over time
        """
        plot_kwargs = plot_kwargs or {}

        if ax is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (10, 6))  # Random Size
            fig, ax = plt.subplots(**plot_kwargs)

        vehicles = self.ds.vehicles
        time = self.ds.time

        for i in range(len(vehicles) - 1):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[i + 1]
            distance = self.ds.distance_between.sel(vehicle1=vehicle1, vehicle2=vehicle2)
            ax.plot(time, distance, label=f'Between Vehicle {vehicle1.item()} and {vehicle2.item()}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Safety Distance')
        ax.set_title('Safety Distance Between Vehicles Over Time')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, ax

    def acceleration_heatmap(
        self, fig=None, ax=None, save_path="results/plots/acceleration_heatmap.png", plot_kwargs: Optional[dict] = None
    ):
        """
        Create a heatmap of vehicle accelerations over time
        """
        plot_kwargs = plot_kwargs or {}

        if ax is None:
            plot_kwargs['figsize'] = plot_kwargs.get('figsize', (12, 8))
            fig, ax = plt.subplots(**plot_kwargs)

        acceleration_data = self.ds.acceleration.values
        vehicles = self.ds.vehicles.values
        time = self.ds.time.values

        sns.heatmap(acceleration_data, ax=ax, cmap='coolwarm', center=0)

        ax.set_xlabel('Time')
        ax.set_ylabel('Vehicle')
        ax.set_title('Vehicle Acceleration Heatmap')
        ax.set_yticks(range(len(vehicles)))
        ax.set_yticklabels(vehicles)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, ax
