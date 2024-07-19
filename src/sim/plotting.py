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


class LambdaPlotter(Plotter):
    """
    Plotting util for xArray Datasets
    """

    def __init__(
        self,
        ds_or_path: Union[xr.Dataset, str],
        sns_context: str = "notebook",  # or "paper", "talk", "poster"
    ):
        # if isinstance(ds_or_path, xr.Dataset):
        #     self.ds = ds_or_path
        # else:
        #     self.ds = xr.open_dataset(ds_or_path)

        # Initialize seaborn
        sns.set()
        sns.set_context(sns_context)
        sns.set_palette("colorblind")

    def risk_plot(self, qE_sums, rts, save_path):
        """
        For an OmegaResults dataset
        Generate a 2x2 plot with Biomass, Profit, Risk, and E*
        In each plot, we reduce across the batch axis, and color by omega
        """

        plt.plot(qE_sums, rts)
        plt.ylabel('Risk')
        plt.xlabel('Sum of E')

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

    def policy_plot(self, qEs, risks, save_path=None):
        time = np.arange(qEs.shape[1])

        colors = self.get_color_wheel(n_colors=len(risks))
        for idx, (qE, risk) in enumerate(zip(qEs, risks)):
            if idx % 5 == 0:
                c = next(colors)
                plt.plot(time, qE, label='{:.1f}'.format(risk), color=c)

        plt.ylabel('Et')
        plt.xlabel('time')
        plt.xlim(0., 5.)
        plt.legend()

        # Trim the whitespace around the image
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
