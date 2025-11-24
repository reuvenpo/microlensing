from warnings import catch_warnings, filterwarnings

import numpy
import numpy.polynomial as npp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from .loc_types import NDFloatArray
from .stat import CHI2_DIFF_CONF_DOF


class Plot:
    """A helper class to quickly generate plots with reasonable defaults."""

    def __init__(self, title, x_label="", y_label=""):
        """Asks for the basic details you'd expect in a plot."""
        fig, ax = plt.subplots(layout="constrained")
        pad = 1 / 4
        fig.get_layout_engine().set(w_pad=pad, h_pad=pad)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        self.ax = ax
        self.fig = fig

    def plot(self, label, x, y, style="o"):
        """A little helper to remind me I can add legend labels."""
        self.ax.plot(x, y, style, label=label)

    def plot_polyfit(self, label, x, y, degree=1, style="--"):
        """Add a polynomial fit to the graph"""
        fit = npp.Polynomial.fit(x, y, deg=degree)
        fit_y = fit(x)
        self.ax.plot(x, fit_y, style, label=label)

        return fit, fit_y

    def plot_func(self, label, x, y, sigma, func, style="--"):
        fit_y = func(x)
        self.ax.errorbar(x=x, y=y, yerr=sigma, marker='o', linestyle='None')
        self.ax.plot(x, fit_y, style, label=label)

    def plot_heatmap(self, x, y, z, z_label="z", num_levels=15, x_min=None, y_min=None, z_min=None):
        # Levels can be used to draw discreet levels based on chi2 confidence levels
        # levels = np.linspace(z.min(), z.max(), num_levels)
        X, Y = np.meshgrid(x, y)
        min = np.min(z)
        levels = CHI2_DIFF_CONF_DOF[2] + min
        filled_contours = self.ax.contourf(X, Y, z, levels=levels)
        # custom_norm = colors.Normalize(vmin=min, vmax=min+CHI2_DIFF_CONF_DOF[2][-1])
        # self.ax.pcolormesh(z, norm=custom_norm)
        self.fig.colorbar(filled_contours, ax=self.ax, label=z_label)
        # cbar =
        # Set Line contours - we can set to expected errors of chi2 for 2 variables
        line_contours = self.ax.contour(X, Y, z, levels=levels, colors='black', linewidths=0.5)

        self.ax.clabel(line_contours, inline=True, fontsize=8, fmt='%.1f')
        # self.ax.set_aspect('equal')
        # plt.scatter(x, y, c='k', marker='.', s=10, label='Original data Points')
        if x_min != None and y_min != None and z_min != None:
            self.ax.plot(x[x_min, 0], y[0, y_min], 'x', markersize=10, color='red')
            self.ax.annotate(text=f"z_min:{z_min}", xy=(x[x_min, 0], y[0, y_min]))

        self.ax.plot()

    def save(self, output_file):
        # with catch_warnings():
        #     filterwarnings("ignore")
        #     self.ax.legend()
        self.fig.savefig(output_file)


def heatmap_corner_plot(
    data: NDFloatArray,
    data_axes: list[NDFloatArray],
    center: NDFloatArray | list[float],
    levels: list[float],
    title: str,
    axis_names: list[str],
):
    """
    :param data: The chunk of data for which a corner plot will be generated
    :param data_axes: A list of arrays of values of parameters used along each axis to calculate `data`
                 Must be organized in the same order as the axes of the `data` block
    :param center: The index of the point in `data` which will be included in all plots
    :return: figure, axes
    """
    dims = len(data.shape)

    axis_combos = []
    mosaic = []
    for row in range(dims - 1):
        mosaic_row = []
        mosaic.append(mosaic_row)
        for col in range(dims - 1):
            if col > row:
                mosaic_row.append(".")
            else:
                combo = (col, row + 1)
                combo_name = str(combo)
                mosaic_row.append(combo_name)
                axis_combos.append((combo_name, combo))
        mosaic_row.append("colorbar")

    fig_size = 8
    fig, axes_dict = plt.subplot_mosaic(
        mosaic,
        width_ratios=[1] * (dims - 1) + [dims / 20],
        layout="compressed",
        # figsize=(fig_size, fig_size),
        dpi=300,
    )

    fig.suptitle(title)

    x_axes = [None] * dims
    y_axes = [None] * dims
    contour_set = None  # Save one to use for the shared colorbar in the end
    for combo_name, combo in axis_combos:
        x, y = combo
        axes = axes_dict[combo_name]
        axes.set_xlabel(axis_names[x])
        axes.set_ylabel(axis_names[y])
        axes.label_outer(remove_inner_ticks=True)

        # Share axes along rows and columns
        shared_x_axis = x_axes[x]
        if shared_x_axis is None:
            x_axes[x] = axes
        else:
            axes.sharex(shared_x_axis)
        shared_y_axis = y_axes[y]
        if shared_y_axis is None:
            y_axes[y] = axes
        else:
            axes.sharey(shared_y_axis)

        contour_axes = [data_axes[axis] for axis in combo]
        data_index = tuple(slice(None) if axis in combo else center[axis] for axis in range(dims))
        contour_data = data[data_index]
        contour_set = axes.contourf(*contour_axes, contour_data.T, levels=levels)
        axes.contour(*contour_axes, contour_data.T, levels=levels, colors="black", linewidths=0.5)

        x0, x1 = axes.get_xlim()
        y0, y1 = axes.get_ylim()
        axes.set_aspect(abs((x1 - x0) / (y1 - y0)), adjustable='box')

    fig.colorbar(contour_set, cax=axes_dict["colorbar"])

    return fig, axes_dict
