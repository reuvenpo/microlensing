from warnings import catch_warnings, filterwarnings

import numpy.polynomial as npp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


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
        levels = [min, min + 2.3, min + 4.61, min + 6.17, min + 9.21, min + 11.8, min + 18.4]
        filled_contours = self.ax.contourf(X, Y, z, levels=levels)
        # custom_norm = colors.Normalize(vmin=min, vmax=min+18.4)
        # self.ax.pcolormesh(z, norm=custom_norm)
        self.fig.colorbar(filled_contours, ax=self.ax, label=z_label)
        # cbar =
        # Set Line contours - we can set to expected errors of chi2 for 2 variables
        line_contours = self.ax.contour(X, Y, z, levels=levels, colors='black', linewidths=0.5)

        self.ax.clabel(line_contours, inline=True, fontsize=8, fmt='%.1f')
        # self.ax.set_aspect('equal')
        # plt.scatter(x, y, c='k', marker='.', s=10, label='Original Data Points')
        if x_min != None and y_min != None and z_min != None:
            self.ax.plot(x[x_min, 0], y[0, y_min], 'x', markersize=10, color='red')
            self.ax.annotate(text=f"z_min:{z_min}", xy=(x[x_min, 0], y[0, y_min]))

        self.ax.plot()

    def save(self, output_file):
        with catch_warnings():
            filterwarnings("ignore")
            # self.ax.legend()
        self.fig.savefig(output_file)
