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

    # The chi squared difference as a function of confidence level and degrees of freedom.
    # Values are given for 68.3%, 90%, 95.4%, 99%, 99.73%, and 99.99%
    CHI2_DIFF_CONF_DOF = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # filler
        [0.00, 1.00, 2.71, 4.00, 6.63, 9.00, 15.1], # 1 DoF
        [0.00, 2.30, 4.61, 6.17, 9.21, 11.8, 18.4], # 2 DoF
        [0.00, 3.53, 6.25, 8.02, 11.3, 14.2, 21.1], # 3 DoF
        [0.00, 4.72, 7.78, 9.70, 13.3, 16.3, 23.5], # 4 DoF
        [0.00, 5.89, 9.24, 11.3, 15.1, 18.2, 25.7], # 5 DoF
        [0.00, 7.04, 11.6, 12.8, 16.8, 20.1, 27.8], # 6 DoF
    ])

    def plot_heatmap(self, x, y, z, z_label="z", num_levels=15, x_min=None, y_min=None, z_min=None):
        # Levels can be used to draw discreet levels based on chi2 confidence levels
        # levels = np.linspace(z.min(), z.max(), num_levels)
        X, Y = np.meshgrid(x, y)
        min = np.min(z)
        levels = self.CHI2_DIFF_CONF_DOF[2] + min
        filled_contours = self.ax.contourf(X, Y, z, levels=levels)
        # custom_norm = colors.Normalize(vmin=min, vmax=min+self.CHI2_DIFF_CONF_DOF[2][-1])
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
        with catch_warnings():
            filterwarnings("ignore")
            self.ax.legend()
        self.fig.savefig(output_file)
