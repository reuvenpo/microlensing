from warnings import catch_warnings, filterwarnings

import numpy.polynomial as npp
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    """A helper class to quickly generate plots with reasonable defaults."""
    def __init__(self, title, x_label="", y_label=""):
        """Asks for the basic details you'd expect in a plot."""
        fig, ax = plt.subplots(layout="constrained")
        pad = 1/4
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

    def plot_heatmap(self, label, x, y, z, z_label="z", num_levels=15, cmap_name="Heat Map"):
        #Levels can be used to draw discreet levels based on chi2 confidence levels
        levels = np.linspace(z.min(), z.max(), num_levels)
        tricontour = self.ax.tricontourf(x, y, z, levels=levels, cmap=cmap_name)

        figure = self.ax.figure
        cbar = figure.colorbar(tricontour)
        cbar.set_label('Z Value (Color Intensity)')

        #plt.scatter(x, y, c='k', marker='.', s=10, label='Original Data Points')

        return tricontour

    def save(self, output_file):
        with catch_warnings():
            filterwarnings("ignore")
            self.ax.legend()
        self.fig.savefig(output_file)
