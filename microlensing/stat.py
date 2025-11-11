# This file should hold all our statistics functions
import random
from collections import abc

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from .loc_types import NDFloatArray
from . import utils
from . import theory

Prediction = abc.Callable[[NDFloatArray], NDFloatArray]
Prediction_single = abc.Callable[[float], float]


def chi_squared(x: NDFloatArray, y: NDFloatArray, sigma: NDFloatArray, f: Prediction) -> float:
    """Compute the chi squared of the data in `y`, `x`, `sigma` given the model `f`

    Example:
        ```
        chi_squared(x, y, sigma, lambda x: a*x + b)
        ```
    """
    prediction = f(x)
    chi_2 = np.sum(((y - prediction) / sigma) ** 2)
    return chi_2


def point_chi_squared(x: float, y: float, sigma: float, f: Prediction_single) -> float:
    prediction = f(x)
    chi_2 = ((y - prediction) / sigma) ** 2
    return chi_2


# Part A
def parabola_fit(datapoints: NDFloatArray):
    """Uses numpy polynomial fit with the least squares for a polynomial of 3rd degree"""
    datapoints = datapoints.transpose()
    coef_array = Polynomial.fit(datapoints[0], datapoints[1], deg=2).convert().coef
    return coef_array


def bootstrapping_parabola(x: NDFloatArray, y: NDFloatArray):
    """Assuming 1000 samples of the coefficients"""
    a_0 = np.zeros([10000])
    a_1 = np.zeros([10000])
    a_2 = np.zeros([10000])
    datapoints: NDFloatArray = np.array([x, y]).transpose()
    num_rows = datapoints.shape[0]
    for i in range(10000):
        sample_size = np.random.randint(low=6, high=num_rows)
        sample_indices = np.random.choice(num_rows, size=sample_size, replace=True)
        sample = datapoints[sample_indices]
        coef = parabola_fit(sample)
        a_0[i] = coef[0]
        a_1[i] = coef[1]
        a_2[i] = coef[2]

    # Switching a_0 to u_0 using inverse function A_0 IS U_0 FROM THIS POINT FORWARD
    # a_0 = theory.extract_u0(a_0)
    # Binning for fit

    a_0_values, a_0_bin_edges = np.histogram(a_0, bins='auto')
    a_0_bin_centers = (a_0_bin_edges[:-1] + a_0_bin_edges[1:]) / 2
    a_1_values, a_1_bin_edges = np.histogram(a_1, bins='auto')
    a_1_bin_centers = (a_1_bin_edges[:-1] + a_1_bin_edges[1:]) / 2
    a_2_values, a_2_bin_edges = np.histogram(a_2, bins='auto')
    a_2_bin_centers = (a_2_bin_edges[:-1] + a_2_bin_edges[1:]) / 2
    # Fitting Gaussians for coefficients
    # p0 = [0, 1, a_0[0], 10]
    a_0_popt = curve_fit(gauss, xdata=a_0_bin_centers, ydata=a_0_values, maxfev=5000)[0]
    a_1_popt = curve_fit(gauss, xdata=a_1_bin_centers, ydata=a_1_values, maxfev=5000)[0]
    a_2_popt = curve_fit(gauss, xdata=a_2_bin_centers, ydata=a_2_values, maxfev=5000)[0]
    # By order - the returned array of curve_fit will have x_0 (center of the curve) at index 2,
    # and sigma of the curve at index 3
    return a_0_popt[2], a_0_popt[3], a_1_popt[2], a_1_popt[3], a_2_popt[2], a_2_popt[3]


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


# End Part A

"""Part B+C"""


def search_chi_sqaure_min(
        x: NDFloatArray,
        y: NDFloatArray,
        sigma: NDFloatArray,
        search_parameters: NDFloatArray,
        func: Prediction_single,
        chi_limit,
        step_size=0.1,
        resolution=100
):
    """Limit search"""
    limits = limit_search(chi_limit, func, search_parameters, sigma, step_size, x, y)
    parameters_axis = utils.split_axis(limits, resolution)
    pass


def limit_search(chi_limit, func, parameters: NDFloatArray,
                 sigma: NDFloatArray, dtypeNDFloatArray, step_size: float,
                 x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
    limits = np.array([])
    for index, value in np.ndenumerate(parameters):
        chi = 0
        param_search = parameters.copy()
        while chi < chi_limit:
            param_search[index] += step_size
            chi = chi_squared(x, y, sigma, func(x, *parameters))
        upperLim = param_search[index]
        chi = 0
        param_search = parameters.copy()
        while chi < chi_limit:
            param_search[index] -= step_size
            chi = chi_squared(x, y, sigma, func(x, *parameters))
        lowerLim = param_search[index]

        limits = np.append(limits, [[lowerLim, upperLim]])
    return limits
