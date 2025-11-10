# This file should hold all our statistics functions
import random
from collections import abc

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from .types import NDFloatArray

Prediction = abc.Callable[[NDFloatArray], NDFloatArray]


def chi_squared(x: NDFloatArray, y: NDFloatArray, sigma: NDFloatArray, f: Prediction) -> float:
    """Compute the chi squared of the data in `y`, `x`, `sigma` given the model `f`

    Example:
        ```
        chi_squared(x, y, sigma, lambda x: a*x + b)
        ```
    """
    prediction = f(x)
    chi_2 = np.sum(((y - prediction)/sigma)^2)
    return chi_2


"""Part A"""
def parabola_fit(datapoints: NDFloatArray):
    """Uses numpy polynomial fit with the least squares for a polynomial of 3rd degree"""
    datapoints = datapoints.transpose()
    coef_array= Polynomial.fit(datapoints[0], datapoints[1],deg=3).convert().coef
    return coef_array

def bootstrapping_parabola(x: NDFloatArray, y: NDFloatArray):
    """Assuming 1000 samples of the coefficients"""
    a_0 = np.array([])
    a_1 = np.array([])
    a_2 = np.array([])
    datapoints = np.concatenate((x,y)).transpose()
    for i in range(1000):
        sample = np.random.Generator.choice(datapoints,
                                            random.randrange(4,datapoints.size))
        coef = parabola_fit(sample)
        a_0 = np.append(a_0, coef[0])
        a_1 = np.append(a_1, coef[1])
        a_2 = np.append(a_2, coef[2])

    """Binning for fit"""
    a_0_hist = np.histogram(a_0,bins = 'auto')
    a_1_hist = np.histogram(a_1,bins = 'auto')
    a_2_hist = np.histogram(a_2,bins = 'auto')
    """Fitting Gaussians for coefficients"""
    a_0_popt = curve_fit(gauss, xdata=a_0_hist[0], ydata=a_0_hist[1])
    a_1_popt = curve_fit(gauss, a_1_hist[0], a_1_hist[1])
    a_2_popt = curve_fit(gauss, a_2_hist[0], a_2_hist[1])
    """By order - the returned array of curve_fit will have x_0 (center of the curve) at index 2,
    and sigma of the curve at index 3"""
    return a_0_popt[2], a_0_popt[3], a_1_popt[2], a_1_popt[3], a_2_popt[2], a_2_popt[3]


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
"""End Part A"""

"""Part B+C"""

def search_chi_sqaure_min(x:NDFloatArray, y:NDFloatArray, sigma:NDFloatArray, search_parameters:NDFloatArray, func: callable, chi_limit, step_size=0.1):
    """Limit search"""
    limits = Limit_Search(chi_limit, func, search_parameters, sigma, step_size, x, y)

    pass


def Limit_Search(chi_limit, func, parameters: ndarray[tuple[Any, ...], dtype[float64]],
                 sigma: ndarray[tuple[Any, ...], dtype[float64]], step_size: float,
                 x: ndarray[tuple[Any, ...], dtype[float64]], y: ndarray[tuple[Any, ...], dtype[float64]]) -> NDFloatArray:
    limits = np.array([])
    for i in parameters.size:
        chi = 0
        param_search = parameters.copy()
        while chi < chi_limit:
            param_search[i] += step_size
            chi = chi_squared(x, y, sigma, func(x, *parameters))
        upperlim = param_search[i]
        chi = 0
        param_search = parameters.copy()
        while chi < chi_limit:
            param_search[i] -= step_size
            chi = chi_squared(x, y, sigma, func(x, *parameters))
        lowerlim = param_search[i]

        limits = np.append(limits, [[lowerlim, upperlim]])
    return limits