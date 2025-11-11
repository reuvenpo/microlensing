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
def parabola_fit(x: NDFloatArray, y: NDFloatArray):
    """Uses numpy polynomial fit with the least squares for a polynomial of 3rd degree"""
    coef_array = Polynomial.fit(x, y, deg=2).convert().coef
    return coef_array


def bootstrapping_parabola(x: NDFloatArray, y: NDFloatArray):
    """Assuming 1000 samples of the coefficients"""
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} != y.shape={y.shape}")

    N = 1000
    a_0 = np.zeros(N)
    a_1 = np.zeros(N)
    a_2 = np.zeros(N)

    num_rows = x.shape[0]
    for i in range(N):
        sample_size = np.random.randint(low=6, high=num_rows)
        sample_indices = np.random.choice(num_rows, size=sample_size, replace=True)
        x_samp = x[sample_indices]
        y_samp = y[sample_indices]
        coef = parabola_fit(x_samp, y_samp)
        a_0[i] = coef[0]
        a_1[i] = coef[1]
        a_2[i] = coef[2]

    # Switching a_0 to u_0 using inverse function A_0 IS U_0 FROM THIS POINT FORWARD
    # a_0 = theory.extract_u0(a_0)
    a_0_val, a_0_sigma = np.mean(a_0), np.std(a_0)
    a_1_val, a_1_sigma = np.mean(a_1), np.std(a_1)
    a_2_val, a_2_sigma = np.mean(a_2), np.std(a_2)
    return a_0_val, a_0_sigma, a_1_val, a_1_sigma, a_2_val, a_2_sigma

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
                 sigma: NDFloatArray, step_size: float,
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
