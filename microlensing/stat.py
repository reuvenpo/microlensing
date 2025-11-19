# This file should hold all our statistics functions
import random
from collections import abc
from typing import List

import numpy as np
from numpy.ma.core import shape
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from .loc_types import NDFloatArray
from . import utils
from . import theory

Prediction = abc.Callable[[NDFloatArray], NDFloatArray]
Prediction_single = abc.Callable[[float, NDFloatArray], NDFloatArray]


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


def chi_squared_aggregate(x: NDFloatArray, y: NDFloatArray, sigma: NDFloatArray, f: Prediction) -> float:
    """Compute the chi squared of the data in `y`, `x`, `sigma` given the model `f`

    Example:
        ```
        chi_squared(x, y, sigma, lambda x: a*x + b)
        ```
    """
    prediction = f(x)
    chi_2 = ((y - prediction) / sigma) ** 2
    return chi_2


# Part A
def parabola_fit(x: NDFloatArray, y: NDFloatArray):
    """Uses numpy polynomial fit with the least squares for a polynomial of 3rd degree"""
    coef_array = Polynomial.fit(x, y, deg=2).convert().coef
    return coef_array


def bootstrapping_parabola(x: NDFloatArray, y: NDFloatArray, iterations: int = 1000, min_sample: int = 6):
    """Assuming 1000 samples of the coefficients"""
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} != y.shape={y.shape}")

    # Assuming a parabola of the form
    # a_0 + a_1 * x + a_2 * x**2
    a_0 = np.zeros(iterations)
    a_1 = np.zeros(iterations)
    a_2 = np.zeros(iterations)

    num_rows = x.size
    for i in range(iterations):
        sample_size = np.random.randint(low=min_sample, high=num_rows)
        sample_indices = np.random.choice(num_rows, size=sample_size, replace=False)
        x_samp = x[sample_indices]
        y_samp = y[sample_indices]
        coef = parabola_fit(x_samp, y_samp)
        a_0[i] = coef[0]
        a_1[i] = coef[1]
        a_2[i] = coef[2]

    # Switching a_0 to u_0 using inverse function A_0 IS U_0 FROM THIS POINT FORWARD
    u_0 = theory.extract_u0(a_0)
    u_0_val, u_0_sigma = np.mean(u_0), np.std(u_0)
    a_1_val, a_1_sigma = np.mean(a_1), np.std(a_1)
    a_0_val = np.mean(a_0)
    a_2_val = np.mean(a_2)
    a_2 = np.clip(a_2, None, 0)
    tau = theory.extract_tau(u_0, a_2)
    tau_val, tau_sigma = np.nanmean(tau), np.nanstd(tau)
    return u_0_val, u_0_sigma, tau_val, tau_sigma, a_1_val, a_1_sigma, a_0_val,a_2_val


# End Part A

"""Part B+C"""


def search_chi_sqaure_min(
        x: NDFloatArray,
        y: NDFloatArray,
        sigma: NDFloatArray,
        search_parameters: NDFloatArray,
        static_params: List[float],
        func: Prediction_single,
        chi_limit,
        step_size=0.1,
        resolution=100
):
    """Limit search"""
    dof = (x.size - search_parameters.size)
    limits = limit_search(chi_limit, func, search_parameters, static_params, sigma, dof, x, y)
    axis = utils.split_axis(limits, resolution)
    # axis = np.zeros(shape=(search_parameters.shape[0], resolution))
    # for i in range(search_parameters.shape[0]):
    #     axis[i] = np.linspace(search_parameters[i]*0.5, search_parameters[i]*2, num=resolution)
    # meshgrids = utils.prepare_computation_blocks(parameters_axis)
    meshgrid = np.meshgrid(*axis, indexing='ij', sparse=True)
    chi2 = np.zeros(shape=(resolution,) * search_parameters.shape[0])
    for i in range(x.shape[0]):
        t = x[i]
        val = y[i]
        sig = sigma[i]
        # expecting func to have signature of f(x,a_0,...,a_n, c_0,...,c_n)
        # where a_0 are searched parameters and c are constants not to be searched
        f = lambda x_i: func(x_i, *meshgrid, *static_params)
        val = chi_squared_aggregate(t, val, sig, f) / dof
        chi2 += val
    # reduce chi for degrees of freedom
    return chi2, np.min(chi2), meshgrid


def limit_search(chi_limit, func, parameters: NDFloatArray, static_params: NDFloatArray,
                 sigma: NDFloatArray, dof: float,
                 x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
    limits = np.zeros(shape=(parameters.shape[0], 2))
    chi_base = chi_squared(x, y, sigma, lambda t: func(t, *parameters, *static_params)) / dof
    chi_limit = chi_base * 20
    for index, value in np.ndenumerate(parameters):
        i = 0
        chi = chi_base
        param_search = parameters.copy()
        base_step = value * 0.01
        while chi < chi_limit and i < 1000:
            param_search[index] += base_step
            i += 1
            chi = chi_squared(x, y, sigma, lambda t: func(t, *param_search, *static_params))
            chi = chi / dof
        upperLim = param_search[index]
        chi=chi_base
        i = 0
        param_search = parameters.copy()
        while chi < chi_limit and i < 1000:
            param_search[index] -= base_step
            i += 1
            chi = chi_squared(x, y, sigma, lambda t: func(t, *param_search, *static_params))
            chi = chi / dof
        lowerLim = 0
        if param_search[index] > 0:
            lowerLim = param_search[index]

        limits[index] = np.array([[lowerLim, upperLim]])
    return limits
