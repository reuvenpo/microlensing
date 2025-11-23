# This file should hold all our statistics functions
import random
from collections import abc
from typing import List, Tuple

import numpy as np
from numpy.ma.core import shape
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from .loc_types import NDFloatArray
from . import utils
from . import theory

Prediction = abc.Callable[[NDFloatArray], NDFloatArray]


# The chi squared difference as a function of confidence level and degrees of freedom.
# Values are given for 68.3%, 90%, 95.4%, 99%, 99.73%, and 99.99%
CHI2_DIFF_CONF_DOF = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # filler
    [0.00, 1.00, 2.71, 4.00, 6.63, 9.00, 15.1],  # 1 DoF
    [0.00, 2.30, 4.61, 6.17, 9.21, 11.8, 18.4],  # 2 DoF
    [0.00, 3.53, 6.25, 8.02, 11.3, 14.2, 21.1],  # 3 DoF
    [0.00, 4.72, 7.78, 9.70, 13.3, 16.3, 23.5],  # 4 DoF
    [0.00, 5.89, 9.24, 11.3, 15.1, 18.2, 25.7],  # 5 DoF
    [0.00, 7.04, 11.6, 12.8, 16.8, 20.1, 27.8],  # 6 DoF
])


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
    search_parameters: List[float],
    static_params: List[float],
    func,
    resolution=100,
):
    """Limit search"""
    dof = x.size - len(search_parameters)
    limits = limit_search(func, search_parameters, static_params, sigma, dof, x, y)
    axes = utils.build_axes(limits, resolution)

    # axes = np.zeros(shape=(search_parameters.shape[0], resolution))
    # for i in range(search_parameters.shape[0]):
    #     axes[i] = np.linspace(search_parameters[i]*0.5, search_parameters[i]*2, num=resolution)
    # meshgrids = utils.prepare_computation_blocks(parameters_axis)

    meshgrid = np.meshgrid(*axes, indexing='ij', sparse=True, copy=False)
    chi2 = np.zeros(shape=(resolution,) * len(search_parameters), dtype=np.float32)
    for i in range(x.size):
        t = x[i]
        val = y[i]
        sig = sigma[i]
        # expecting func to have signature of f(x,a_0,...,a_n, c_0,...,c_n)
        # where a_0 are searched parameters and c are constants not to be searched
        f = lambda x_i: func(x_i, *meshgrid, *static_params)
        point_chi2 = chi_squared_aggregate(t, val, sig, f) / dof
        chi2 += point_chi2
        if i % 10 == 0:
            print(f"processed {i}/{x.size} points")

    return chi2, np.unravel_index(np.argmin(chi2), chi2.shape), meshgrid, axes


def limit_search(
    func,
    parameters: List[float],
    static_params: List[float],
    sigma: NDFloatArray,
    dof: float,
    x: NDFloatArray,
    y: NDFloatArray
) -> List[Tuple[float, float]]:
    # limits = np.zeros(shape=(len(parameters), 2))
    limits = [(0.0, 0.0)] * len(parameters)
    chi_base = chi_squared(x, y, sigma, lambda x: func(x, *parameters, *static_params)) / dof
    chi_limit = chi_base * CHI2_DIFF_CONF_DOF[len(parameters), -1] + 1
    max_steps = 200
    for index, value in enumerate(parameters):
        base_step = value * 0.01

        i = 0
        chi = chi_base
        param_search = parameters.copy()
        while chi < chi_limit and i < max_steps:
            param_search[index] += base_step
            chi = chi_squared(x, y, sigma, lambda x: func(x, *param_search, *static_params))
            chi /= dof
            i += 1
        upper_limit = param_search[index]

        i = 0
        chi = chi_base
        param_search = parameters.copy()
        while chi < chi_limit and i < max_steps:
            param_search[index] -= base_step
            chi = chi_squared(x, y, sigma, lambda x: func(x, *param_search, *static_params))
            chi /= dof
            i += 1
        lower_limit = param_search[index] if param_search[index] > 0 else base_step

        limits[index] = (lower_limit, upper_limit)
    return limits
