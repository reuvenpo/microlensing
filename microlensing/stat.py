# This file should hold all our statistics functions
import random
from collections import abc

import numpy as np
import numpy.polynomial.polynomial as poly
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

def parabola_fit(datapionts: NDFloatArray):
    """Uses numpy polynomial fit with least squares for a polynomial of 3rd degree"""
    datapoints = datapionts.transpose()
    coef_array=poly.Polynomial.fit(datapoints[0], datapoints[1],deg=3).convert().coef
    return coef_array

def bootstrapping_parabola(x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
    """Assuming a 1000 samples of the coefficients"""
    a_0 = np.array()
    a_1 = np.array()
    a_2 = np.array()
    datapoints = np.concatenate((x,y)).transpose()
    for i in range(1000):
        """Using permutation on data points - no resampeling of points"""
        sample = np.random.permutation(datapoints)[:random.randrange(4,x.size)]
        coef = parabola_fit(x, sample)
        a_0 = np.append(a_0, coef[0])
        a_1 = np.append(a_1, coef[1])
        a_2 = np.append(a_2, coef[2])
    return