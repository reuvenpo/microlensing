# This file should hold all our statistics functions
from collections import abc

import numpy as np

from types import NDFloatArray

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
