# This file should contain definitions of functions derived directly from our
# theoretical framework
from logging import exception
from .loc_types import NDFloatArray
import numpy as np


def u_at(t: NDFloatArray, u0: float, t0: float, tau: float) -> NDFloatArray:
    """Compute the distance scale at times `t` given `u0`, `t0`, and `tau`"""
    u = (((t - t0) / tau) ** 2 + u0 ** 2) ** (1 / 2)
    return u


def u_at_single(t: float, u0: NDFloatArray, t0: NDFloatArray, tau: NDFloatArray) -> NDFloatArray:
    """Compute the distance scale at times `t` given `u0`, `t0`, and `tau`"""
    u = (((t - t0) / tau) ** 2 + u0 ** 2) ** (1 / 2)
    return u


def magnification_with_blending(
        t: float,
        u_0: NDFloatArray,
        t0: NDFloatArray,
        tau: NDFloatArray,
        f_bl: NDFloatArray,
        i_star: NDFloatArray
) -> NDFloatArray:
    """computes magnification with blending and base brightness at time t. Use for batching over Chi Squared"""
    u = u_at_single(t, u_0, t0, tau)
    I = i_star * f_bl * u + i_star * (1 - f_bl)
    return I


def magnification_at(u: NDFloatArray) -> NDFloatArray:
    """Compute the magnification due to microlensing at distances `u`"""
    u2 = u ^ 2
    magnification = (u2 + 2) / (u * (u2 + 4) ^ (1 / 2))
    return magnification


def extract_u0(a_0: NDFloatArray) -> NDFloatArray:
    """Assuming u_0 > 1, using the inverse function according to wolfram alpha and symbolab
    https://www.wolframalpha.com/input?i=inverseFunction%5B%28x%5E2%2B2%29%2F%28x*sqrt%28x%5E2%2B4%29%29%5D
    https://www.symbolab.com/solver/function-inverse-calculator/inverse%20f%5Cleft(x%5Cright)%3D%5Cfrac%7B%5Cleft(x%5E%7B2%7D%2B2%5Cright)%7D%7B%5Cleft(x%5Ccdot%20sqrt%5Cleft(x%5E%7B2%7D%2B4%5Cright)%5Cright)%7D?or=input"""

    # Checking using inverse on np.any to check if all values are a) positive, b)
    # negative, c) contain zero. Else values are bouncing around the zero indicating
    # u_0 close to infty indicating einstien ring.

    # expected - all values of a_0 > 0
    if ~np.any(a_0 > 0):
        return np.sqrt(2 * (1 - a_0 ** 2 + a_0 * np.sqrt(a_0 ** 2 - 1) / (a_0 ** 2 - 1)))
    if ~np.any(a_0 < 0):
        return -np.sqrt(2 * (1 - a_0 ** 2 - a_0 * np.sqrt(a_0 ** 2 - 1) / (a_0 ** 2 - 1)))
    if np.any(a_0 == 0):
        raise Exception("Found Einstein Ring, a_0=0 -> u_0=infty")
    else:
        raise Exception("a_0 is not consistent in positivity, find another sample")


def extract_tau(u0: float, sigma_u0: float, a2: float, sigma_a2):
    tau2 = -8 / (a2 * (u0 ** 2) * ((u0 ** 2) + 4) ** (3 / 2))
    tau = (tau2) ** (1 / 2)
    # To add - sigma tau based on partial differentials
    sigma_tau = 0
    return tau, sigma_tau
