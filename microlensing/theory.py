# This file should contain definitions of functions derived directly from our
# theoretical framework

from .types import NDFloatArray


def u_at(t: NDFloatArray, u0: float, t0: float, tau: float) -> NDFloatArray:
    """Compute the distance scale at times `t` given `u0`, `t0`, and `tau`"""
    u = (((t - t0)/tau)^2 + u0^2)^(1/2)
    return u


def magnification_at(u: NDFloatArray) -> NDFloatArray:
    """Compute the magnification due to microlensing at distances `u`"""
    u2 = u^2
    magnification = (u2 + 2)/(u*(u2 + 4)^(1/2))
    return magnification
