# This file should contain parsing of the `phot.dat` files and converting them
# to whatever format we need
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy import ndarray, dtype

from .loc_types import NDFloatArray


def mag_to_i(mag):
    """ Convert Magnitude into Percentage Intensity - as per experiment instructions
    Intensity of magnitude 0 for I-band based on https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/ref.html
    Using Johnson scale I_0 gives us intensity in 1 Jy= 10−23 erg⋅s−1⋅cm−2⋅Hz−1
    """
    i_0 = 2635
    return np.full_like(mag, 10) ** (mag / -2.5) * i_0


@dataclass()
class PhotDat:
    """Class representing the contents of phot.dat files from the OGLE experiment database"""
    # The five columns of the `phot.dat` files
    hjd: NDFloatArray
    intensity: NDFloatArray
    intensity_err: NDFloatArray
    see_est: NDFloatArray
    sky_level: NDFloatArray

    @classmethod
    def from_file(cls, path: str) -> "PhotDat":
        """Initializing the data class with path"""
        try:
            # Assuming data is in a structured format like CSV or similar
            df = np.genfromtxt(path, delimiter=" ")
            hjd = df[..., 0]
            I_mag: NDFloatArray = df[..., 1]
            I_mag_err: NDFloatArray = df[..., 2]
            see_est = df[..., 3]
            sky_level = df[..., 4]

            intensity = mag_to_i(I_mag)
            intensity_err = mag_to_i(I_mag_err)

            return cls(hjd, intensity, intensity_err, see_est, sky_level)

        except FileNotFoundError:
            print(f"File '{path}' not found.")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
