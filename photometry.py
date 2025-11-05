# This file should contain parsing of the `phot.dat` files and converting them
# to whatever format we need
from dataclasses import dataclass
from typing import Self

from types import NDFloatArray


@dataclass()
class PhotDat:
    # The five columns of the `phot.dat` files
    hjd: NDFloatArray
    i_mag: NDFloatArray
    i_mag_err: NDFloatArray
    see_est: NDFloatArray
    sky_level: NDFloatArray

    @classmethod
    def from_file(cls, path: str) -> "PhotDat":
        pass
