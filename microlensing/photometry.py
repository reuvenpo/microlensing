# This file should contain parsing of the `phot.dat` files and converting them
# to whatever format we need
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .types import NDFloatArray


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
        """Initializing the data class with path, using example
        from https://www.geeksforgeeks.org/python/reading-dat-file-in-python/"""
        try:
            # Assuming data is in a structured format like CSV or similar
            df = pd.read_csv(path, delimiter='\t', names=['hjd', 'i_mag', 'i_mag_err', 'see_est'])
            hjd = df.hjd.to_numpy(dtype=np.float16)
            i_mag = df.i_mag.to_numpy(dtype=np.float16)
            I_mag_err = df.i_mag_err.to_numpy(dtype=np.float16)
            see_est = df.see_est.to_numpy(dtype=np.float16)
            sky_level = df.sky_level.to_numpy(dtype=np.float16)

        except FileNotFoundError:
            print(f"File '{path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
