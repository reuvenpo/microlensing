# This file is for miscellaneous functions that don't fit elsewhere
import numpy as np


def prepare_computation_blocks(*parameters):
    """This function uses `expand_dims` and `broadcast_arrays` to prepare
    for computing in a parameter subspace defined by the values
    of the `parameters` arrays.

    for example, the following code will compute the 100x100 multiplication table:
    ```
    a = np.array(range(1, 101))
    b = np.array(range(1, 101))
    a, b = prepare_computation_blocks(a, b)
    table = a * b
    ```
    """
    return np.broadcast_arrays(
        *(np.expand_dims(parameter, axis=i) for i, parameter in enumerate(parameters))
    )
