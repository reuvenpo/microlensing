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
    # Note - Sparse = True for better performance, to get correct index out of mesh grid ->
    # use unmeshed 1 arrays and access them by order
    # eg:
    # av,bv,cv=np.meshgrid(a,b,c, indexing='ij', sparse=True)
    # d = av*bv*cv
    # ind = np.unravel_index(np.argmin(d), d.shape)
    # a_cor = a[ind[0]]
    # or
    # a_cor = av[ind[0], 0, 0]
    # b_cor = bv[0,ind[1],0]
    return np.meshgrid(*parameters, indexing='ij', sparse=True)
    # return np.broadcast_arrays(
    #     *(np.expand_dims(parameter, axis=i) for i, parameter in enumerate(parameters))
    # )


def split_axis(limits, resolution):
    axis = np.array([])
    for i,axis_limit in np.ndenumerate(limits):
        np.append(axis, np.linspace(axis_limit[0],axis_limit[1],num=resolution))
    return axis