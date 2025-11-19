# This file is for miscellaneous functions that don't fit elsewhere
import numpy as np
import scipy
from microlensing.loc_types import NDFloatArray


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
    axis = np.zeros(shape=(limits.shape[0], resolution))
    for i in range(limits.shape[0]):
        axis_limit = limits[i]
        axis[i] = np.linspace(axis_limit[0], axis_limit[1], num=resolution)
    return axis


def locate_peak_range(intensity, hjd, rel_height=1 / 3):
    SHOW = (None, None)
    peaks, props = scipy.signal.find_peaks(intensity, rel_height=rel_height, width=SHOW, height=SHOW,
                                           prominence=SHOW)
    # print(peaks)
    # print(props)
    max_peak_index = np.argmax(props["peak_heights"])
    peak_index = peaks[max_peak_index]
    left_base_index = props["left_bases"][max_peak_index]
    right_base_index = props["right_bases"][max_peak_index]
    left_width_index = int(np.floor(props["left_ips"][max_peak_index]))
    right_width_index = int(np.ceil(props["right_ips"][max_peak_index]))

    # time = hjd[left_base_index:right_base_index + 1]
    # intensity = intensity[left_base_index:right_base_index + 1]
    # pb_time = hjd[left_width_index:right_width_index + 1]
    # pb_intensity = intensity[left_width_index:right_width_index + 1]

    peak_time = hjd[peak_index]
    return left_width_index, right_width_index + 1, peak_time

    pl = plot.Plot("intensity over time", "time", "intensity")
    pl.plot("data", time, intensity)
    pl.plot("peak", photometry.hjd[peak_index], photometry.intensity[peak_index], style="X")
    pl.plot("left base", photometry.hjd[left_base_index], photometry.intensity[left_base_index], style="X")
    pl.plot("right base", photometry.hjd[right_base_index], photometry.intensity[right_base_index], style="X")
    pl.plot("left width", photometry.hjd[left_width_index], photometry.intensity[left_width_index], style="X")
    pl.plot("right width", photometry.hjd[right_width_index], photometry.intensity[right_width_index], style="X")
    pl.plot_polyfit("parabolic fit", pb_time, pb_intensity, degree=2)
    pl.ax.legend()
    pl.save("../output/test_find_peaks.png")


def upper_and_lower_param_confidence(chi: NDFloatArray, parameters: NDFloatArray, chi_confidence=3.3):
    """Expects parameters to be the meshgrid used to create chi"""
    dim = chi.ndim
    # Expecting where to give indices that follow the rule
    indices_below_threshold = np.where(chi <= chi_confidence)
    min_indices = [np.min(indices) for indices in indices_below_threshold]
    max_indices = [np.max(indices) for indices in indices_below_threshold]

    # Extracting the value of the min\max index of each dimension in the indices
    min_coords = np.array([
        parameters[d][(0,) * d + (min_indices[d],) + (0,) * (dim-d-1)]
        for d in range(dim)
    ])

    max_coords = np.array([
        parameters[d][(0,) * d + (max_indices[d],) + (0,) * (dim-d-1)]
        for d in range(dim)
    ])

    return min_coords, max_coords
