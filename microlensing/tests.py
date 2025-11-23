import numpy as np
import scipy
from microlensing import stat as st
from microlensing import utils
from microlensing import theory
from microlensing import photometry as ph
from microlensing import plot


def test_bootstrap_parabola():
    x = np.arange(-10, 10, 0.5)
    mu, sigma = 0, 1  # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 4 * x ** 2 - 9 * x + 3 + s
    print(st.bootstrapping_parabola(x, y))


def test_data(file_path, i_mag_base, u_0, tau, t_0):
    photometry = ph.PhotDat.from_file(file_path)
    # from Ogle - rewrite for specific file, I took an example from 2024
    i_base = ph.mag_to_i(i_mag_base)

    # Centering function
    photometry.hjd -= t_0
    # i_base is taken from OGLE estimation, took a 1.5 multiplier as it works well when
    # A=4 as in my case, play with multiplier to get closer or further from peak
    sample_indices = np.where(photometry.intensity > (i_base * 2.5))
    x = photometry.hjd[sample_indices]
    y = photometry.intensity[sample_indices] / i_base

    u_0_val, u_0_sigma, tau_val, tau_sigma, a_1, a_1_sigma = (
        st.bootstrapping_parabola(x, y)
    )
    print(
        f"u_0: {u_0_val:.3} + {u_0_sigma:.3}; {u_0}\n"
        f"tau: {tau_val:.3} {tau:.3}\n"
        f"a_1: {a_1:.3} {a_1_sigma:.3} {abs(a_1_sigma / a_1):.2%}"
    )


def test_find_peaks(file_path):
    photometry = ph.PhotDat.from_file(file_path)

    SHOW = (None, None)
    peaks, props = scipy.signal.find_peaks(photometry.intensity, rel_height=1/3, width=SHOW, height=SHOW, prominence=SHOW)
    # print(peaks)
    # print(props)
    max_peak_index = np.argmax(props["peak_heights"])
    peak_index = peaks[max_peak_index]
    left_base_index = props["left_bases"][max_peak_index]
    right_base_index = props["right_bases"][max_peak_index]
    left_width_index = int(np.floor(props["left_ips"][max_peak_index]))
    right_width_index = int(np.ceil(props["right_ips"][max_peak_index]))
    print(left_width_index, right_width_index)

    time = photometry.hjd[left_base_index:right_base_index+1]
    intensity = photometry.intensity[left_base_index:right_base_index+1]
    pb_time = photometry.hjd[left_width_index:right_width_index+1]
    pb_intensity = photometry.intensity[left_width_index:right_width_index+1]

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


def test_chi2():
    x = np.arange(-10, 10, 0.5)
    mu, sigma = 0, 1  # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 4 * x ** 2 - 9 * x + 3 + s
    f = lambda x, a0, a1, a2: a0 + a1 * x + a2 * x ** 2
    sigma_array = np.full(40, 0.1)
    chi2, chi_min, axis = st.search_chi_sqaure_min(x, y, sigma_array, np.array([4, -9, 3]), [], f, 10)
    index = np.array(np.unravel_index(np.argmin(chi2), chi2.shape))
    print(f"chimin:{chi_min}, index:{index}")
    print(f"a_0:{axis[0][index[0], 0, 0]}, a_1:{axis[1][0, index[1], 0]}, a_2:{axis[2][0, 0, index[2]]}")


def test_chi2_2d():
    x = np.arange(-10, 10, 0.5)
    mu, sigma = 0, 0.1  # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 9 * x + 3 + s
    f = lambda x, a0, a1: a0 + a1 * x
    sigma_array = np.full(40, 0.)
    chi2, chi_min, axis = st.search_chi_sqaure_min(x, y, sigma_array, np.array([3, 9]), [], f, 10)
    index = np.array(np.unravel_index(np.argmin(chi2), chi2.shape))

    p = plot.Plot(title="heatmap", x_label="a_0", y_label="a_1")
    p.plot_heatmap(x=axis[0], y=axis[1], z=chi2, z_label="chi^2", x_min=axis[0][index[0], 0],
                   y_min=axis[1][0, index[1]], z_min=chi_min)
    p.save("../output/shdfgjsgj")
    print(f"chimin:{chi_min}, index:{index}")
    print(f"a_0:{axis[0][index[0], 0]}, a_1:{axis[1][0, index[1]]}")


def test_part_b(file_path):
    photometry = ph.PhotDat.from_file(file_path)
    # from Ogle - rewrite for specific file, I took an example from 2024
    i_base = ph.mag_to_i(16.239)
    u_0 = 0.252
    tau = 85.801
    t_0 = 2460440.283
    sigma = photometry.intensity_err
    # Centering function
    # i_base is taken from OGLE estimation, took a 1.5 multiplier as it works well when
    # A=4 as in my case, play with multiplier to get closer or further from peak
    # sample_indices = np.where(photometry.intensity > (i_base * 2.5))
    x = photometry.hjd
    y = photometry.intensity / i_base
    # define search func
    chi2, chi_min, axis = st.search_chi_sqaure_min(x, y, sigma, np.array([u_0, t_0]), [tau], theory.total_magnification,
                                                   100)
    index = np.where(chi2 == chi_min)

    p = plot.Plot(title="heatmap", x_label="a_0", y_label="a_1")
    p.plot_heatmap(x=axis[0], y=axis[1], z=chi2, z_label="chi^2", x_min=axis[0][index[0], 0],
                   y_min=axis[1][0, index[1]], z_min=chi_min, num_levels=15)
    p.save("../output/shdfgjsgj")
    print(f"chimin:{chi_min}, index:{index}")
    print(f"u_0:{axis[0][index[0], 0]}, , t_0:{axis[1][0, index[1]]}; u_0,t_0 theo = {u_0:.3} {t_0:.3}")


def test_part_c(file_path):
    photometry = ph.PhotDat.from_file(file_path)
    # from Ogle - rewrite for specific file, I took an example from 2024
    i_base = ph.mag_to_i(16.239)
    u_0 = 0.252
    tau = 85.801
    t_0 = 2460440.283
    sigma = photometry.intensity_err / i_base
    # Centering function
    # i_base is taken from OGLE estimation, took a 1.5 multiplier as it works well when
    # A=4 as in my case, play with multiplier to get closer or further from peak
    # sample_indices = np.where(photometry.intensity > (i_base * 2.5))
    x = photometry.hjd
    y = photometry.intensity / i_base
    # define search func
    chi2, chi_min, axis = st.search_chi_sqaure_min(x, y, sigma, np.array([u_0, t_0, tau]), [],
                                                   theory.magnification_with_blending, 100)
    index = np.where(chi2 == chi_min)

    print(f"chimin:{chi_min}, index:{index}")
    print(
        f"u_0:{axis[0][index[0], 0]}, , t_0:{axis[1][0, index[1]]}, tau:{axis[2][0, 0, index[2]]}; u_0,t_0 theo = {u_0:.3} {t_0:.3}")


def test_part_b1(file):
    photometry = ph.PhotDat.from_file(file)
    # from Ogle - rewrite for specific file, I took an example from 2024
    i_base = ph.mag_to_i(16.781)
    u_0 = 0.440
    tau = 108
    t_0 = 2460122.725
    sigma = photometry.intensity_err
    # Centering function
    # i_base is taken from OGLE estimation, took a 1.5 multiplier as it works well when
    # A=4 as in my case, play with multiplier to get closer or further from peak
    # sample_indices = np.where(photometry.intensity > (i_base * 2.5))
    x = photometry.hjd-t_0+100
    y = photometry.intensity/i_base
    # define search func
    chi2, chi_min, axis = st.search_chi_sqaure_min(x, y, sigma, np.array([u_0, 100]), [tau, 1],
                                                   theory.magnification_with_blending, 100)
    index = np.where(chi2 == chi_min)
    sig_avg = np.average(sigma)
    chi_max = np.max(chi2)
    print(np.average(y))
    print(np.average(sigma))
    o=sig_avg*chi_min**0.5
    print(o)
    print(o/sig_avg)
    chi2 /= chi_min
    # print(chi2)
    p = plot.Plot(title="heatmap", x_label="a_0", y_label="a_1")
    p.plot_heatmap(x=axis[0], y=axis[1], z=chi2.transpose(), z_label="chi^2", x_min=index[0],
                   y_min=index[1], z_min=1)
    p.save("../output/shdfgjsgj")
    print(f"chimin:{chi_min}, index:{index}")
    print(f"chi at index:{chi2[index]}")
    print(f"u_0:{axis[0][index[0], 0]}, , t_0:{axis[1][0, index[1]]}; u_0,t_0 theo = {u_0:.3} {t_0:.3}")


def test_corner_plot():
    ax = np.linspace(1-50, 100-50, 50, dtype=np.float32)
    ay = np.linspace(1-50, 100-50, 50, dtype=np.float32)
    az = np.linspace(1-50, 100-50, 50, dtype=np.float32)
    aw = np.linspace(1-50, 100-50, 50, dtype=np.float32)
    x, y, z, w = np.meshgrid(ax, ay, az, aw, copy=False, sparse=True, indexing="ij")
    # x, y, z = np.meshgrid(ax, ay, az, copy=False, sparse=True, indexing="ij")
    data = (x**2 + 10*y**2) * z * np.sin(w/10)
    # data = (x**2 + 10*y**2) * z

    levels = plot.CHI2_DIFF_CONF_DOF[2] * 10
    figure, axes = plot.heatmap_corner_plot(
        data,
        [ax, ay, az, aw],
        [25, 25, 25, 25],
        levels,
        title="An example center plot",
        axis_names=["x", "y", "z", "w"],
    )
    figure.savefig("../output/test_corner.png")


if __name__ == '__main__':
    ogle_2024_blg_170 = "../data/blending-1/OGLE-2024-BLG-0170.csv"

    # test_bootstrap_parabola()
    # test_data(ogle_2024_blg_170, 16.239, 0.252, 85.801, 2460440.283)
    # test_find_peaks(ogle_2024_blg_170)
    # test_chi2_2d()
    # test_part_b(ogle_2024_blg_170)
    # test_part_c(ogle_2024_blg_170)
    # test_part_b1("../data/blending-1/OGLE-2023-BLG-0096.csv")
    test_corner_plot()
