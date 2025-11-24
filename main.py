import time

import numpy as np

import microlensing.plot
from microlensing import stat
from microlensing import theory
from microlensing.plot import Plot, heatmap_corner_plot
from microlensing.stat import chi_squared, bootstrapping_parabola, search_chi_sqaure_min, CHI2_DIFF_CONF_DOF
from microlensing import photometry as ph
from microlensing import utils
import os.path


def main(file_path, output_path="./output", part_b=False, part_c=False, u_0=0.0, t_0=0.0, tau=0.0, f_bl=None):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {file_name}")
    photometry = ph.PhotDat.from_file(file_path)
    peak_start, peak_end, i_base, peak_time, left_base_index, right_base_index = find_peak_and_base(photometry)
    # normalizing intensity for the rest of the calculations
    intensity = photometry.intensity[left_base_index:right_base_index] / i_base
    intensity_err = photometry.intensity_err[left_base_index:right_base_index] / i_base
    # Centralizing around the peak for the Bootstrapping, taylor requires parabola centered around 0
    # Centralize around 100 before parts B and C
    time = photometry.hjd[left_base_index:right_base_index] - peak_time

    peak_start -= left_base_index
    peak_end -= left_base_index

    # Part A
    u_0_val, u_0_sigma, tau_val, tau_sigma, a_1_val, a_1_sigma, a_0_val, a_2_val = (
        bootstrapping_parabola(time[peak_start:peak_end], intensity[peak_start:peak_end])
    )
    with open(os.path.join(output_path, f"{file_name} - Parabola Parameters.txt"), "w") as f:
        print(
            f"u_0: {u_0_val:.3f} + {u_0_sigma:.3f} {abs(u_0_sigma / u_0_val):.2%}; {u_0:.3f}\n"
            f"tau: {tau_val:.3f} + {tau_sigma:.3f} {abs(tau_sigma / tau_val):.2%};  {tau:.3f}\n"
            f"a_1: {a_1_val:.3f} {a_1_sigma:.3f} {abs(a_1_sigma / a_1_val):.2%}\n"
            f"t_0: {peak_time:.3f}; {t_0:.3f}",
            file=f
        )
    plot_parabola = Plot(
        f"Parabola Approx of Peak - Centered to 0 for {file_name}",
        x_label="t [days]",
        y_label="Intensity (normalized to base value)"
    )
    plot_parabola.plot_func(
        label="",
        x=time[peak_start:peak_end],
        y=intensity[peak_start:peak_end],
        sigma=intensity_err[peak_start:peak_end],
        func=lambda t: a_2_val * t ** 2 + a_1_val * t + a_0_val
    )
    plot_parabola.save(os.path.join(output_path, f"{file_name} - Parabola Fit Around Peak"))
    # End Of Part A
    # Centralize t_0 around 100 for parts B and C
    time += 100

    # Part b - search for u_0, t_0, pass tau and assume f_bl=1
    if part_b:
        f_bl = 1
        chi2, index_chi_min, meshgrid, axes = search_chi_sqaure_min(
            time,
            intensity,
            intensity_err,
            [u_0_val, 100],
            [tau_val, f_bl],
            theory.magnification_with_blending,
            48,
        )
        chi_min = chi2[index_chi_min]
        # chi2 /= chi_min

        plot_heatmap = Plot(title=f"Heatmap - 2 var search for {file_name}", x_label="u_0", y_label="t_0")
        plot_heatmap.plot_heatmap(
            x=meshgrid[0],
            y=meshgrid[1],
            z=chi2.transpose(),
            z_label="chi^2",
            # var_num=2,
            x_min=index_chi_min[0],
            y_min=index_chi_min[1],
            z_min=1,
        )
        plot_heatmap.save(os.path.join(output_path, f"{file_name} - Heatmap chi2 as function of u_0 and t_0"))

        # Find Max and Min Values in Confidence levels
        lower_coor, upper_coor = utils.upper_and_lower_param_confidence(
            chi2, meshgrid, CHI2_DIFF_CONF_DOF[2, 1] + chi_min,
        )

        u_0_max = upper_coor[0]
        u_0_min = lower_coor[0]
        t_0_max = upper_coor[1]
        t_0_min = lower_coor[1]

        u_0_chi2 = axes[0][index_chi_min[0]]
        t_0_chi2 = axes[1][index_chi_min[1]]
        u_0_err = (u_0_max - u_0_min) / 2
        t_0_err = (t_0_max - t_0_min) / 2

        with open(os.path.join(output_path, f"{file_name} - 2D Heatmap Params.txt"), "w") as f:
            print(
                f"chimin: {chi_min:.3f}, index: {index_chi_min}\n"
                f"u_0: {u_0_chi2:.3f} {abs(u_0_err / u_0_chi2):.2%}, u_0_max = {u_0_max:.3f}, u_0_min = {u_0_min:.3f}\n"
                f"t_0: {t_0_chi2:.3f} {abs(t_0_err / t_0_chi2):.2%}, t_0_max = {t_0_max:.3f}, t_0_min = {t_0_min:.3f}\n"
                f"u_0,t_0 theo = {u_0:.3f} {100}",
                file=f
            )

    # Part c - search for u_0, t_0, tau and f_bl
    if part_c:
        f_bl = 0.5
        chi2, index_chi_min, meshgrid, axes = search_chi_sqaure_min(
            time,
            intensity,
            intensity_err,
            [u_0_val, 100, tau_val,f_bl],
            [],
            theory.magnification_with_blending,
            48,
        )
        chi_min = chi2[index_chi_min]
        # chi2 /= chi_min

        fig, _axes_dict = heatmap_corner_plot(
            data=chi2,
            data_axes=axes,
            center=index_chi_min,
            levels=microlensing.plot.CHI2_DIFF_CONF_DOF[4] + chi_min,
            # levels=None,
            title=f"Corner Plot - 4 var search over amplification for {file_name}",
            axis_names=["u_0", "t_0", "tau", "f_bl"]
        )
        fig.savefig(os.path.join(output_path, f"{file_name} - Corner Plot chi2"))

        # Find Max and Min Values in Confidence levels
        lower_coor, upper_coor = utils.upper_and_lower_param_confidence(chi2, meshgrid, chi_min + CHI2_DIFF_CONF_DOF[4, 1])

        u_0_max = upper_coor[0]
        u_0_min = lower_coor[0]
        t_0_max = upper_coor[1]
        t_0_min = lower_coor[1]
        tau_max = upper_coor[2]
        tau_min = lower_coor[2]
        f_bl_max = upper_coor[3]
        f_bl_min = lower_coor[3]

        u_0_chi2 = axes[0][index_chi_min[0]]
        t_0_chi2 = axes[1][index_chi_min[1]]
        tau_chi2 = axes[2][index_chi_min[2]]
        f_bl_chi2 = axes[3][index_chi_min[3]]
        u_0_err = (u_0_max - u_0_min) / 2
        t_0_err = (t_0_max - t_0_min) / 2
        tau_err = (tau_max - tau_min) / 2
        f_bl_err = (f_bl_max - f_bl_min) / 2

        with open(os.path.join(output_path, f"{file_name} - 2D Heatmap Corner Params.txt"), "w") as f:
            print(
                f"chimin: {chi_min}, index: {index_chi_min}\n"
                f"u_0: {u_0_chi2:.3f} {abs(u_0_err / u_0_chi2):.2%}, u_0_max = {u_0_max:.3f}, u_0_min = {u_0_min:.3f}\n"
                f"t_0: {t_0_chi2:.3f} {abs(t_0_err / t_0_chi2):.2%}, t_0_max = {t_0_max:.3f}, t_0_min = {t_0_min:.3f}\n"
                f"tau: {tau_chi2:.3f} {abs(tau_err / tau_chi2):.2%}, tau_max = {tau_max:.3f}, tau_min = {tau_min:.3f}\n"
                f"f_bl: {f_bl_chi2:.3f} {abs(f_bl_err / f_bl_chi2):.2%}, f_bl_max = {f_bl_max:.3f}, f_bl_min = {f_bl_min:.3f}\n"
                f"u_0,t_0,tau,f_bl theo = {u_0:.3f} {100} {tau:.3f} {f_bl:.3f}",
                file=f
            )


def find_peak_and_base(photometry: ph.PhotDat):
    peak_start, peak_end, peak_time , _, _ = utils.locate_peak_range(photometry.intensity, photometry.hjd, rel_height=1 / 3)
    wide_peak_start, wide_peak_end, _, left_base_index, right_base_index = utils.locate_peak_range(photometry.intensity, photometry.hjd, rel_height=0.85)
    base_intensity = np.average(
        np.delete(
            photometry.intensity, range(wide_peak_start, wide_peak_end)
        )
    )
    return peak_start, peak_end, base_intensity, peak_time, left_base_index, right_base_index


if __name__ == '__main__':
    # Blending = 1
    main('./data/blending-1/OGLE-2023-BLG-0172.csv', part_b=True, part_c=True, u_0=0.272, t_0=2460142.368, tau=101.795, f_bl=1.0)
    main('./data/blending-1/OGLE-2023-BLG-0096.csv', part_b=True, part_c=True, u_0=0.44, t_0=2460122.725, tau=108.527, f_bl=1.0)
    main('./data/blending-1/OGLE-2024-BLG-0170.csv', part_b=True, part_c=True, u_0=0.252, t_0=2460440.283, tau=85.801, f_bl=1.0)

    # Blending < 1
    main('./data/blending-lt-1/OGLE-2023-BLG-0004.csv', part_b=True, part_c=True, u_0=0.309, t_0=2460067.509, tau=55.992, f_bl=0.794)
    main('./data/blending-lt-1/OGLE-2023-BLG-0122.csv', part_b=True, part_c=True, u_0=0.457, t_0=2460047.436, tau=21.084, f_bl=0.767)
    main('./data/blending-lt-1/OGLE-2023-BLG-0373.csv', part_b=True, part_c=True, u_0=0.534, t_0=2460102.777, tau=33.194, f_bl=0.708)
    main('./data/blending-lt-1/OGLE-2023-GD-0004.csv', part_b=True, part_c=True, u_0=0.346, t_0=2460153.320, tau=122.653, f_bl=0.124)
    print("\b")
