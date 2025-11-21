import time

import numpy as np

import microlensing.plot
from microlensing import stat
from microlensing import theory
from microlensing.plot import Plot
from microlensing.stat import chi_squared, bootstrapping_parabola, search_chi_sqaure_min
from microlensing import photometry as ph
from microlensing import utils
import os.path


def main(file_path, output_path="./output", part_b=False, part_c=False, u_0=0.0, t_0=0.0, tau=0.0, f_bl=None):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    photometry = ph.PhotDat.from_file(file_path)
    peak_start, peak_end, i_base, peak_time = find_peak_and_base(photometry)
    # normalizing intensity for the rest of the calculations
    intensity = photometry.intensity / i_base
    intensity_err = photometry.intensity_err / i_base
    # Centralizing around the peak for the Bootstrapping, taylor requires parabola centered around 0
    # Centralize around 100 before parts B and C
    time = photometry.hjd - peak_time
    # Part A
    u_0_val, u_0_sigma, tau_val, tau_sigma, a_1_val, a_1_sigma, a_0_val, a_2_val = (
        bootstrapping_parabola(time[peak_start:peak_end], intensity[peak_start:peak_end]))
    with open(os.path.join(output_path, f"Parabola Parameters - {file_name}.txt"), "w") as f:
        print(
            f"u_0: {u_0_val:.3} + {u_0_sigma:.3} {abs(u_0_sigma / u_0_val):.2%}%; {u_0}\n"
            f"tau: {tau_val:.3n} + {tau_sigma:.3} {abs(tau_sigma / tau_val):.2%}%;  {tau:.3}\n"
            f"a_1: {a_1_val:.3} {a_1_sigma:.3} {abs(a_1_sigma / a_1_val):.2%}%\n"
            f"t_0: {peak_time:.3}; {t_0:.3}",
            file=f
        )
    plot_parabola = Plot(
        "Parabola Approx of Peak - Centered to 0",
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
    plot_parabola.save(os.path.join(output_path, "Parabola Fit Around Peak - " + file_name))
    # End Of Part A
    # Centralize t_0 around 100 for parts B and C
    time += 100

    # Part b - search for u_0, t_0, pass tau and assume f_bl=1
    if part_b:
        f_bl = 1
        chi2, chi_min, axis = search_chi_sqaure_min(time, intensity, intensity_err, np.array([u_0_val, 100]),
                                                    [tau_val, f_bl],
                                                    theory.magnification_with_blending, 128)
        index = np.where(chi2 == chi_min)
        chi2 /= chi_min
        # print(chi2)
        plot_heatmap = Plot(title="Heatmap - 2 var search", x_label="u_0", y_label="t_0")
        plot_heatmap.plot_heatmap(x=axis[0], y=axis[1], z=chi2.transpose(), z_label="chi^2", var_num=2, x_min=index[0],
                                  y_min=index[1], z_min=1)
        plot_heatmap.save(os.path.join(output_path, "Heatmap chi2 as function of u_0 and t_0 " + file_name))

        # Find Max and Min Values in Confidence levels
        lower_coor, upper_coor = utils.upper_and_lower_param_confidence(chi2, axis, 3.3)

        u_0_max = upper_coor[0]
        u_0_min = lower_coor[0]
        t_0_max = upper_coor[1]
        t_0_min = lower_coor[1]

        u_0_chi2 = axis[0][index[0], 0][0]
        t_0_chi2 = axis[1][0, index[1]][0]
        u_0_err = (u_0_max - u_0_min) / 2
        t_0_err = (t_0_max - t_0_min) / 2

        with open(os.path.join(output_path, f"2D Heatmap Params - {file_name}.txt"), "w") as f:
            print(f"chimin:{chi_min}, index:{index}\n"
                  f"chi at index:{chi2[index]}\n"
                  f"u_0:{u_0_chi2} {abs(u_0_err / u_0_chi2):.2%}%, u_0_max = {u_0_max}, u_0_min = {u_0_min}\n"
                  f" t_0:{t_0_chi2} {abs(t_0_err / t_0_chi2):.2%}%, t_0_max = {t_0_max}, t_0_min = {t_0_min}\n"
                  f" u_0,t_0 theo = {u_0:.3} {100:.3}",
                  file=f)

        # Part c - search for u_0, t_0, tau and f_bl
    if part_c:
        f_bl = 0.5
        chi2, chi_min, axis = search_chi_sqaure_min(time, intensity, intensity_err,
                                                    np.array([u_0_val, 100, tau_val,f_bl]),
                                                    [],
                                                    theory.magnification_with_blending, 128)
        index = np.where(chi2 == chi_min)
        chi2 /= chi_min
        # print(chi2)
        corner_plot = Plot(title="Corner Plot - 4 var search", x_label="u_0", y_label="t_0")
        corner_plot.heatmap_corner_plot(data=chi2, data_axes=axis, center=index,
                                        levels=microlensing.plot.CHI2_DIFF_CONF_DOF[4] + 1,
                                        title="Corner Plot - 4 var search over amplification",
                                        axis_names=["u_0", "t_0", "tau", "f_bl"])
        corner_plot.save(os.path.join(output_path, "Corner Plot chi2 - " + file_name))

        # Find Max and Min Values in Confidence levels
        lower_coor, upper_coor = utils.upper_and_lower_param_confidence(chi2, axis, 3.3)

        u_0_max = upper_coor[0]
        u_0_min = lower_coor[0]
        t_0_max = upper_coor[1]
        t_0_min = lower_coor[1]
        tau_max = upper_coor[2]
        tau_min = lower_coor[2]
        f_bl_max = upper_coor[3]
        f_bl_min = lower_coor[3]

        u_0_chi2 = axis[0][index[0], 0, 0][0]
        t_0_chi2 = axis[1][0, index[1], 0][0]
        tau_chi2 = axis[2][0, 0, index[2]][0]
        f_bl_chi2 = axis[3][0, 0, 0, index[3]][0]
        u_0_err = (u_0_max - u_0_min) / 2
        t_0_err = (t_0_max - t_0_min) / 2
        tau_err = (tau_max - tau_min) / 2
        f_bl_err = (f_bl_max - f_bl_min) / 2

        with open(os.path.join(output_path, f"2D Heatmap Params - {file_name}.txt"), "w") as f:
            print(f"chimin:{chi_min}, index:{index}\n"
                  f"chi at index:{chi2[index]}\n"
                  f"u_0:{u_0_chi2} {abs(u_0_err / u_0_chi2):.2%}%, u_0_max = {u_0_max}, u_0_min = {u_0_min}\n"
                  f" t_0:{t_0_chi2} {abs(t_0_err / t_0_chi2):.2%}%, t_0_max = {t_0_max}, t_0_min = {t_0_min}\n"
                  f" tau:{tau_chi2} {abs(tau_err / tau_chi2):.2%}%, tau_max = {tau_max}, tau_min = {tau_min}\n"
                  f" f_bl:{f_bl_chi2} {abs(f_bl_err / f_bl_chi2):.2%}%, f_bl_max = {f_bl_max}, f_bl_min = {f_bl_min}\n"
                  f" u_0,t_0,tau,f_bl theo = {u_0:.3} {100} {tau} {f_bl}",
                  file=f)


def find_peak_and_base(photometry: ph.PhotDat):
    peak_start, peak_end, peak_time = utils.locate_peak_range(photometry.intensity, photometry.hjd, rel_height=1 / 3)
    wide_peak_start, wide_peak_end, a = utils.locate_peak_range(photometry.intensity, photometry.hjd, rel_height=0.85)
    base_intensity = np.average(
        np.delete(
            photometry.intensity, range(wide_peak_start, wide_peak_end)
        )
    )
    return peak_start, peak_end, base_intensity, peak_time


if __name__ == '__main__':
    file_path = './data/blending-1/OGLE-2023-BLG-0172.csv'
    # file_path = './data/blending-1/OGLE-2023-BLG-0096.csv'
    # file_path = './data/blending-1/OGLE-2024-BLG-0170.csv'
    main(file_path, part_b=False, part_c=True)
