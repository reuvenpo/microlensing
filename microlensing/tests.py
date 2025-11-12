import numpy as np
from microlensing import stat as st
from microlensing import utils as ut
from microlensing import theory as th
from microlensing import photometry as ph
from microlensing import plot as p

def test_bootstrap_parabola():
    x = np.arange(-10, 10, 0.5)
    mu, sigma = 0, 1  # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 4 * x ** 2 - 9 * x + 3 + s
    print(st.bootstrapping_parabola(x, y))

file_path = "./phot - 2024 - BLG - 170 .txt"
def test_data():

    photometry = ph.PhotDat.from_file(file_path)
    # from Ogle - rewrite for specific file, I took an example from 2024
    i_base = (10 ** (16.239 / (-2.5))) * 2635
    u_0 = 0.252
    tau = 85.801
    t_0 = 2460440.283
    #Centering function
    photometry.hjd -= t_0
    #i_base is taken from OGLE estimation, took a 1.5 multiplier as it works well when
    # A=4 as in my case, play with multiplier to get closer or further from peak
    sample_indices = np.where(photometry.intensity > (i_base*2.5 ))
    x = photometry.hjd[sample_indices]
    y = photometry.intensity[sample_indices]/i_base

    a_0, a_0_sigma, a_1, a_1_sigma, a_2, a_2_sigma = (
        st.bootstrapping_parabola(x, y)
    )
    u_0_measured = th.extract_u0(a_0)
    tau_measured = th.extract_tau(u_0_measured,0,a_2,0)
    print(u_0_measured, u_0, tau_measured, tau, a_1, a_1_sigma)

# test_bootstrap_parabola()
test_data()
