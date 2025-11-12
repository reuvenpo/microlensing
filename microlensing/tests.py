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
    i_base = ph.mag_to_i(16.239)
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

    u_0_val, u_0_sigma, tau_val, tau_sigma, a_1, a_1_sigma= (
        st.bootstrapping_parabola(x, y)
    )
    print(f"u_0: {u_0_val:.3} {u_0} tau: {tau_val:.3} {tau} a_1: {a_1:.3} {a_1_sigma:.3} {abs(100*a_1_sigma/a_1)}%")


def test_chi2():
    x = np.arange(-10, 10, 0.5)
    mu, sigma = 0, 1  # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 4 * x ** 2 - 9 * x + 3 + s
    f = lambda x,a0,a1,a2: a0 + a1*x + a2*x**2
    sigma_array = np.full(40, 0.1)
    chi2, chi_min, axis = st.search_chi_sqaure_min(x,y,sigma_array,np.array([4,-9,3]),[],f,3,0.1,10)
    index = np.array(np.unravel_index(np.argmin(chi2),chi2.shape))
    print (f"chimin:{chi_min}, index:{index}")
    print(f"a_0:{axis[0,index[0],0,0]}, a_1:{axis[1,0,index[1],0]}, a_2:{axis[2,0,0,index[2]]}")

if __name__ == '__main__':
    #test_bootstrap_parabola()
    #test_data()
    test_chi2()
