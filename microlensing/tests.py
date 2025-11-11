import numpy as np
from microlensing import stat as st

def test_bootstrap_parabola():
    x = np.arange(-10,10,0.5)
    mu, sigma = 0, 0.2 # mean and standard deviation
    rng = np.random.default_rng()
    s = rng.normal(mu, sigma, 40)
    y = 4*x**2-9*x+3+s
    print(st.bootstrapping_parabola(x,y))

test_bootstrap_parabola()