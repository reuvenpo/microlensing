import time

import numpy as np

from microlensing import stat
from microlensing import theory
from microlensing.plot import Plot
from microlensing.stat import chi_squared
from microlensing import photometry


def main():
    a = np.ones(shape=[200]*4, dtype=np.float16)
    time.sleep(10)
    print(a[0])


if __name__ == '__main__':
    main()
