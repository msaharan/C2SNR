# To compare power spectrum obtained from CAMB data at different redshifts
# P(k) x Growth factor Vs. k

import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  

kn,dkn = np.loadtxt("../../../Pow_spec_test_code/power_spectrum/CAMB_linear.txt", unpack=True)

plt.subplots()
for redshift in range(1,11):
    fgrow = fgrowth(redshift, 0.308, unnormed=True)
    plt.plot(kn,fgrow*dkn, label='z = {}, fg = {}'.format(redshift, round(fgrow,5)))

plt.xlabel(r'k $(h \; Mpc^{-1})$')
plt.ylabel(r'P(k) ($Mpc \; h^{-1})^3$')
plt.suptitle(r"Power Spectrum $\times$ Growth factor")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("powspec_at_diff_z_True.pdf")
plt.show()
