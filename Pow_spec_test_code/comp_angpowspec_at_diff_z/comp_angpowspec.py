#To compare angular matter power spectrum 'C(l)' Vs. 'l' at different redshifts 
# Uses the interpolated P(k) which was originally obtained from CAMB data

import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  

def integration(i,redshift):
    integ = integrate.quad(lambda x: np.power((xs[redshift] - x)/xs[redshift], 2)* np.interp(l[i]/x, kn,dkn),0,xs[redshift])[0]
    return integ

#constants
omegam0 = 0.308
c = 3 * 10**8 
H0 = 73.8 * 1000
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,1001)
constantfactor = np.zeros(11)
result = np.arange(0,1001) 

#reading the data file
kn,dkn = np.loadtxt("../../../Pow_spec_test_code/power_spectrum/CAMB_linear.txt", unpack=True)

plt.subplots()
for redshift in range(1,11):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*fgrowth(redshift, 0.308, unnormed=False)
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(10,1000):
        result[i] = integration(i,redshift)
    plt.plot(l[10:1000],constantfactor[redshift] * result[10:1000], label='z = {}'.format(redshift))

plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("comp_angpowspec.pdf")
plt.show()
