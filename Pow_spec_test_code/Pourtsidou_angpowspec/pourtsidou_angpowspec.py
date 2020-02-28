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

def Ee(redshift):
    junk = np.sqrt(H0**2 * ((omegam0 * (1 + redshift)**3) + omegal))
    return junk

#constants
omegam0 = 0.308
omegal = 0.692
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
kn,dkn = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)

plt.subplots()
for redshift in range(2,3):
    constantfactor[redshift] = (9)* np.power(H0/c,3) * np.power(omegam0,2)
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(10,1000):
        result[i] = integration(i,redshift)
    plt.plot(l[10:1000],constantfactor[redshift] *(1/Ee(redshift)) * result[10:1000], label='z = {}'.format(redshift))

plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum (Non-Linear; Pourtsidou's expression)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.savefig("pourtsidou_angpowspec.pdf")
plt.show()
