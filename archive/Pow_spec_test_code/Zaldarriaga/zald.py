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
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
N_l = 1.4*10**(-14)

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,10001)
constantfactor = np.zeros(11)
result = np.arange(0,10001) 

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)

plt.subplots()
for redshift in range(5,6):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(10,10000):
        result[i] = integration(i,redshift)
     
    plt.plot(l[10:10000],N_l*l[10:10000]* (l[10:10000]+1) * 10**6 / (2*np.pi)) 
    plt.plot(l[10:10000],constantfactor[redshift]* result[10:10000]*l[10:10000]* (l[10:10000]+1) / (2*np.pi), label='z = {}'.format(redshift)) 

plt.xlabel('l')
plt.ylabel(r'$C_{l}^N$')
plt.suptitle("Noise Angular Power Spectrum")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.savefig("zald.pdf")
plt.show()
