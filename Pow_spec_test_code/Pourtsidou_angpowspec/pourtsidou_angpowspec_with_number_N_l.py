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

#def Ee(redshift):
#    junk = np.sqrt(H0**2 * ((omegam0 * (1 + redshift)**3) + omegal))
#    return junk

#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
N_l = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,1001)
constantfactor = np.zeros(11)
result = np.arange(0,1001) 
yerr = np.zeros(1001)
dCl = np.zeros(1001)

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)

plt.subplots()
for redshift in range(2,3):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(10,1000):
        result[i] = integration(i,redshift)
        dCl[i] = np.sqrt(2/((2*i + 1)*delta_l*f_sky)) * ((constantfactor[redshift]*result[i]) + N_l*(i*(i+1)))
     
    plt.errorbar(l[10:1000],constantfactor[redshift]* result[10:1000], yerr = dCl[10:1000], label='z = {}'.format(redshift)) # plots C_l with error bars


plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum (Non-Linear; Pourtsidou's expression)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.savefig("pourtsidou_angpowspec_with_number_N_l.pdf")
plt.show()
