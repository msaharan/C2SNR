import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  

def integration(i,redshift):
    return integrate.quad(lambda x: np.power((xs[redshift] - x)/xs[redshift], 2)* np.interp(l[i]/x, kn,dkn),0,xs[redshift])[0]

def dblintegrand(i, l1, l2):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return ((result[small_l] * i*small_l) + (result[i - small_l] *i* (i - small_l)))**2/((result[small_l]))  * (result[i - small_l]))

def dblintegration(i):
    return integrate.dblquad(lambda l1, l2: dblintegrand(i, l1, l2), 1, i, lambda l2: 1,lambda l2:i)[0]


#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2
l_upper_limit = 10000
l_plot_ll = 10
l_plot_ul = 1000

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,l_upper_limit)
constantfactor = np.zeros(11)
result = np.arange(0,l_upper_limit) 
N_L = np.zeros(l_upper_limit)
dC_L = np.zeros(l_upper_limit)

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)

plt.subplots()
for redshift in range(2,3):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(l_plot_ll,l_plot_ul):
        result[i] = integration(i,redshift)
        N_L[i] = 2* (i**2) * (2*np.pi)**2 / dblintegration(i)

        dC_L[i] = np.sqrt(2/((2*i + 1)*delta_l* f_sky)) *((constantfactor[redshift]*result[i]) + N_L[i]) 

    plt.errorbar(l[l_plot_ll:l_plot_ul],constantfactor[redshift]*result[l_plot_ll:l_plot_ul], yerr=dC_L[l_plot_ll:l_plot_ul], label='z = {}'.format(redshift)) 

plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum (Non-Linear; Pourtsidou's expression)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.savefig("pourtsidou_angpowspec_ignoring_sum_k.pdf")
plt.show()
