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

def dblintegrand(i, l1, l2):
    small_l = int(np.sqrt(l1**2 + l2**2))

#    return ((result[small_l] * i * small_l) + (result[i - small_l] *i * (i-small_l)))**2/((result[small_l] + (C_l * small_l**2 ))  * (result[i - small_l] + (C_l * (i-small_l)**2)))
    return ((result[small_l] * i * small_l) + (result[i - small_l] *i * (i-small_l)))**2/((result[small_l])  * (result[i - small_l]))

def dblintegration(i):
    #    return integrate.dblquad(lambda l1, l2: dblintegrand(i, l1, l2),1, l_plot_ul, lambda l2: 1,lambda l2:l_plot_ul)[0]
    return integrate.dblquad(lambda l1, l2: dblintegrand(i, l1, l2),1, i, lambda l2: 1,lambda l2:i)[0]


#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8
H0 = 73.8 * 1000
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.4*10**(-14)

l_upper_limit = 11000
l_plot_ll = 100
l_plot_ul = 2000

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
for redshift in range(5,6):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration

    for i in range(l_plot_ll,l_plot_ul):
        print("---------------------------------------------------"+str(i))
        result[i] = integration(i,redshift)
        N_L[i] = 2* (i**2) * (2*np.pi)**2 / dblintegration(i)
     
    plt.plot(l[l_plot_ll:l_plot_ul],N_L[l_plot_ll:l_plot_ul]/ (2*np.pi)) 
#    plt.plot(l[l_plot_ll:l_plot_ul],N_L[l_plot_ll:l_plot_ul]*l[l_plot_ll:l_plot_ul]* (l[l_plot_ll:l_plot_ul]+1) / (2*np.pi)) 

    plt.plot(l[l_plot_ll:l_plot_ul],constantfactor[redshift]* result[l_plot_ll:l_plot_ul]*l[l_plot_ll:l_plot_ul]* (l[l_plot_ll:l_plot_ul]+1) *10**6 / (2*np.pi), label='z = {}'.format(redshift)) 

plt.xlabel('l')
plt.ylabel(r'$C_{l}^N$')
plt.suptitle("Noise Angular Power Spectrum")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.savefig("zald.pdf")
plt.show()
