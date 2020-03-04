import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth                                                                                                                                                   

#finding fgrowth
fgrow = np.zeros(11)
for i in range(11):
    fgrow[i] = fgrowth(i, 0.308, unnormed=False) # unnormed = True for redshift higher than 2 and False for redshift lower than 2; Read the manual
    print(fgrow[i])

#constants
zcounter = 2 # redshift for this code
omegam0 = 0.308
c = 3 * 10**8
H0 = 73.8 * 1000
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678}
constantfactor = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrow[zcounter] * (1+zcounter))**2

# notation
# 'n' -> new
# 'o' -> old

#array definitions
xs = np.zeros(11)

l = np.arange(0,1001)
resultn = np.arange(0,1001) 
resulto = np.arange(0,1001)

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_linear.txt", unpack=True)
ko,dko = np.loadtxt("../Data_files/input_spectrum.txt", unpack=True) 
ko = 10**ko                             # don't have to do 
dko = 10**dko                           # do this for new file 
dko  = dko * 2 * (np.pi)**2 /(ko**3) # Second column in new file is P(k) instead of delta(k)

for i in range(10,1000):
    xs[zcounter] = cd.comoving_distance(zcounter, **cosmo) # upper limit of integration
    resultn[i] =  integrate.quad(lambda x: np.power((xs[zcounter] - x)/xs[zcounter], 2)* np.interp(l[i]/x, kn,dkn),0,xs[zcounter])[0]
    resulto[i] =  integrate.quad(lambda x: np.power((xs[zcounter] - x)/xs[zcounter], 2)* np.interp(l[i]/x, ko,dko),0,xs[zcounter])[0]

plt.subplots()
plt.plot(l[10:1000],constantfactor * resultn[10:1000], label="New")
plt.plot(l[10:1000],constantfactor * resulto[10:1000], label="Old")
plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("new_vs_old_angpowspec_iteration2.pdf")
plt.show()
