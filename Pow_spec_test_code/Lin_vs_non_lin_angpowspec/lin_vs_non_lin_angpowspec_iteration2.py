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
    fgrow[i] = fgrowth(i, 0.308, unnormed=False)

#constants
zcounter = 2 
omegam0 = 0.308
c = 3 * 10**8                                                               
H0 = 73.8 * 1000                                                                     
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
constantfactor = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrow[zcounter] * (1+zcounter))**2
                                                                                    

# notation
# 'n' -> Linear
# 'o' -> Non-Linear

#array definitions
xs = np.zeros(11)

l = np.arange(0,1001)
resultn = np.arange(0,1001) 
resulto = np.arange(0,1001)

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_linear.txt", unpack=True)
ko,dko = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True) 

for i in range(10,1000):
    xs[zcounter] = cd.comoving_distance(zcounter, **cosmo) # upper limit of integration
    resultn[i] =  integrate.quad(lambda x: np.power((xs[zcounter] - x)/xs[zcounter], 2)* np.interp(l[i]/x, kn,dkn),0,xs[zcounter])[0]
    resulto[i] =  integrate.quad(lambda x: np.power((xs[zcounter] - x)/xs[zcounter], 2)* np.interp(l[i]/x, ko,dko),0,xs[zcounter])[0]

plt.subplots()
plt.plot(l[10:1000],constantfactor * resultn[10:1000], label="Linear")
plt.plot(l[10:1000],constantfactor * resulto[10:1000], label="Non-Linear")
plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Angular Power Spectrum; z = {}".format(zcounter))
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("comp_lin_non_lin_angpowspec_iteration2.pdf")
plt.show()
