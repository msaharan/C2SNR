import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  

#constants
N_l = 1.4* (10**(-14))

#array definitions
l = np.arange(0,10001)
result = np.arange(0,10001) 

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)

plt.subplots()
for redshift in range(5,6):
    plt.plot(l[100:10000],N_l * l[100:10000] * (l[100:10000] + 1) * (10**6) ) 
    #plt.plot(l[100:10000],N_l * l[100:10000] * (l[100:10000] + 1) * (10**6) / (2 * np.pi)) 

plt.xlabel('l')
plt.ylabel(r'$C_{l}$')
plt.suptitle("Noise Angular Power Spectrum")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(1E-2,10)
plt.savefig("zalraddiaga.pdf")
plt.show()
