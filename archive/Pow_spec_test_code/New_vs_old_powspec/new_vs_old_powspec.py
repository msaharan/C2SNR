# P(k) vs k plot using CAMB data

import matplotlib.pyplot as plt
import numpy as np
import math

pi = np.pi
#new file
kc,dkc = np.loadtxt("../Data_files/CAMB_linear.txt", unpack=True)
#old file
kg,dkg = np.loadtxt("../Data_files/input_spectrum.txt", unpack=True)
kg = 10.0**kg    # don't have to do this 
dkg = 10.0**dkg  # for new file


plt.subplots()
plt.plot(kc,dkc, label = "New") # Second column in new file is P(k) instead of delta(k)
plt.plot(kg,dkg*2*pi**2/kg**3, label = "Old")

plt.xscale("log")
plt.xlabel(r"k ($h \; Mpc^{-1}$)")
plt.ylabel(r"P(k) $(Mpc \; h^{-1})^3$")
plt.yscale("log")
plt.suptitle("Matter power spectrum (CAMB)", fontsize=13)
plt.legend()
plt.savefig("matterpowspec.pdf")
plt.show()

