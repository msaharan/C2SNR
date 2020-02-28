# P(k) vs k plot with Linear and Non-Linear Power Spectrum

import matplotlib.pyplot as plt
import numpy as np
import math

pi = np.pi
#new file
kc,dkc = np.loadtxt("../Data_files/CAMB_linear.txt", unpack=True)
#old file
kg,dkg = np.loadtxt("../Data_files/CAMB_non_linear.txt", unpack=True)


plt.subplots()
plt.plot(kc,dkc, label = "Linear")
plt.plot(kg,dkg, label = "Non-Linear")

plt.xscale("log")
plt.xlabel(r"k ($h \; Mpc^{-1}$)")
plt.ylabel(r"P(k) $(Mpc \; h^{-1})^3$")
plt.yscale("log")
plt.suptitle("Linear vs Non-Linear Matter power spectrum", fontsize=13)
plt.legend()
plt.savefig("lin_vs_non_linear_matterpowspec.pdf")
plt.show()

