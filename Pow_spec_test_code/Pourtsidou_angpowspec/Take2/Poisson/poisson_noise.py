import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate

#############################################################
## Poisson Noise
#############################################################
def poisson_noise(ell,n):
    return 10**n * mass_moment_4 / ( ell**2 * eta_D2_L * mass_moment_2**2 )
#------------------------------------------------------------
#constants
l_plot_low_limit = 10
l_plot_upp_limit = 700
err_stepsize = 50
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
eta_D2_L = 5.94 * 10**12
eta = 0.027

L = np.zeros(n_err_points)
N_L = np.zeros(n_err_points)

plt.subplots()
for n in range(0,5):
    counter = 0
    print("Eta = {}".format(eta/10**n))
    fileout = open("poisson_noise_eta_{}.txt".format(n), "a")
    for ell in range(l_plot_low_limit + 10, l_plot_upp_limit, err_stepsize):
        L[counter] = ell
        N_L[counter] = poisson_noise(ell,n)
        print(L[counter], N_L[counter])
        fileout.write("{}   {}\n".format(L[counter], N_L[counter]))
        counter = counter + 1
    plt.scatter(L, N_L, label = 'eta$_{}$ = {}'.format(n,eta/10**n))
    fileout.close()
plt.ylim(10**-17, 10**-6)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel(r'$N_L(L)$')
plt.suptitle(r"Poisson Noise")
plt.legend()
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.savefig("poisson_noise.pdf")
plt.show()
