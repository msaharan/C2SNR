import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate

def poisson_noise(ell):
    return mass_moment_4 / ( ell**2 * eta_D2_L * mass_moment_2**2 )
def distance(var):
    return cd.comoving_distance(var, **cosmo)

#constants
omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
h = 0.673
H0 = 100 * h * 1000 # m s**(-1) / Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_plot_low_limit = 10
l_plot_upp_limit = 2500
err_stepsize = 50
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
delta_z = 0.5
redshift = 6.349
D2_L = distance(redshift)**2 * abs(distance(redshift  + (delta_z / 2)) - distance(redshift - (delta_z / 2)))
eta_D2_L = 0.88 * h**3 * D2_L

L = np.zeros(n_err_points)
N_L = np.zeros(n_err_points)

# plt.subplots()

counter = 0
fileout = open("noise.txt", "w")
for ell in range(l_plot_low_limit, l_plot_upp_limit, err_stepsize):
    L[counter] = ell
    N_L[counter] = poisson_noise(ell)
    print(L[counter], N_L[counter])
    fileout.write("{}   {}\n".format(L[counter], N_L[counter]))
    counter = counter + 1
#plt.scatter(L, N_L)
fileout.close()

