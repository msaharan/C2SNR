import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate
from tqdm.auto import tqdm

###############################################################################
# Constant factor in the expression of angular power spectrum
###############################################################################
def constantfactor(redshift):
    return 9 * (H0/c)**3 * omegam0**2 * (fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
#------------------------------------------------------------------------------

###############################################################################
## To calculate the angular power spectrum using the linear matter power spectrum
###############################################################################

# This function accounts for K_parallel also. 
# It uses the discretized version of K_parallel (its jth mode) (2Pi/box_length)*j.
def angpowspec_integrand_with_j(ell, redshift_variable, j): 
    distance_at_redshift = np.interp(redshift_variable, redshift_from_file, distance_from_file) 
    return ((chi_s[redshift] - distance_at_redshift)/chi_s[redshift])**2 * np.interp((2 * ( (ell/distance_at_redshift)**2 + (2 * 3.14 * j / chi_s[redshift])**2 ))**0.5, PS, dPS)

def angpowspec_integration_with_j(ell, j): 
    return integrate.quad(lambda redshift_variable: angpowspec_integrand_with_j(ell, redshift_variable, j), 0, redshift)[0]

#------------------------------------------------------------------------------

###############################################################################
# Constants
###############################################################################
omegam0 = 0.308
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
j_low_limit = 1
j_upp_limit = 3
j_max = 126
eta = 0
eta_D2_L = 5.94 * 10**12 / 10**eta
redshift = 2
l_max = 700
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
chi_s = np.zeros(11)
eta_array = [0]
angpowspec_with_j = np.zeros((int(2**0.5 * l_max), j_upp_limit))
#------------------------------------------------------------------------------

###############################################################################
# CAMB linear power spectrum
###############################################################################
PS,dPS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_linear.txt", unpack=True)
#----------------------------------------------------------------------------

###############################################################################
# Reading file - comoving distance vs redshift
###############################################################################
redshift_from_file, distance_from_file = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/comoving_distance_between_redshift_0_and_2.txt", unpack = True)
#--------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
# Comoving distance between z = 0 and z = source redshift
chi_s[redshift] = cd.comoving_distance(redshift, **cosmo)

fileout = open("./J_text_files/gaussian_plus_poisson_integration_over_redshift_angpowspec_with_j_j_upp_limit_{}_lmax_{}_eta_{}.txt".format(j_upp_limit-1, l_max, eta), "w")

for L in tqdm(range(1, int(l_max * 2**0.5))):
    for j in range(j_low_limit, j_upp_limit):
        angpowspec_with_j[L, j] = constantfactor(redshift) * angpowspec_integration_with_j(L, j) / (L * (L + 1))
        fileout.write('{}   {}   {}\n'.format(L, j, angpowspec_with_j[L, j]))
fileout.close()

