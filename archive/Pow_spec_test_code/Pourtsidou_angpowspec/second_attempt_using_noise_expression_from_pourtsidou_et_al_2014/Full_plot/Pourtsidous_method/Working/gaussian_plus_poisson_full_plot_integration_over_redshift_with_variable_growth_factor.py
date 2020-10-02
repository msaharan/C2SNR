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
    return 9 * (H0/c)**3 * omegam0**2

def constantfactor_constant_growth_factor_unnormed_true(redshift):
    return 9 * (H0/c)**3 * omegam0**2 * (fgrowth(redshift, omegam0, unnormed=True) * (1 + redshift))**2

def constantfactor_constant_growth_factor_unnormed_false(redshift):
    return 9 * (H0/c)**3 * omegam0**2 * (fgrowth(redshift, omegam0, unnormed=False) * (1 + redshift))**2

#------------------------------------------------------------------------------

#########################################################
## Hubble ratio H(z)/H0
#######################################################
def hubble_ratio(variable):
    return (omegam0 * (1 + variable)**3 + omegal)**0.5
#----------------------------------------------------------

###############################################################################
## To calculate the angular power spectrum using the linear matter power spectrum
###############################################################################

## By iterpolating the comoving distance at any redshift while integrating

# This function assumes that the K_parallel mode is << K_perpendicular mode. 
# K_parallel mode is entirely neglected.
def angpowspec_integrand_without_j_unnormed_true(z, ell):
    distance_at_redshift = np.interp(z, redshift_from_file, distance_from_file)
    return ((chi_s[redshift] - distance_at_redshift)/chi_s[redshift])**2 * np.interp(ell/distance_at_redshift, PS,dPS) * (fgrowth(z, omegam0, unnormed = True) * (1 + z))**2 / hubble_ratio(z)

def angpowspec_integrand_without_j_unnormed_false(z, ell):
    distance_at_redshift = np.interp(z, redshift_from_file, distance_from_file)
    return ((chi_s[redshift] - distance_at_redshift)/chi_s[redshift])**2 * np.interp(ell/distance_at_redshift, PS,dPS) * (fgrowth(z, omegam0, unnormed = False) * (1 + z))**2 / hubble_ratio(z)

def angpowspec_integration_without_j_unnormed_true(ell):
    return integrate.quad(angpowspec_integrand_without_j_unnormed_true, 0, redshift, args = (ell, ))[0]

def angpowspec_integration_without_j_unnormed_false(ell):
    return integrate.quad(angpowspec_integrand_without_j_unnormed_false, 0, redshift, args = (ell, ))[0]

def angpowspec_integrand_without_j_constant_growth_factor(z, ell):
    distance_at_redshift = np.interp(z, redshift_from_file, distance_from_file)
    return ((chi_s[redshift] - distance_at_redshift)/chi_s[redshift])**2 * np.interp(ell/distance_at_redshift, PS,dPS) / hubble_ratio(z)

def angpowspec_integration_without_j_constant_growth_factor(ell):
    return integrate.quad(angpowspec_integrand_without_j_constant_growth_factor, 0, redshift, args = (ell, ))[0]

###############################################################################
# Constants
###############################################################################

omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 67.3 * 1000
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}
C_l = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2
l_plot_low_limit = 10
l_plot_upp_limit = 550
err_stepsize = 36
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
j_upp_limit = 2
j_max = 126
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta = 0
eta_D2_L = 5.94 * 10**12 / 10**eta
redshift = 2
l_max = 550
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
chi_s = np.zeros(11)
fgrow = np.zeros(11)
angpowspec_without_j_error_bar = np.zeros(n_err_points)
L_array = np.arange(0, l_plot_upp_limit)

eta_array = [0]
x_junk = np.zeros(l_max)
y_junk = np.zeros(l_max)

angpowspec_without_j_unnormed_true = np.zeros(int(2**0.5 * l_max))
angpowspec_without_j_unnormed_false = np.zeros(int(2**0.5 * l_max))

angpowspec_without_j_constant_growth_factor_unnormed_true = np.zeros(int(2**0.5 * l_max))
angpowspec_without_j_constant_growth_factor_unnormed_false = np.zeros(int(2**0.5 * l_max))

angpowspec_without_j_signal_in_final_plot_unnormed_true = np.zeros(l_plot_upp_limit)
angpowspec_without_j_signal_in_final_plot_unnormed_false = np.zeros(l_plot_upp_limit)

angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_true = np.zeros(l_plot_upp_limit)
angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_false = np.zeros(l_plot_upp_limit)
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

plt.subplots()

###############################################################################
# Plot from Pourtsidou et al. 2014 (with error bars)
###############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)
plt.plot(x_2, y_2, color='black', label='Pourtsidou et al. 2014 (z=2)')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
# Comoving distance between z = 0 and z = source redshift
chi_s[redshift] = cd.comoving_distance(redshift, **cosmo)

# Filling the angular power spectrum array. We will need this while plotting 
# angular power spectrum vs L
for L in tqdm(range(1, l_max)):
    angpowspec_without_j_unnormed_true[L] = constantfactor(redshift) * angpowspec_integration_without_j_unnormed_true(L) / (L * (L + 1))
    
    angpowspec_without_j_unnormed_false[L] = constantfactor(redshift) * angpowspec_integration_without_j_unnormed_false(L) / (L * (L + 1))
    
    angpowspec_without_j_constant_growth_factor_unnormed_true[L] = constantfactor_constant_growth_factor_unnormed_true(redshift) * angpowspec_integration_without_j_constant_growth_factor(L) / (L * (L + 1))
    
    angpowspec_without_j_constant_growth_factor_unnormed_false[L] = constantfactor_constant_growth_factor_unnormed_false(redshift) * angpowspec_integration_without_j_constant_growth_factor(L) / (L * (L + 1))

for L in tqdm(range(1, l_plot_upp_limit)):
    angpowspec_without_j_signal_in_final_plot_unnormed_true[L] = (L * (L + 1) / (2 * 3.14)) * angpowspec_without_j_unnormed_true[L]
    
    angpowspec_without_j_signal_in_final_plot_unnormed_false[L] = (L * (L + 1) / (2 * 3.14)) * angpowspec_without_j_unnormed_false[L]
    
    angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_true[L] = (L * (L + 1) / (2 * 3.14)) * angpowspec_without_j_constant_growth_factor_unnormed_true[L]
    
    angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_false[L] = (L * (L + 1) / (2 * 3.14)) * angpowspec_without_j_constant_growth_factor_unnormed_false[L]

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], 2 * angpowspec_without_j_signal_in_final_plot_unnormed_true[l_plot_low_limit:l_plot_upp_limit], color='blue', label='This work (z = {}), V, T, (x 2)'.format(redshift))

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], 2 * angpowspec_without_j_signal_in_final_plot_unnormed_false[l_plot_low_limit:l_plot_upp_limit], color='violet', label='This work (z = {}), V, F, (x 2)'.format(redshift))

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], 2 * angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_true[l_plot_low_limit:l_plot_upp_limit], color='red', label='This work (z = {}), C, T, (x 2)'.format(redshift))

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], 2 * angpowspec_without_j_signal_in_final_plot_constant_growth_factor_unnormed_false[l_plot_low_limit:l_plot_upp_limit], color='green', label='This work (z = {}), C, F, (x 2)'.format(redshift))

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Angular Power Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
plt.ylim(1E-9,1E-7)
plt.savefig("./J_plots/gaussian_plus_poisson_full_plot_integration_over_redshift_comparison_growth_factor.pdf".format(j_upp_limit - 1, l_max, eta))
plt.show()
#------------------------------------------------------------------------------
