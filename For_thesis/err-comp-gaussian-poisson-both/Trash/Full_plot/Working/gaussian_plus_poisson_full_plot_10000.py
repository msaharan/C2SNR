import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate
from tqdm.auto import tqdm

#############################################################
## Constant factor in the expression of angular power spectrum
############################################################
def constantfactor(redshift):
    return (9/4) * (H0/c)**4 * omegam0**2 *(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
#----------------------------------------------------------

####################################################
## Comoving distance between z = 0 to some redshift
##################################################
def distance(redshift):
    return cd.comoving_distance(redshift, **cosmo)
#------------------------------------------------

##############################################################
## To calculate the angular power spectrum using the linear matter power spectrum
############################################################
# This function assumes that the K_parallel mode is << K_perpendicular mode. 
# K_parallel mode is entirely neglected.
def angpowspec_integration_without_j(ell,redshift):
    return integrate.quad(lambda x: ((chi_s[redshift] - x)/chi_s[redshift])**2 * np.interp(ell/x, PS,dPS) , 0, chi_s[redshift])[0]

# This function accounts for K_parallel also. 
# It uses the discretized version of K_parallel (its jth mode) (2Pi/box_length)*j.

def angpowspec_integration_with_j(ell, j,redshift): 
    return integrate.quad(lambda x: ((chi_s[redshift] - x)/chi_s[redshift])**2 * np.interp((2 * ( (ell/x)**2 + (2 * 3.14 * j / chi_s[redshift])**2 ))**0.5, PS, dPS), 0, chi_s[redshift])[0]                                                                                           
#----------------------------------------------------------

#############################################################
## To calculate the noise moments (equation 14 in Pourtsidou et al. 2014)
############################################################
def N4_integrand(ell,l1,l2, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2*(angpowspec_without_j[ell] + (C_l)) * (angpowspec_without_j[abs(ell-small_l)] + (C_l)) / two_pi_squared

def N4(ell,redshift):
    return integrate.dblquad(lambda l1,l2: N4_integrand(ell,l1,l2, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]

def N3_integrand(ell, l1, l2,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return (angpowspec_with_j[ell, j] + (C_l)) + (angpowspec_with_j[abs(ell-small_l),j]  + (C_l)) / two_pi_squared

def N3(ell, j, redshift):
    return Cshot(redshift) * 2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]

def N2(ell, j, redshift):
    return Tz(redshift)**2 * 0.21**2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / (eta_D2_L**2 * 0.3**4)

def N1(ell, j, redshift):
    return Tz(redshift)**2 * j_max * 0.357 * 2 * (integrate.dblquad(lambda l1,l2: 1 , 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / two_pi_squared) * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / (eta_D2_L**2 * 0.3**3)

def N0(ell, redshift):
    return Tz(redshift)**4 * j_max**2 * 46.64 * (integrate.dblquad(lambda l1,l2: 1 , 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / two_pi_squared)**2 / (eta_D2_L**3 * 0.3**4)
#-------------------------------------------------------------------

###############################################################
## To calculate the denominator of the lensing recosntruction noise term (eq 14 in Pourtsidou et al. 2014)
##############################################################
def noise_denominator_integrand(ell,j, l1, l2, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return angpowspec_with_j[ell, j]*ell*small_l + angpowspec_with_j[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * Cshot(redshift)

def noise_denominator_integration(ell,j, redshift):
    return integrate.dblquad(lambda l1,l2: noise_denominator_integrand(ell,j, l1, l2, redshift),0 , l_max, lambda l2: 0,lambda l2: l_max)[0] / two_pi_squared

def Tz(redshift):
    return 0.0596 * (1+redshift)**2/np.sqrt(0.308 * (1+redshift)**3 + 0.692)

def Cshot(redshift):
    return Tz(redshift)**2 *(1/eta_D2_L) * mass_moment_2 / mass_moment_1**2
#-----------------------------------------------------------------------

#########################################################################
# Constants
#########################################################################
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2
l_plot_low_limit = 10
l_plot_upp_limit = 700
err_stepsize = 300
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
j_upp_limit = 41
j_max = 126
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta_D2_L = 5.94 * 10**12
redshift = 2
l_max = 10000
#---------------------------------------------------------------------------

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
N_L = np.zeros(n_err_points)

noise_denominator_sum = np.zeros(n_err_points)
N0_sum = np.zeros(n_err_points)
N1_sum = np.zeros(n_err_points)
N2_sum = np.zeros(n_err_points)
N3_sum = np.zeros(n_err_points)

angpowspec_without_j = np.zeros(int(2**0.5 * l_max))
angpowspec_with_j = np.zeros((int(2**0.5 * l_max), j_upp_limit))
N4_array = np.zeros(n_err_points)
delta_C_L = np.zeros(n_err_points)
#------------------------------------------------------------------------------

###############################################################################
# CAMB linear power spectrum
##############################################################################
PS,dPS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_linear.txt", unpack=True)
#----------------------------------------------------------------------------

plt.subplots()

#############################################################################
# Plot from Pourtsidou et al. 2014 (with error bars)
#############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("../../Data_files/pourtsidou_xyscan_curve_z_2.txt", unpack=True)

xplot_2 = np.arange(10, x_2[int(np.size(x_2))-1], x_2[int(np.size(x_2))-1]/2000)
tck_2 = interpolate.splrep(x_2,y_2, s=0)
tckerr_2 = interpolate.splrep(x_2,dyh_2,s=0)
yploterr_2 = interpolate.splev(xplot_2, tckerr_2, der=0)
yplot_2 = interpolate.splev(xplot_2, tck_2, der=0)
plt.errorbar(xplot_2,yplot_2, yerr=yploterr_2,color='black', ecolor='yellow', label='Pourtsidou et al. 2014 (z=2)')
#----------------------------------------------------------------------------

############################################################################
# Plot from this work
###########################################################################

# Comoving distance between z = 0 and z = source redshift
chi_s[redshift] = cd.comoving_distance(redshift, **cosmo)

# Filling the angular power spectrum array. We will need this while plotting 
# angular power spectrum vs L
for L in tqdm(range(0, int(l_max * 2**0.5))):
    angpowspec_without_j[L] = constantfactor(redshift) * angpowspec_integration_without_j(L, redshift)

# Creating a text file to store the L, C_L(L), and N_L(L) values. This file
# can be later used to see the value of error bars
fileout = open("./J_text_files/gaussian_plus_poisson_error_bar_integration_over_distance_j_upp_limit_{}_lmax_{}.txt".format(j_upp_limit-1, l_max), "w")

# This j-loop is required to calculate the lensing reconstruction noise
for L in tqdm(range(0, int(l_max * 2**0.5))):
    for j in range(j_low_limit, j_upp_limit):
        angpowspec_with_j[L, j] = constantfactor(redshift) * angpowspec_integration_with_j(L, j, redshift)

L_counter = 0
for L in range(l_plot_low_limit + 10, l_plot_upp_limit, err_stepsize):
    for j in tqdm(range(j_low_limit, j_upp_limit)):
        noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j, redshift)

        N1_sum[L_counter] = N1_sum[L_counter] + N1(L, j, redshift)
        N2_sum[L_counter] = N2_sum[L_counter] + N2(L, j, redshift)
        N3_sum[L_counter] = N3_sum[L_counter] + N3(L, j, redshift)

    N_L[L_counter] = (L**2 * (N0(L, redshift) + N1_sum[L_counter] + N2_sum[L_counter] + N3_sum[L_counter] + N4(L, redshift)) ) / (noise_denominator_sum[L_counter]**2)

    delta_C_L[L_counter] = ( 2 / ((2*L + 1) * delta_l * f_sky))**0.5 * (angpowspec_without_j[L] + N_L[L_counter]) 

    plt.errorbar(L_array[L], angpowspec_without_j[L], yerr = delta_C_L[L_counter], capsize=3, ecolor='blue')

    fileout.write("{}   {}   {}\n".format(L, delta_C_L[L_counter], N_L[L_counter]))

    print("L = {}, delta_C_L = {}, N_L = {}\n".format(L, delta_C_L[L_counter], N_L[L_counter]))
    L_counter = L_counter + 1

fileout.close()

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], angpowspec_without_j[l_plot_low_limit:l_plot_upp_limit], color='blue', label='This work (z = {})'.format(redshift))
plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Angular Power Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
plt.ylim(1E-9,1E-7)
plt.savefig("./J_plots/gaussian_plus_poisson_full_plot_integration_over_distance_j_upp_limit_{}_lmax_{}.pdf".format(j_upp_limit - 1, l_max))
plt.show()
#-----------------------------------------------------------------------------------------

