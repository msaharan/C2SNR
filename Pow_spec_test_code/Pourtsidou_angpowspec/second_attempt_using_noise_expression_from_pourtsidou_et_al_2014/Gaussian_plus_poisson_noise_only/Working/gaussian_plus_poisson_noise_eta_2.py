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
    return integrate.dblquad(lambda l1,l2: N4_integrand(ell,l1,l2, redshift), 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0]

def N3_integrand(ell, l1, l2,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return (angpowspec_with_j[ell, j] + (C_l)) + (angpowspec_with_j[abs(ell-small_l),j]  + (C_l)) / two_pi_squared

def N3(ell, j, redshift):
    return Cshot(redshift) * 2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0]

def N2(ell, j, redshift):
    return Tz(redshift)**2 * 0.21**2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0] / (eta_D2_L**2 * 0.3**4)

def N1(ell, j, redshift):
    return Tz(redshift)**2 * j_max * 0.357 * 2 * (integrate.dblquad(lambda l1,l2: 1 , 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0] / two_pi_squared) * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0] / (eta_D2_L**2 * 0.3**3)

def N0(ell, j, redshift):
    return Tz(redshift)**4 * j_max**2 * 46.64 * (integrate.dblquad(lambda l1,l2: 1 , 0, current_l_max, lambda l2: 0, lambda l2: current_l_max)[0] / two_pi_squared)**2 / (eta_D2_L**3 * 0.3**4)
#-------------------------------------------------------------------

###############################################################
## To calculate the denominator of the lensing recosntruction noise term (eq 14 in Pourtsidou et al. 2014)
##############################################################
def noise_denominator_integrand(ell,j, l1, l2, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return angpowspec_with_j[ell, j]*ell*small_l + angpowspec_with_j[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * Cshot(redshift)

def noise_denominator_integration(ell,j, redshift):
    return integrate.dblquad(lambda l1,l2: noise_denominator_integrand(ell,j, l1, l2, redshift),0 , current_l_max, lambda l2: 0,lambda l2: current_l_max)[0] / two_pi_squared

def Tz(redshift):
    return 0.0596 * (1+redshift)**2/np.sqrt(0.308 * (1+redshift)**3 + 0.692)

def Cshot(redshift):
    return Tz(redshift)**2 *(1/eta_D2_L) * mass_moment_2 / mass_moment_1**2
#--------------------------------------------------------------------

#constants
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
j_upp_limit = 126
j_max = 126
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta_D2_L_initial = 5.94 * 10**12
redshift = 2

#array definitions
chi_s = np.zeros(11)
fgrow = np.zeros(11)
L_array = np.zeros(n_err_points)
l_max_array = [700]
eta_array = [2]
x_junk = np.zeros(np.amax(l_max_array))
y_junk = np.zeros(np.amax(l_max_array))
N_L = np.zeros((j_max, n_err_points, len(l_max_array), len(eta_array)))

noise_denominator_sum = np.zeros(n_err_points)
N0_sum = np.zeros(n_err_points)
N1_sum = np.zeros(n_err_points)
N2_sum = np.zeros(n_err_points)
N3_sum = np.zeros(n_err_points)

angpowspec_without_j = np.zeros(int(2**0.5 * np.amax(l_max_array)))
angpowspec_with_j = np.zeros((int(2**0.5 * np.amax(l_max_array)), j_upp_limit))
N4_array = np.zeros(n_err_points)

#reading the data file
PS,dPS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_linear.txt", unpack=True)

plt.subplots()
plt.xlabel('L')
plt.ylabel('N$_L$')
plt.suptitle("Gaussian + Poisson noise")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)

eta_counter = 0
for eta in eta_array:
    print('Eta{}'.format(eta))
    previous_l_max = 0
    l_max_counter = 0
    eta_D2_L = eta_D2_L_initial / 10**eta
    chi_s[redshift] = cd.comoving_distance(redshift, **cosmo)
    for current_l_max in l_max_array:

        for L in tqdm(range(int(previous_l_max * 2**0.5), int(current_l_max * 2**0.5))):
            angpowspec_without_j[L] = constantfactor(redshift) * angpowspec_integration_without_j(L, redshift)

        for j in range(j_low_limit, j_upp_limit):
            print('-------------j = {}----------------'.format(j))
            fileout = open("./J_files/gaussian_plus_poisson_integration_over_distance_j_upp_limit_{}_lmax_{}_eta_{}.txt".format(j,current_l_max, eta), "w")

            for L in tqdm(range(int(previous_l_max * 2**0.5), int(current_l_max * 2**0.5))):
                angpowspec_with_j[L, j] = constantfactor(redshift) * angpowspec_integration_with_j(L, j, redshift)

            L_counter = 0
            for L in range(l_plot_low_limit + 10, l_plot_upp_limit, err_stepsize):
                print('*************L = {}******************'.format(L))
                noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j, redshift)

                N0_sum[L_counter] = N0_sum[L_counter] + N0(L, j, redshift)
                N1_sum[L_counter] = N1_sum[L_counter] + N1(L, j, redshift)
                N2_sum[L_counter] = N2_sum[L_counter] + N2(L, j, redshift)
                N3_sum[L_counter] = N3_sum[L_counter] + N3(L, j, redshift)
                
                L_array[L_counter] = L
                N_L[j, L_counter, l_max_counter, eta_counter] = (L**2 * (N0_sum[L_counter] + N1_sum[L_counter] + N2_sum[L_counter] + N3_sum[L_counter] + N4(L, redshift)) ) / (noise_denominator_sum[L_counter]**2)
                fileout.write("{}   {}\n".format(L_array[L_counter], N_L[j, L_counter, l_max_counter], eta_counter))
                print("L = {}, N_L = {}\n".format(L, N_L[j, L_counter, l_max_counter, eta_counter]))
                L_counter = L_counter + 1
            
            y_low_limit = N_L[j, 0, l_max_counter, eta_counter] / 10
            y_upp_limit = N_L[j_low_limit, L_counter-1, 0, 0] * 10
            plt.ylim(y_low_limit , y_upp_limit)
            plt.scatter(L_array, N_L[j, : ,l_max_counter, eta_counter], label = '({},{},$\eta_{}$)'.format(j, current_l_max, eta), s = 5)
            # plt.legend()
            plt.savefig("./J_plots/gaussian_plus_poisson_integration_over_distance_j_upp_limit_{}_lmax_{}_eta_{}.pdf".format(j,current_l_max, eta))
            fileout.close()
        previous_l_max = current_l_max
        l_max_counter = l_max_counter + 1
    eta_counter = eta_counter + 1
plt.show()

