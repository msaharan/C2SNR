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

def N0(ell, j, redshift):
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
#l_upper_limit = 20000
l_plot_low_limit = 10
l_plot_upp_limit = 700
err_stepsize = 300
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
j_low_limit = 1
j_upp_limit = 2
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_max = 126
l_max = l_plot_upp_limit
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta_D2_L_initial = 5.94 * 10**12
redshift = 2

#array definitions
chi_s = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,l_max)
x_junk = np.zeros(l_max)
y_junk = np.zeros(l_max)
angpowspec_without_j = np.zeros(int(l_max * 2**0.5)) 
L_array = np.zeros(n_err_points)
N_L = np.zeros(n_err_points)
delta_C_L = np.zeros(n_err_points)
#angpowspec_with_j = np.zeros((l_upper_limit, j_ul))
angpowspec_with_j = np.zeros((int(l_max * 2**0.5), j_upp_limit))

#reading the data file
PS,dPS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_linear.txt", unpack=True)

plt.subplots()
for n in range(0,2):
    print('Eta{}'.format(n))
    eta_D2_L = eta_D2_L_initial / 10**n
    # This should be angular diameter distance
    chi_s[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration
    fileout = open("gaussian_plus_poisson_noise_j_upp_limit_{}_lmax_{}_eta_{}.txt".format(j_upp_limit, l_max, n), "w")
    # Filling the angular power spectrum array
    for L in tqdm(range (0, int(l_max * 2**0.5))):
        angpowspec_without_j[L] = constantfactor(redshift) * angpowspec_integration_without_j(L, redshift)
        
        for j in range(j_low_limit, j_upp_limit):
            angpowspec_with_j[L,j] = constantfactor(redshift) * angpowspec_integration_with_j(L,j,redshift)
    
    counter = 0
    # Error bars for angular power spectrum
    for L in range(l_plot_low_limit + 10, l_plot_upp_limit,err_stepsize):
        noise_denominator_for_this_j = 0
        noise_denominator_sum = 0
        N3_for_this_j = 0
        N3_sum = 0
        N2_for_this_j = 0
        N2_sum = 0
        N1_for_this_j = 0
        N1_sum = 0
        N0_for_this_j = 0
        N0_sum = 0
        print("-----------Calculating error bars for --------------------L = "+str(L))
        for j in range(j_low_limit, j_upp_limit):
            print("j = " + str(j))
            # Sum over j in the denominator of the lensing reconstruction noise term
            noise_denominator_for_this_j = noise_denominator_integration(L,j, redshift)
            noise_denominator_sum = noise_denominator_sum + noise_denominator_for_this_j
            # Sum over j in the third noise moment N3(L) 
            N3_for_this_j = N3(L, j, redshift)
            N3_sum = N3_sum + N3_for_this_j
            N2_for_this_j = N2(L, j, redshift)
            N2_sum = N2_sum + N2_for_this_j
            N1_for_this_j = N1(L, j, redshift)
            N1_sum = N1_sum + N1_for_this_j
            N0_for_this_j = N0(L, j, redshift)
            N0_sum = N0_sum + N0_for_this_j

        L_array[counter] = L
        print('N0 = {}'.format(N0_sum))
        print('N1 = {}'.format(N1_sum))
        print('N2 = {}'.format(N2_sum))
        print('N3 = {}'.format(N3_sum))
        print('N4 = {}'.format(N4(L, redshift)))
        N_L[counter] = (L**2 * (N0_sum + N1_sum + N2_sum + N3_sum + N4(L, redshift)) ) / (noise_denominator_sum**2)
        fileout.write("{}   {}\n".format(L, N_L[counter]))
        print("N_L = {}\n".format(N_L[counter]))
        counter = counter + 1
    plt.scatter(L_array, N_L, label = 'eta$_{}$'.format(n))
    fileout.close()

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Gaussian + Poisson Noise")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(N_L[0]/10 , N_L[n_err_points - 1] * 10)
plt.xlim(10,1000)
plt.savefig("gaussian_plus_poisson_noise_j_upp_limit_{}_lmax_{}.pdf".format(j_upp_limit, l_max))
plt.show()
