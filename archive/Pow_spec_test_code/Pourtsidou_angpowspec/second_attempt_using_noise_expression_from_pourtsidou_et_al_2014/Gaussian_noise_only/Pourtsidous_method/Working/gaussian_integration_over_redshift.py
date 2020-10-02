import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate

#############################################################
## Constant factor in the expression of angular power spectrum
############################################################
def constantfactor(redshift):
    return 9 * (H0/c)**3 * omegam0**2 *(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
#----------------------------------------------------------

##########################################################
## Hubble ratio H(z)/H0
########################################################
def hubble_ratio(redshift):
    return (omegam0 * (1 + redshift)**3 + omegal)**0.5
#----------------------------------------------------------

####################################################
## Comoving distance between z = 0 to some redshift (Mpc)
###################################################
def distance(redshift):
    return cd.comoving_distance(redshift, **cosmo)
#------------------------------------------------

##############################################################
## To calculate the angular power spectrum using the linear matter power spectrum
############################################################

# This function assumes that the K_parallel mode is << K_perpendicular mode. 
# K_parallel mode is entirely neglected.

def angpowspec_integration_without_j(ell,redshift):
    return integrate.quad(lambda z_variable: ((chi_s[redshift] - distance(z_variable))/chi_s[redshift])**2 * np.interp(ell/distance(z_variable), PS,dPS) / h_ratio , 0, redshift)[0]

# This function accounts for K_parallel also. 
# It uses the discretized version of K_parallel.

def angpowspec_integration_with_j(ell, j,redshift): 
    return integrate.quad(lambda z_variable: ((chi_s[redshift] - distance(z_variable))/chi_s[redshift])**2 * np.interp(np.sqrt(2 * ( (ell/distance(z_variable))**2 + (two_pi_over_chi_s * j)**2 )), PS, dPS) / h_ratio, 0, redshift)[0]
#----------------------------------------------------------

#############################################################
## To calculate the denominator of lensing reconstruction noise
#############################################################
def noise_denominator_integrand(ell,j, l1, l2,redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return (angpowspec_with_j[ell,j] * ell * small_l + angpowspec_with_j[abs(ell-small_l),j] * ell * (ell - small_l))**2/((angpowspec_with_j[ell, j] + C_l) * (angpowspec_with_j[abs(ell-small_l), j] + C_l))

def noise_denominator_integration(ell,j, redshift):
    return integrate.dblquad(lambda l1,l2: noise_denominator_integrand(ell, j, l1, l2, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]
#------------------------------------------------------------

#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.724 * 10**(-16)
delta_l = 36
f_sky = 0.2
l_upper_limit = 20000
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
l_max = 14000
two_pi_squared = 2 * 3.14 * 2 * 3.14

#array definitions
chi_s = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,l_upper_limit)
x_junk = np.zeros(l_upper_limit)
y_junk = np.zeros(l_upper_limit)
angpowspec_without_j = np.zeros(l_upper_limit) 
N_L = np.zeros(l_upper_limit)
delta_C_L = np.zeros(l_upper_limit)
angpowspec_with_j = np.zeros((l_upper_limit, j_upp_limit))

#reading the data file
PS,dPS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_linear.txt", unpack=True)

fileout = open("gaussian_integration_over_redshift_jmax_{}_lmax_{}.txt".format(j_max,l_max), "a")

plt.subplots()

"""
#######################################################
## Curve from Pourtsidou et al. 2014
#####################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_curve_z_2.txt", unpack=True)

xplot_2 = np.arange(10, x_2[int(np.size(x_2))-1], x_2[int(np.size(x_2))-1]/2000)
tck_2 = interpolate.splrep(x_2,y_2, s=0)
tckerr_2 = interpolate.splrep(x_2,dyh_2,s=0)
yploterr_2 = interpolate.splev(xplot_2, tckerr_2, der=0)
yplot_2 = interpolate.splev(xplot_2, tck_2, der=0)
plt.errorbar(xplot_2,yplot_2, yerr=yploterr_2,color='black', ecolor='yellow', label='Pourtsidou et al. 2014 (z=2)')
#---------------------------------------------------
"""

for redshift in range(2,3):
    chi_s[redshift] = cd.comoving_distance(redshift, **cosmo)
    h_ratio = hubble_ratio(redshift)
    two_pi_over_chi_s = 2 * 3.14 / chi_s[redshift]
    # Filling the angular power spectrum array
    for L in range (l_plot_low_limit, l_plot_upp_limit):
        print("Calculating ang. pow. spec. for L = "+str(L))
        angpowspec_without_j[L] = constantfactor(redshift) * angpowspec_integration_without_j(L, redshift)
        for j in range(j_low_limit, j_upp_limit):
            angpowspec_with_j[L,j] = constantfactor(redshift) * angpowspec_integration_with_j(L,j,redshift)
 
    # Error bars for angular power spectrum
    for L in range(l_plot_low_limit + 10, l_plot_upp_limit, err_stepsize):
        noise_denominator_for_this_j = 0
        noise_denominator_sum = 0
        print("-----------Calculating error bars for --------------------L = "+str(L))
        for j in range(j_low_limit, j_upp_limit):
            print("j = " + str(j))
            # Sum over j in the denominator of the lensing reconstruction noise term
            noise_denominator_for_this_j = noise_denominator_integration(L,j, redshift)
            noise_denominator_sum = noise_denominator_sum + noise_denominator_for_this_j
            # Sum over j in the third noise moment N3(L) 

        N_L[L] = (L**2  * two_pi_squared * 2 ) / noise_denominator_sum

        delta_C_L[L] = np.sqrt(2/((2*L + 1)*delta_l* f_sky)) *((angpowspec_without_j[L]) + N_L[L]) 
        plt.errorbar(l[L], angpowspec_without_j[L], yerr = delta_C_L[L], capsize=3, ecolor='blue')
        fileout.write("{}   {}   {}\n".format(L, delta_C_L[L], N_L[L]))
        print("deltaC_L = {}\n".format(delta_C_L[L]))
"""
plt.plot(l[l_plot_low_limit:l_plot_upp_limit], angpowspec_without_j[l_plot_low_limit:l_plot_upp_limit], color='blue', label='This work (z = {})'.format(redshift))
"""
fileout.close()

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle(r"Angular Power Spectrum error bars (Gaussian)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1E-9,1E-7)
plt.xlim(10,1000)
plt.savefig("gaussain_integration_over_redshift_jmax_{}_lmax_{}.pdf".format(j_max,l_max))
plt.show()
