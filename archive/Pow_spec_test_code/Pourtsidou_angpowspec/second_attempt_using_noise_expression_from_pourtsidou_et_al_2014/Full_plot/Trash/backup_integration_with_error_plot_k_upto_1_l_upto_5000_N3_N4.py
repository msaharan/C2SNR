import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate

##############################################################
# To calculate the angular power spectrum using the linear matter power spectrum
############################################################

# This function assumes that the K_parallel mode is << K_perpendicular mode. 
# K_parallel mode is entirely neglected.

def angpowspec_integration_without_j(ell,redshift):
    return integrate.quad(lambda x: np.power((chi_s[redshift] - x)/chi_s[redshift], 2)* np.interp(ell/x, PS,dPS),0,chi_s[redshift])[0]

# This function accounts for K_parallel also. 
# It uses the discretized version of K_parallel.

def angpowspec_integration_with_j(ell, j,redshift): 
    return integrate.quad(lambda x: np.power((chi_s[redshift] - x)/chi_s[redshift], 2)* np.interp(np.sqrt(2*((ell/x)**2) + j**2), PS,dPS),0,chi_s[redshift])[0]
#############################################################

#############################################################
## To calculate the noise moments (equation 14 in Pourtsidou et al. 2014)
############################################################
def N4_integrand(ell,l1,l2, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2*(angpowspec_without_j[ell]* constantfactor[redshift] + (C_l)) * (angpowspec_without_j[abs(ell-small_l)]* constantfactor[redshift] + (C_l)) / two_pi_squared

def N4(ell,redshift):
    return integrate.dblquad(lambda l1,l2: N4_integrand(ell,l1,l2, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]

def N3_integrand(ell, l1, l2,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return (angpowspec_with_j[ell, j] * constantfactor[redshift] + (C_l)) + (angpowspec_with_j[abs(ell-small_l),j] * constantfactor[redshift] + (C_l)) / two_pi_squared

def N3(ell, j, redshift):
    return Cshot(redshift) * 2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]

def N2(ell, j, redshift):
    return Tz(redshift)**2 * 0.21**2 * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / ((5.94 * 10**12)**2 * 0.3**4)

def N1(ell, j, redshift):
    return Tz(redshift)**2 * j_max * 0.357 * 2 * (integrate.dblquad(lambda l1,l2: 1 , 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / two_pi_squared) * integrate.dblquad(lambda l1,l2: N3_integrand(ell,l1,l2, j, redshift), 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / ((5.94 * 10**12)**2 * 0.3**3)

def N0(ell, j, redshift):
    return Tz(redshift)**4 * j_max**2 * 46.64 * (integrate.dblquad(lambda l1,l2: 1 , 0, l_max, lambda l2: 0, lambda l2: l_max)[0] / two_pi_squared)**2 / ((5.94 * 10**12)**3 * 0.3**4)
###########################################################

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
    return Tz(redshift)**2 *(1/5.94 * 10**12) * mass_moment_2 / mass_moment_1**2
##############################################################

#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.6*10**(-16)
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
constantfactor = np.zeros(11)
angpowspec_without_j = np.arange(0,l_upper_limit) 
N_L = np.zeros(l_upper_limit)
delta_C_L = np.zeros(l_upper_limit)
#angpowspec_with_j = np.zeros((l_upper_limit, j_ul))
angpowspec_with_j = np.zeros((l_upper_limit, j_upp_limit))

#reading the data file
PS,dPS = np.loadtxt("../../Data_files/CAMB_linear.txt", unpack=True)

fileout = open("integration_with_error_plot_k_upto_2_l_upto_1000_N4.txt", "a")

plt.subplots()

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

for redshift in range(2,3):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    # This should be angular diameter distance
    #chi_s[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration
    chi_s[redshift] = cd.angular_diameter_distance(redshift, **cosmo) # upper limit of integration
    

    # Filling the angular power spectrum array
    for L in range (l_plot_low_limit, l_plot_upp_limit):
        print("Calculating ang. pow. spec. for L = "+str(L))
        angpowspec_without_j[L] = angpowspec_integration_without_j(L, redshift)
        
        for j in range(j_low_limit, j_upp_limit):
            angpowspec_with_j[L,j] = angpowspec_integration_with_j(L,j,redshift)
        
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


        N_L[L] = (L**2 * (N0_sum + N1_sum + N2_sum + N3_sum + N4(L, redshift)) ) / (noise_denominator_sum**2)
        print("Without NL {}".format(np.sqrt(2/((2*L + 1)*delta_l* f_sky)) *((constantfactor[redshift]*angpowspec_without_j[L]))))
        delta_C_L[L] = np.sqrt(2/((2*L + 1)*delta_l* f_sky)) *((constantfactor[redshift]*angpowspec_without_j[L]) + N_L[L]) 
        print("With NL {}".format(N_L[L]))
        plt.errorbar(l[L], constantfactor[redshift]*angpowspec_without_j[L], yerr=delta_C_L[L], capsize=3, ecolor='blue')
        fileout.write("{}   {}\n".format(L,delta_C_L[L]))
        print("deltaC_L = {}\n".format(delta_C_L[L]))
    
plt.plot(l[l_plot_low_limit:l_plot_upp_limit], constantfactor[redshift]*angpowspec_without_j[l_plot_low_limit:l_plot_upp_limit], color='blue', label='This work (z = {})'.format(redshift))

fileout.close()

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle(r"Angular Power Spectrum - LinearPS - N4 Only")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(1E-9,1E-7)
plt.xlim(10,1000)
plt.savefig("integration_with_error_plot_k_upto_2_l_upto_1000_N4.pdf")
plt.show()
