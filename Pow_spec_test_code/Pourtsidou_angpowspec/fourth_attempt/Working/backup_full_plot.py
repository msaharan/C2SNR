import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

##############################################################################
## Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------
# ###############################################################################
# ## Ang dia distance between z = 0 and some redshift
# ###############################################################################
# def distance(var):
#     return cd.angular_diameter_distance(var, **cosmo)
# #------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def constantfactor(redshift):
   return 9 * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

###############################################################################
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constf):
    dist = distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2/ hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf), limit=2000)[0]
'''
def angpowspec_integrand_with_j(z, ell, dist_s, constf, j):
    dist = distance(z)
    k_var = ((ell/dist)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return constf * (1 - dist/dist_s)**2 * PK.P(z, k_var) * (1 + z)**2/ hubble_ratio(z)

def angpowspec_integration_with_j(ell, j, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_with_j, 0.0001, redshift, args = (ell, dist_s, constf, j), limit=2000)[0]
'''

def C_l(z, ell, dist_s, constf, j):
    dist = distance(z)
    k_var = ((ell/dist)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return Tz(z)**2 * PK.P(z, k_var) / eta_D2_L
#------------------------------------------------------------------------------

###############################################################################
## To calculate the noise moments (equation 14 in Pourtsidou et al. 2014)
###############################################################################
def N4_integrand(l1, l2, ell, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2 * (angpowspec_without_j[ell] + (C_l_N)) * (angpowspec_without_j[abs(ell-small_l)] + (C_l)) / two_pi_squared

def N4(ell):
    return integrate.nquad(N4_integrand, [(0, l_max), (0, l_max)], args = (ell, redshift), opts = [{'limit' : 5000}, {'limit' : 5000}])[0]

def N3_integrand(l1, l2, ell, j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return Cshot(redshift) * 2 * (angpowspec_with_j[ell, j] + (C_l_N)) + (angpowspec_with_j[abs(ell-small_l),j]  + (C_l)) / two_pi_squared

def N3(ell, j, redshift):
    return integrate.nquad(N3_integrand, [(0, l_max), (0, l_max)], args = (ell, j, redshift), opts = [{'limit' : 5000}, {'limit' : 5000}])[0]
    #opts = [{'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}, {'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}]
def N2(ell, j, N3_for_this_L_j, redshift):
    return Tz(redshift)**2 * mass_moment_2**2 * N3_for_this_L_j / (2 * eta_D2_L**2 * mass_moment_1**4 * Cshot(redshift))

def N1(ell, j, N3_for_this_L_j, redshift):
    return Tz(redshift)**2 * j_max * mass_moment_3 * (l_max**2 / two_pi_squared) * N3_for_this_L_j / (eta_D2_L**2 * mass_moment_1**3 * Cshot(redshift))

def N0(ell, redshift):
    return Tz(redshift)**4 * j_max**2 * mass_moment_4 * (l_max**2 / two_pi_squared)**2 / (eta_D2_L**3 * mass_moment_1**4)
#------------------------------------------------------------------------------

###############################################################################
# To calculate the denominator of the lensing recosntruction noise term (eq 14 in Pourtsidou et al. 2014)
###############################################################################
def noise_denominator_integrand(l1, l2, ell,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))                                                    
    return (angpowspec_with_j[ell, j]*ell*small_l + angpowspec_with_j[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * Cshot(redshift)) / two_pi_squared

def noise_denominator_integration(ell,j, redshift):
    return integrate.nquad(noise_denominator_integrand, [(0, l_max), (0, l_max)] , args = (ell, j, redshift), opts = [{'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}, {'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}])[0] 

def Tz(redshift):
    return 0.0596 * (1+redshift)**2/np.sqrt(omegam0 * (1+redshift)**3 + omegal)

def Cshot(redshift):
    return Tz(redshift)**2 *(1/eta_D2_L) * mass_moment_2 / mass_moment_1**2

#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 100 *  1000 # h m s**(-1) / Mpc units
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}
C_l_N = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2
l_plot_low_limit = 10
l_plot_upp_limit = 600
# err_stepsize = 36
err_stepsize = 250
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
# j_upp_limit = 41
j_upp_limit = 2
j_max = 126
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta = 0
eta_D2_L = 5.94 * 10**12 / 10**eta
redshift = 2
l_max = 600

k_max = np.ceil(l_max/distance(redshift))
if k_max > 0 and k_max <= 10:
    k_lim = 1
elif k_max > 10 and k_max <= 100:
    k_lim = 2
elif k_max > 100 and k_max <= 1000:
    k_lim = 3
elif k_max > 1000 and k_max <=10000:
    k_lim = 4

pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-6,k_lim,1000)
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = k_max, k_hunit = True, hubble_units = True) 
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.arange(0, l_plot_upp_limit)
angpowspec_without_j = np.zeros(int(l_max))
N_L = np.zeros(n_err_points)

noise_denominator_sum = np.zeros(n_err_points)
N0_sum = np.zeros(n_err_points)
N1_sum = np.zeros(n_err_points)
N2_sum = np.zeros(n_err_points)
N3_sum = np.zeros(n_err_points)

angpowspec_without_j = np.zeros(int(2**0.5 * l_max))

angpowspec_without_j_signal_in_final_plot = np.zeros(l_plot_upp_limit)

angpowspec_with_j = np.zeros((int(2**0.5 * l_max), j_upp_limit))
N4_array = np.zeros(n_err_points)
delta_C_L = np.zeros(n_err_points)
#------------------------------------------------------------------------------

plt.subplots()

###############################################################################
# Plot from Pourtsidou et al. 2014
###############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)

plt.plot(x_2, y_2,color='black', label='P14, z = 2.0')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
plot_err = open("./Text_files/plt_err_integration_over_redshift_j_upp_limit_{}_lmax_{}_eta_{}.txt".format(j_upp_limit-1, l_max, eta), "w")
plot_curve = open("./Text_files/plt_integration_over_redshift_j_upp_limit_{}_lmax_{}_eta_{}.txt".format(j_upp_limit-1, l_max, eta), "w")

for L in tqdm(range(1, int(l_max * 2**0.5))):                                                
    # angpowspec_without_j[L] = angpowspec_integration_without_j(L, redshift) / (L * (L + 1))
    angpowspec_without_j[L] = angpowspec_integration_without_j(L, redshift)

for L in tqdm(range(1, l_plot_upp_limit)):
    # angpowspec_without_j_signal_in_final_plot[L] = (L * (L + 1) / (2 * 3.14)) * angpowspec_without_j[L]
    angpowspec_without_j_signal_in_final_plot[L] = (1 / (2 * 3.14)) * angpowspec_without_j[L]
    plot_curve.write('{}    {}\n'.format(L, angpowspec_without_j_signal_in_final_plot[L]))
plot_curve.close()


for L in tqdm(range(1, int(l_max * 2**0.5))):
    for j in range(j_low_limit, j_upp_limit):
        angpowspec_with_j[L, j] = angpowspec_integration_with_j(L, j, redshift) / (L * (L + 1))

L_counter = 0
for L in range(l_plot_low_limit + 8, l_plot_upp_limit, err_stepsize):

    for j in tqdm(range(j_low_limit, j_upp_limit)):
        noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j, redshift)
        N3_for_this_L_j = N3(L, j, redshift)        
        N1_sum[L_counter] = N1_sum[L_counter] + N1(L, j, N3_for_this_L_j, redshift)
        N2_sum[L_counter] = N2_sum[L_counter] + N2(L, j, N3_for_this_L_j, redshift)
        N3_sum[L_counter] = N3_sum[L_counter] + N3_for_this_L_j

    # lensing reconstruction noise N_L(L); equation 14 in Pourtsidou et al. 2014
    N_L[L_counter] = (L**2 * (N0(L, redshift) + N1_sum[L_counter] + N2_sum[L_counter] + N3_sum[L_counter] + N4(L)) ) / (noise_denominator_sum[L_counter]**2)
    
    # error bars: delta_C_L(L); equation 21 in Pourtsidou et al. 2014
    delta_C_L[L_counter] = (L * (L + 1) / (2 * 3.14)) * ( 2 / ((2*L + 1) * delta_l * f_sky))**0.5 * (angpowspec_without_j[L] + N_L[L_counter]) 

    #plt.errorbar(L_array[L], angpowspec_without_j_signal_in_final_plot[L], yerr = delta_C_L[L_counter], capsize=3, ecolor='blue')
    plot_err.write('{}  {}  {}  {}\n'.format(L, angpowspec_without_j_signal_in_final_plot[L], delta_C_L[L_counter], N_L[L_counter]))
    #fileout.write("{}   {}   {}\n".format(L, delta_C_L[L_counter], N_L[L_counter]))
    L_counter = L_counter + 1

plot_err.close()


