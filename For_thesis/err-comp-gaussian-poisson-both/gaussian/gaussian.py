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

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def z_constantfactor(redshift):
   return (9/4) * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
## Ang dia distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

def Tz(redshift):
    return 0.0596 * (1+redshift)**2/hubble_ratio(redshift)

def C_l(ell, j, z):
    dist_s = distance(z)
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return Tz(z)**2 * PK.P(z, k_var) / D2_L

#############################################################
## To calculate the denominator of lensing reconstruction noise
#############################################################
def noise_denominator_integrand(l1, l2, ell,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))                                               
    return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * (ell - small_l))**2 / (L**2 * (2 * 3.14)**2 * 2 * ( C_l_array[ell, j] + C_l_N)  * ( C_l_array[abs(ell - small_l), j] + C_l_N))

def noise_denominator_integration(ell,j, redshift):
    return integrate.nquad(noise_denominator_integrand, [(0, l_max),( 0, l_max)], args = (ell, j, redshift), opts = [{'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}, {'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}])[0]
#------------------------------------------------------------
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
err_stepsize = 140
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
j_upp_limit = 126
#j_upp_limit = 2
j_max = 51
two_pi_squared = 2 * 3.14 * 2 * 3.14
redshift = 2
D2_L = distance(redshift)**2 * (abs( distance(2.085) - distance(1.915)))
l_max = 10000

C_l_array = np.zeros((int(2**0.5 * l_max), j_upp_limit))
N_L = np.zeros(n_err_points)


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

plot_err = open('./text-files/gaussian.txt', "w")
for L in range (l_plot_low_limit, l_plot_upp_limit):
    for j in range(j_low_limit, j_upp_limit):
        C_l_array[L, j] = C_l(L,j,redshift)

L_counter = 0
for L in range(l_plot_low_limit + 8, l_plot_upp_limit, err_stepsize):
    noise_denominator_sum = np.zeros(n_err_points)
    for j in tqdm(range(j_low_limit, j_upp_limit)):
        noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j, redshift)

    N_L[L_counter] = 1 / (noise_denominator_sum[L_counter]**2)

    plot_err.write('{}  {}\n'.format(L, N_L[L_counter]))
    L_counter = L_counter + 1
plot_err.close()
