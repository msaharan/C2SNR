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
# Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def constantfactor(redshift):
   return (9/4) * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
# Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

###############################################################################
## P_[CII]^N
###############################################################################

def pc2n_s1( PS,k, z):
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=12.0
    transmission=0.3
    A_survey=2.0
    d_nu=1500.
    NEFD=0.155
    N_pix=1500.
    t_int=1500.*3600.
    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    return PN_CII
#------------------------------------------------------------------------------

###############################################################################
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constf):
    dist = distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf), limit=20000)[0]

def c_lj(ell, j, z):
    dist_s = distance(z)
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return tee_z**2 * PK.P(z, k_var) /eta_D2_L
#------------------------------------------------------------------------------

###############################################################################
# Noise Denominator
###############################################################################
def noise_denominator_integrand(l1, l2, ell,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * (ell - small_l))**2 / ((C_l_array[ell, j] + C_l_N) + C_l_array[abs(ell-small_l), j] + (C_l_N))

def noise_denominator_integration(ell,j, redshift):
    return integrate.dblquad(lambda l1, l2: noise_denominator_integrand(l1, l2, ell, j, redshift), l_min, l_max, lambda l1: l_min, lambda l1: np.sqrt(l_max**2 - l1**2))[0] / (ell**2 * (2 * 3.14)**2)
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
h = 0.673
H0 = 100 * h * 1000 # m s**(-1) / Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
redshift = 6.349

tz, k = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/tz_z6_fine.txt', unpack = True)

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
#pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = True, hubble_units = True)

l_min = int(np.min(k) * distance(redshift))
l_max = 50000

delta_l = 36
f_sky = 0.2
l_plot_low_limit = 10
l_plot_upp_limit = 2500
err_stepsize = 50
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
j_upp_limit = 61
j_max = j_upp_limit
two_pi_squared = 2 * 3.14 * 2 * 3.14

eta = 0.088 * h**3
delta_z = 0.5
D2_L = distance(redshift)**2 * abs(distance(redshift  + (delta_z / 2)) - distance(redshift - (delta_z / 2)))
eta_D2_L = 0.88 * h**3 * D2_L
Cshot = (4 * 10**6 * 2 * 3.14**2 / 0.98**3) / D2_L
tee_z  = (np.min(tz) + np.max(tz)) / 2
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.arange(l_plot_low_limit, l_plot_upp_limit)
N_L = np.zeros(n_err_points)

noise_denominator_sum = np.zeros(n_err_points)

C_l_array = np.zeros((int(2**0.5 * l_max), j_upp_limit))
delta_C_L = np.zeros(n_err_points)
pc2 = np.zeros(np.size(k))
cc2 = np.zeros(np.size(k))

pc2 = PK.P(redshift, k) * tee_z**2
cc2 = PK.P(redshift, k) * tee_z**2 / D2_L
C_l_N = pc2n_s1(pc2, k, redshift) / D2_L
#------------------------------------------------------------------------------

###############################################################################
# Write data
###############################################################################
plot_err = open('noise.txt', 'w')

l_fil, j_fil, clj = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/For_thesis/full-plot/text-files/clfile_j_61_l_50001_z_6.349.txt', unpack=True)
inp = interpolate.interp2d(l_fil, j_fil, clj, kind='linear')


for L in tqdm(range(l_plot_low_limit, int(l_max * 2 **0.5))):
    #for L in tqdm(range(l_plot_low_limit, l_max)):
    for j in range(j_low_limit, j_upp_limit):
        #        C_l_array[L, j] = c_lj(L, j, redshift)
        C_l_array[L, j] = inp(L, j)
"""
for L in tqdm(range(l_plot_low_limit , int(l_max * 2 **0.5) )):
    for j in range(j_low_limit, j_upp_limit):
        C_l_array[L, j] = c_lj(L, j, redshift)
"""
L_counter = 0
for L in range(l_plot_low_limit, l_plot_upp_limit, err_stepsize):
    res_j = 0
    for j in tqdm(range(j_low_limit, j_upp_limit)):
        res_sml = 0
        for l1 in range(0, l_max, 100):
            for l2 in range(0, int(np.sqrt(l_max**2 - l1**2)), 100):
                res_sml = res_sml + noise_denominator_integrand(l1, l2, L, j, redshift)
        res_j = res_j + res_sml
    noise_denominator_sum[L_counter] = res_j
    N_L[L_counter] = 1 / (noise_denominator_sum[L_counter])
    plot_err.write('{}  {}\n'.format(L, N_L[L_counter]))
    L_counter = L_counter + 1
plot_err.close()
#------------------------------------------------------------------------------



