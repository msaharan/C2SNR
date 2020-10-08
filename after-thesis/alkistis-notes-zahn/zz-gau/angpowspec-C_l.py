#-- Plots the angular power spectrum: C_L^{\theta \theta} only as per
#-- the eq. 21 of Zahn and Zaldarriaga 2006 BUT with a c^3 in the 
#-- denominator. 

#---- How to read the comments ------#
# commented code
## top level heading
### sub-heading and so on
#-- A paragraph/description (reserved for descriptive text, not to be
#-- used for code)
#-------------------------------------

#------- Short codes and references ---#
#-- P_2014 = Pourtsidou et al. 2014
#-- ZZ_2006 = Zahn and Zaldarriaga 2006
#-------------------------------------

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
import time

##############################################################################
## Run specific constants
##############################################################################

### redshift of the source (signal emitter)
z_s = 8

### Fourier modes corresponding to the angular binning
#-- This is the L of N(L) and C_(L, j). We observe the signal in this 
#-- particular mode. Observing the signal over a large L range should
#-- give better and reliable results. Mathematically, it can be seen from
#-- the equation 15 of Pourtsidou et al 2014. The instrumental noise will
#-- be less if we take a larger l_max.
l_min = 100
l_max = 1000

### To control the plot range
#-- We might perform the angular power integration for a large range of L
#-- but might like to see a smaller range in the plot.
l_plot_ll = l_min #-- lower limit
l_plot_ul = l_max #-- upper limit
l_plot_step = 10 #-- step size

#---------------------- run specific constants end ---------------------------

#---------------------------- functions --------------------------------------#

##############################################################################
## Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5

###############################################################################
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constfac):
    dist = distance(z)
    return ( 1 /  ( ell * (ell + 1) ) ) * constfac * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def angpowspec_integration_without_j(ell, z):
  #dist_s = distance(z)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, z, args = (ell, dist_s, constfac), limit=20000)[0]

#------------------------- functions end -------------------------------------

##############################################################################
## Constants
###############################################################################

### cosmological constants (Zahn and Zaldarriaga 2006) 
omegam0 = 0.3
omegal = 0.7
omegab = 0.04
h = 0.7
omegabh2 = 0.04 * h**2
omegach2 = (omegam0 * h)**2 - omegabh2
H0 = 100 * h * 1000                       # ms**(-1)/Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
c = 3 * 10**8

## the constant factor in the ang pow spec expression
constfac = 9 * (H0)**3 * omegam0**2 / c**3
#----------------------------- constants end ----------------------------------


###############################################################################
## CAMB parameters
###############################################################################

#-- The parameters other than omega(m/c/b) anh h have not been taken from ZZ 2006. Do not use these
#-- in the final calculations. Make sure to use the correct values. 
#-- TODO for Mohit: Try to find out if changing ns, As and YHe causes major change
#-- in the results. 
#-- From the previous experience with these numbers, I don't think
#-- that there will be big change in the results so I am proceeding with these values.
#--
#-- k_unit = True -> input of k will be in h/Mpc units
#-- hubble_units = True -> output power spectrum PK.P(k, z) will be in Mpc/h units.
#--
#-- This PK and k business is very confusing because ZZ_2006 or P_2014 might have used
#-- k and PK with or without h in the units. I once tried to find it out but my tests were poorly 
#-- structured so I couldn't quite find out which one they have used. I was trying 
#-- different h, true, false combinations to see which combination gives the best match 
#-- with the plot from P_2014.
#-- See the tests in: 
#-- C2SNR/tree/master/archive/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou
#-- or if you can't find them there, check the permanent location:
#-- C2SNR/tree/before_archive_20201002/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=omegach2, ombh2=omegabh2, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09) # ZZ_2006 took n = 1 and sigma_8 = 0.9
results = camb.get_background(pars)
k = 10**np.linspace(-5,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = True, hubble_units = True)

#------------------------ CAMB parameters end --------------------------------

###############################################################################
## compute and store data
###############################################################################
#-- main data
aps_data = open('./text-files/angpowspec-lmax-{}.txt'.format(l_max), 'w')
#-- supporting data; to check the behaviour of the ang pow spec integrand
#i_aps_data = open('./text-files/integrand-angpowspec-lmax_{}.txt'.format(l_max), 'w')

#-- dist_s = distance between the observer and the emitter (harmless => global parameter)
dist_s = distance(z_s)

for L in tqdm(range(l_plot_ll, l_plot_ul, l_plot_step)):
    aps_data.write('{}    {}\n'.format(L, angpowspec_integration_without_j(L, z_s)))
#    signal_integrand.write('{}    {}\n'.format(L, angpowspec_integrand_without_j(z_s, L, dist_s, constfac)))
aps_data.close()
#-------------------- compute and store data end -------------------------------

#"""

###############################################################################
## Read data and plot
###############################################################################
## canvas and fonts
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

## read
L, CL = np.loadtxt('./text-files/angpowspec-lmax-{}.txt'.format(l_max), unpack=True)
#L, integrand = np.loadtxt('./text-files/signal_integrand_j_{}_lmax_{}.txt'.format(j_upp_limit - 1, l_max), unpack=True)
c = 3 * 10**8

## plot and decorate
plt.plot(L, c * CL * L * (L + 1) / (2 * 3.14), label = '$C_L$', color = 'black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)
plt.ylabel('$C_L$', fontsize = fs)
plt.legend(fontsize = fs)
#plt.xlim(l_plot_low_limit, l_plot_upp_limit)
#plt.ylim(4E-10, 3E-8)
plt.savefig('./plots/angpowspec-lmax-{}.pdf'.format(l_max), bbox_inches='tight')
plt.show()

#------------------------ read data and plot end --------------------------------
