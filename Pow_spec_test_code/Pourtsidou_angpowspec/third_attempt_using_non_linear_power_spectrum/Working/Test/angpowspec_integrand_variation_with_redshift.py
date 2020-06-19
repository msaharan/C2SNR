import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

###############################################################################
# Constant factor in the expression of angular power spectrum
###############################################################################
def constantfactor(redshift):
    return 9 * (H0/c)**3 * omegam0**2 
#------------------------------------------------------------------------------

###############################################################################
## Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

#########################################################
## Hubble ratio H(z)/H0
#######################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#----------------------------------------------------------

###############################################################################
## To calculate the angular power spectrum using the linear matter power spectrum
###############################################################################
def angpowspec_integrand(z, ell):
    dist = distance(z)
    return (1 - dist/chi_s)**2 * PK.P(z, ell/dist)/ hubble_ratio(z)
#------------------------------------------------------------------------------

###############################################################################
# Constants
###############################################################################

omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 67.3 * 1000
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}
l_plot_low_limit = 10
l_plot_upp_limit = 550
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta = 0
eta_D2_L = 5.94 * 10**12 / 10**eta
redshift = 2
l_max = l_plot_upp_limit

nz = 100
kmax = 2
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.02205, omch2=0.1199)
pars.InitPower.set_params(As=2.196e-9, ns=0.9624)
pars.set_for_lmax(2500, lens_potential_accuracy=1)

results = camb.get_background(pars)

k=np.exp(np.log(10)*np.linspace(-6,3,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.arange(0, l_plot_upp_limit)
zlist = [0.5, 1.0, 1.5]
angpowspec_integrand_array = np.zeros(int(l_max))
#------------------------------------------------------------------------------

plt.subplots()
###############################################################################
# Plot from this work
###############################################################################
# Comoving distance between z = 0 and z = source redshift


chi_s = distance(redshift)
for z in zlist:
    for L in tqdm(range(0, int(l_plot_upp_limit))):
        angpowspec_integrand_array[L] = angpowspec_integrand(z, L)
    plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], angpowspec_integrand_array[l_plot_low_limit:l_plot_upp_limit], label='NL, z = {}'.format(z))
    

plt.xlabel('L')
plt.ylabel('APS Integrand')
plt.suptitle("APS Integrand (NL)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
plt.savefig("./J_plots/angpowspec_integrand.pdf")

plt.show()

#------------------------------------------------------------------------------
