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
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constf):
    dist = distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) / hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf), limit=20000)[0]
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
l_ll = 300
l_ul = 1000
redshift = 6.349

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
tz, k = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/tz_z6_fine.txt', unpack = True)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = True, hubble_units = True) 
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.zeros(l_ul - l_ll)
angpowspec_without_j = np.zeros(l_ul - l_ll)
#------------------------------------------------------------------------------

###############################################################################
# Write data
###############################################################################
counter = 0
for i in tqdm(range(l_ll, l_ul)):
    angpowspec_without_j[counter] = angpowspec_integration_without_j(L, redshift)
    L_array[counter] = i
    counter = counter + 1
plt.plot(L_array, angpowspec_without_j)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$')
plt.ylabel(r'$C_L$')
plt.legend()
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.savefig('./plots/signal.pdf')
plt.show()
#------------------------------------------------------------------------------
