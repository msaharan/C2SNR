
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

z_s = 8

l_min = 100
l_max = 10000
l_step = 1

j_ll = 1
j_ul = 1
j_step = 1
j_max = 10


def distance(var):
    return cd.comoving_distance(var, **cosmo)

def k_max(z, ell_max):
    return np.ceil(ell_max/distance(z))

##############################################################################
## Constants
###############################################################################

### cosmological constants (Zahn and Zaldarriaga 2006) 
"""ZAhn
h = 0.7
omegam0 = 0.3
omegamh2 = omegam0 * h**2
omegal = 0.7
omegab = 0.04

omegabh2 = 0.04 * h**2
omegach2 = (omegam0 * h)**2 - omegabh2
H0 = 100 * h  # *1000                     # ms**(-1)/Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
YHe = 0.24770
ns = 0.9603
As = 2.196e-09
c = 3 * 10**8
"""
#old
omegam0 = 0.308
omegal = 0.692
omegabh2 = 0.02214
omegach2 = 0.1187
YHe = 0.24
ns = 0.961
As = 2.196e-09

c = 3 * 10**8
h = 0.678
H0 = 100 * h * 1000 # h m s**(-1) / Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}


#----------------------------- constants end ----------------------------------

###############################################################################
## CAMB parameters
###############################################################################
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=omegach2, ombh2=omegabh2, YHe = YHe)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=ns, As = As) # ZZ_2006 took n = 1 and sigma_8 = 0.9
results = camb.get_background(pars)
k = 10**np.linspace(-5, k_max(z_s, l_max),1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = k_max(z_s, l_max), k_hunit = True, hubble_units = True)

pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=h * 100, omch2=omegach2, ombh2=omegabh2, YHe = YHe)
pars_nl.DarkEnergy.set_params(w=-1.13)
pars_nl.set_for_lmax(lmax = l_max)
pars_nl.InitPower.set_params(ns=ns, As = As)
results = camb.get_background(pars_nl)
k = 10**np.linspace(-5, k_max(z_s, l_max),1000)
PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = k_max(z_s, l_max), k_hunit = True, hubble_units = True)
#------------------------ CAMB parameters end --------------------------------

plt.plot(k, PK.P(z_s,k), label="new")
plt.scatter(k, PK_nl.P(z_s,k), label="Old")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k (1/Mpc)$')
plt.ylabel('$P(k) (Mpc)^3$')
plt.xlim(1E-5, k_max(z_s, l_max))
plt.legend()
plt.savefig('./plots/matter-ps.pdf')
plt.show()
