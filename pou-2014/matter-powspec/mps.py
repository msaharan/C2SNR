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
import p_2014_parameters as exp
import p_2014_cosmology as cos 

########################################################################
## Job specific parameters
########################################################################

### source redshift
z_s = 2

l_ul = 70000
k_step = 10000
#------------------------------------------------------------------------

########################################################################
## Comoving distance between z = 0 and some redshift
########################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------

########################################################################
## kmax corresponding to lmax
########################################################################

def k_max(l, z):
    return np.ceil(l/distance(z))

########################################################################
## constants
########################################################################

c = 3 * 10**8

### cosmology
h = cos.h
H0 = h * 100 * 1000
omegal = cos.omegal
omegam = cos.omegam
omegab = cos.omegab
omegac = cos.omegac
omegamh2 = omegam * h**2
omegabh2 = omegab * h**2
omegach2 = omegac * h**2
YHe = cos.YHe
As = cos.As
ns = cos.ns
cosmo = {'omega_M_0': omegam, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}

### constant factor ouside the integral in equation 20 of P_2014
constfac = (9) * (H0/c)**3 * omegam**2 

### distance between source and observer
dist_s = distance(z_s)
#-----------------------------------------------------------------------

########################################################################
## CAMB
########################################################################
### linear
pars_l = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0 = 100 * h, omch2 = omegach2, ombh2 = omegabh2, YHe = YHe)
pars_l.DarkEnergy.set_params(w = -1)
pars_l.set_for_lmax(lmax = l_ul)
pars_l.InitPower.set_params(ns = ns, As = As)
#results_l = camb.get_background(pars_l)
results_l = camb.get_results(pars_l)
k_l = np.linspace(10**(-5), k_max(l_ul, z_s) ,k_step)
PK_l = get_matter_power_interpolator(pars_l, nonlinear=False, kmax = np.max(k_l), k_hunit = False, hubble_units = False)

### non-linear
pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2 = omegach2, ombh2 = omegabh2, YHe = YHe)
pars_nl.DarkEnergy.set_params(w = -1)
pars_nl.set_for_lmax(lmax = l_ul)
pars_nl.InitPower.set_params(ns = ns, As = As)
#results_nl = camb.get_background(pars_nl)
results_nl = camb.get_results(pars_nl)
k_nl = np.linspace(10**(-5), k_max(l_ul, z_s) ,k_step)
PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = np.max(k_nl), k_hunit = False, hubble_units = False)

#-----------------------------------------------------------------------


########################################################################
## plot 
########################################################################
### canvas
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

### plot
plt.plot(k_l, PK_l.P(z_s, k_l), label = 'Linear')
plt.plot(k_nl, PK_nl.P(z_s, k_nl), label = 'Non-linear')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k\;(1/Mpc)$', fontsize = fs)
plt.ylabel('$P(k)\;(Mpc^3$)', fontsize = fs)
plt.legend(fontsize = fs)
plt.title('Matter Power Spectrum (z = 2)')
#plt.xlim(l_plot_min, l_plot_max)
#plt.ylim(4E-10, 3E-8)
plt.savefig('./plots/matpowspec_{}.pdf'.format(z_s), bbox_inches='tight')
plt.show()

