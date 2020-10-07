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
# Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def constantfactor(redshift):
   return 9 * (H0)**3 * omegam0**2 / c**2

 ### Eqn 21 of ZZ has c^2 in denominator
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
    return ( 1 /  ( ell * (ell + 1) ) ) * constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf), limit=20000)[0]
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
omegam0 = 0.3
omegal = 0.7
omegab = 0.04
h = 0.7
omegabh2 = 0.04 * 0.07
omegach2 = (omegam0 * h)**2 - omegabh2

c = 3 * 10**8

# My units ms^{-1} / Mpc Check!
H0 = 100 * h * 1000 # m s**(-1) / Mpc units

cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}

fs = 14
mfs = 12

redshift = 8

l_min = 100
l_max = 10000

l_plot_low_limit = l_min
l_plot_upp_limit = l_max

j_low_limit = 1
j_upp_limit = 2

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=omegach2, ombh2=omegabh2, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09) # ZZ took n = 1 and sigma_8 = 0.9
results = camb.get_background(pars)
k = 10**np.linspace(-5,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = True, hubble_units = True)

#------------------------------------------------------------------------------

###############################################################################
# Write data
###############################################################################
plot_curve = open('./text-files/plt_j_{}_lmax_{}.txt'.format(j_upp_limit-1, l_max), 'w')
signal_integrand = open('./text-files/signal_integrand_j_{}_lmax_{}.txt'.format(j_upp_limit-1, l_max), 'w')

constf = constantfactor(redshift)
dist_s = distance(redshift)
print("constf = {}".format(constf))
print("dist_s = {}".format(dist_s))


for L in tqdm(range(l_plot_low_limit, l_plot_upp_limit, 100)):
    plot_curve.write('{}    {}\n'.format(L, angpowspec_integration_without_j(L, redshift)))
    signal_integrand.write('{}    {}\n'.format(L, angpowspec_integrand_without_j(8, L, dist_s, constf)))
plot_curve.close()
#------------------------------------------------------------------------------


fig, ax = plt.subplots(figsize=(7,7))
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

L, CL = np.loadtxt('./text-files/plt_j_{}_lmax_{}.txt'.format(j_upp_limit - 1, l_max), unpack=True)
L, integrand = np.loadtxt('./text-files/signal_integrand_j_{}_lmax_{}.txt'.format(j_upp_limit - 1, l_max), unpack=True)
#plt.plot(L, CL * L * (L + 1) / (2 * 3.14), label = '$C_L$', color = 'black')
plt.plot(L, CL * L * (L + 1) / (2 * 3.14), label = '$C_L$', color = 'black')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)
plt.ylabel('$C_L$', fontsize = fs)
plt.legend(fontsize = fs)
#plt.xlim(l_plot_low_limit, l_plot_upp_limit)
#plt.ylim(4E-10, 3E-8)
plt.savefig('./plots/cl_j_{}_lmax_{}.pdf'.format(j_upp_limit - 1, l_max), bbox_inches='tight')
plt.show()

plt.plot(L, integrand, label = 'Bla')
plt.show()
