import numpy as np
import math
import cosmolopy.distance as cd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

###############################################################################
# Constant factor; Integration over z
###############################################################################
def z_constantfactor(z):
   return (9/4) * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
## Ang dia distance between z = 0 and some z
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

# ###############################################################################
# ## Comoving distance between z = 0 and some z
# ###############################################################################
# def c_distance(var):
#     return cd.comoving_distance(var, **cosmo)
# #------------------------------------------------------------------------------

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

###############################################################################
## Integration over z
###############################################################################
def z_angpowspec_integrand(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / (hubble_ratio(z))

def z_angpowspec_integration(ell, z):
    constf = z_constantfactor(z)
    dist_s = a_distance(z)
    return integrate.quad(z_angpowspec_integrand, 0.00001, z, args = (ell, dist_s, constf))[0]

def z_angpowspec_integrand_nl(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK_nl.P(z, ell/dist) * (1 + z)**2/ (hubble_ratio(z))

def z_angpowspec_integration_nl(ell, z):
    constf = z_constantfactor(z)
    dist_s = a_distance(z)
    return integrate.quad(z_angpowspec_integrand_nl, 0.00001, z, args = (ell, dist_s, constf))[0]
#------------------------------------------------------------------------------

###############################################################################
# Find k_max and k_lim (needed for CAMB) corresponding to l_max.
###############################################################################
def k_max(z, l_max):
    return np.ceil(l_max/a_distance(z))

def k_lim(z, l_max):
    if k_max(z, l_max) > 0 and k_max(z, l_max)<=10:
        return 1
    elif k_max(z, l_max)>10 and k_max(z, l_max)<=100:
        return 2
    elif k_max(z, l_max)>100 and k_max(z, l_max)<=1000:
        return 3
    elif k_max(z, l_max)>1000 and k_max(z, l_max)<=10000:
        return 4
#------------------------------------------------------------------------------

##############################################################################
# Generate power spectrum using CAMB
###############################################################################
# Linear
def lin_ps(z, l_max):
    pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
    pars.DarkEnergy.set_params(w=-1.13)
    pars.set_for_lmax(lmax = l_max)
    pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
    results = camb.get_background(pars)
    k = 10**np.linspace(-6, k_lim(z, l_max),1000)
    return get_matter_power_interpolator(pars, nonlinear=False, kmax =  k_max(z, l_max), k_hunit = True, hubble_units = True) 

# Non-linear
def nonlin_ps(z, l_max):
    pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
    pars_nl.DarkEnergy.set_params(w=-1.13)
    pars_nl.set_for_lmax(lmax = l_max)
    pars_nl.InitPower.set_params(ns=0.9603, As = 2.196e-09)
    results = camb.get_background(pars_nl)
    k = 10**np.linspace(-6, k_lim(z, l_max),1000)
    return get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = k_max(z, l_max), k_hunit = True, hubble_units = True)
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
l_plot_low_limit = 10
l_plot_upp_limit = 600
l_max = l_plot_upp_limit
z_array = np.arange(6.0, 7.0, 0.5)
#------------------------------------------------------------------------------

###############################################################################
# Store the data into text files
###############################################################################
for z in z_array:
    PK = lin_ps(z, l_max)
    PK_nl = nonlin_ps(z, l_max)
    z_file = open("./text-files/lin-z-{}.txt".format(z), 'w')
    z_file_nl = open("./text-files/nonlin-z-{}.txt".format(z), 'w')

    for L in tqdm(range(10, l_max)):
        z_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, z) / (2 * 3.14) ))

    for L in tqdm(range(10, l_max)):
        z_file_nl.write('{}    {}\n'.format(L, z_angpowspec_integration_nl(L, z) / (2 * 3.14) ))

    z_file.close()
    z_file_nl.close()
#------------------------------------------------------------------------------

###############################################################################
# Read the text files and plot
###############################################################################
# Comment everything before this section if you just want to plot the previously
# generated data.

plt.subplots()
for z in z_array:
    L , z_CL = np.loadtxt('./text-files/lin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(L, z_CL, label='z = {}'.format(z))

plt.title('Variation of linear power spectrum with z')
plt.xlabel('L')
plt.ylabel(r'$C_{L}$')
plt.legend()
plt.savefig("./plots/lin.pdf")
# plt.show()

plt.subplots()
for z in z_array:
    L_nl , z_CL_nl = np.loadtxt('./text-files/nonlin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(L_nl, z_CL_nl,  label='z = {}'.format(z))
    
plt.title('Variation of non-linear power spectrum with z')
plt.xlabel('L')
plt.ylabel(r'$C_{L}$')
plt.legend()
plt.savefig("./plots/nonlin.pdf")
# plt.show()
#------------------------------------------------------------------------------
