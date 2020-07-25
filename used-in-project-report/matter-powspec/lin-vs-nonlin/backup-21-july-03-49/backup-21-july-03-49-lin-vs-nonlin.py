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
# Constant factor; Integration over redshift
###############################################################################
def z_constantfactor(redshift):
   return (9/4) * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
## Ang dia distance between z = 0 and some redshift
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

# ###############################################################################
# ## Comoving distance between z = 0 and some redshift
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
## Integration over redshift
###############################################################################
def z_angpowspec_integrand(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / (hubble_ratio(z))

def z_angpowspec_integration(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand, 0.00001, redshift, args = (ell, dist_s, constf))[0]

def z_angpowspec_integrand_nl(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK_nl.P(z, ell/dist) * (1 + z)**2/ (hubble_ratio(z))

def z_angpowspec_integration_nl(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand_nl, 0.00001, redshift, args = (ell, dist_s, constf))[0]
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
l_plot_upp_limit = 30000
redshift = 6
l_max = l_plot_upp_limit

# Find k_max and k_lim (needed for CAMB) corresponding to l_max.
k_max = np.ceil(l_max/a_distance(redshift))
if k_max > 0 and k_max<=10:
    k_lim = 1
elif k_max>10 and k_max<=100:
    k_lim = 2
elif k_max>100 and k_max<=1000:
    k_lim = 3
elif k_max>1000 and k_max<=10000:
    k_lim = 4
#------------------------------------------------------------------------------

##############################################################################
# Generate power spectrum using CAMB
###############################################################################
# Linear
pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-6, k_lim,1000)
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax =  k_max, k_hunit = True, hubble_units = True) 

# Non-linear
pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars_nl.DarkEnergy.set_params(w=-1.13)
pars_nl.set_for_lmax(lmax = l_max)
pars_nl.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars_nl)
k = 10**np.linspace(-6, k_lim,1000)
PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = k_max, k_hunit = True, hubble_units = True)
#------------------------------------------------------------------------------

###############################################################################
# Store the data into text files
###############################################################################
az_file = open("./text-files/aps-lin.txt", 'w')
mz_file = open("./text-files/mps-lin.txt", 'w')
az_file_nl = open("./text-files/aps-nonlin.txt", 'w')
mz_file_nl = open("./text-files/mps-nonlin.txt", 'w')

p = np.zeros(np.size(k))
p_nl = np.zeros(np.size(k))

p = PK.P(redshift, k)
p_nl = PK_nl.P(redshift, k)
for i in range(np.size(k)):
    mz_file.write('{}   {}\n'.format(k[i], p[i]))
    mz_file_nl.write('{}   {}\n'.format(k[i], p_nl[i]))
mz_file.close()
mz_file_nl.close()


for L in tqdm(range(10, l_max)):
    az_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, redshift) / (2 * 3.14) ))
    az_file_nl.write('{}    {}\n'.format(L, z_angpowspec_integration_nl(L, redshift) / (2 * 3.14) ))

az_file.close()
az_file_nl.close()
#------------------------------------------------------------------------------

###############################################################################
# Read the text files and plot
###############################################################################
# Comment everything before this section if you just want to plot the previously
# generated data.

plt.subplots()
aL , az_CL = np.loadtxt('./text-files/aps-lin.txt', unpack = True)
aL_nl , az_CL_nl = np.loadtxt('./text-files/aps-nonlin.txt', unpack = True)

plt.loglog(aL, az_CL, label='Linear')
plt.loglog(aL_nl, az_CL_nl,  label='Non-linear')
# plt.title('Linear vs non-linear matter angular power spectrum at z = 6')
plt.xlabel('L')
plt.ylabel(r'$C_{L}$')
plt.legend()
plt.savefig("./plots/aps-lin-vs-nonlin.pdf")
#------------------------------------------------------------------------------

plt.subplots()
k , P = np.loadtxt('./text-files/mps-lin.txt', unpack = True)
k_nl , P_nl = np.loadtxt('./text-files/mps-nonlin.txt', unpack = True)
plt.loglog(k, P, label='Linear')
plt.loglog(k_nl, P_nl,  label='Non-linear')
plt.xlabel('k [h/Mpc] ')
plt.ylabel(r'$P(k) [Mpc/h]^3$')
plt.xlim(1E-5, int(l_max/distance(redshift)))
plt.legend()
plt.savefig("./plots/mps-lin-vs-nonlin.pdf")

