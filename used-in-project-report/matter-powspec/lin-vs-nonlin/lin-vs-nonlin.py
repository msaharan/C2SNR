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
## Comoving distance between z = 0 and some redshift
###############################################################################
def c_distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

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
    dist = c_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / (hubble_ratio(z))

def z_angpowspec_integration(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = c_distance(redshift)
    return integrate.quad(z_angpowspec_integrand, 0.00001, redshift, args = (ell, dist_s, constf))[0]

def z_angpowspec_integrand_nl(z, ell, dist_s, constf):
    dist = c_distance(z)
    return constf * (1 - dist/dist_s)**2 * PK_nl.P(z, ell/dist) * (1 + z)**2/ (hubble_ratio(z))

def z_angpowspec_integration_nl(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = c_distance(redshift)
    return integrate.quad(z_angpowspec_integrand_nl, 0.00001, redshift, args = (ell, dist_s, constf))[0]
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
# From table 5 of Planck 2013
omegam0 = 0.308
omegal = 0.692
ombh2 = 0.02214
omch2 = 0.1187
YHe = 0.24
ns = 0.961

c = 3 * 10**8
h = 0.678
H0 = 100 * h * 1000 # h m s**(-1) / Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_plot_low_limit = 10
l_plot_upp_limit = 30000
redshift = 6.349
l_max = l_plot_upp_limit

fs = 14
mfs = 12

# Find k_max and k_lim (needed for CAMB) corresponding to l_max.
k_max = np.ceil(l_max/c_distance(redshift))
if k_max > 0 and k_max<=10:
    k_lim = 1
elif k_max>10 and k_max<=100:
    k_lim = 2
elif k_max>100 and k_max<=1000:
    k_lim = 3
elif k_max>1000 and k_max<=10000:
    k_lim = 4
#------------------------------------------------------------------------------
"""
##############################################################################
# Generate power spectrum using CAMB
###############################################################################
# Linear
pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=h*100, omch2=omch2, ombh2=ombh2, YHe = YHe)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=ns, As = 2.196e-09)
results = camb.get_background(pars)
k = np.arange(10**(-2), k_max, 0.001)
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax =  k_max, k_hunit = True, hubble_units = True) 

# Non-linear
pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=h*100, omch2=omch2, ombh2=ombh2, YHe = YHe)
pars_nl.DarkEnergy.set_params(w=-1.13)
pars_nl.set_for_lmax(lmax = l_max)
pars_nl.InitPower.set_params(ns=ns, As = 2.196e-09)
results = camb.get_background(pars_nl)
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
    az_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, redshift)))
    az_file_nl.write('{}    {}\n'.format(L, z_angpowspec_integration_nl(L, redshift)))

az_file.close()
az_file_nl.close()
#------------------------------------------------------------------------------
"""
###############################################################################
# Read the text files and plot
###############################################################################
# Comment everything before this section if you just want to plot the previously
# generated data.

fig, ax = plt.subplots(figsize=(8,8))
aL , az_CL = np.loadtxt('./text-files/aps-lin.txt', unpack = True)
aL_nl , az_CL_nl = np.loadtxt('./text-files/aps-nonlin.txt', unpack = True)

plt.loglog(aL, az_CL, label='Linear')
plt.loglog(aL_nl, az_CL_nl,  label='Non-linear')
# plt.title('Linear vs non-linear matter angular power spectrum at z = 6')
plt.xlabel('L', fontsize = fs)
plt.ylabel('$C_{L}$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/aps-lin-vs-nonlin.pdf", bbox_inches='tight')

#------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7,7))
k , P = np.loadtxt('./text-files/mps-lin.txt', unpack = True)
k_nl , P_nl = np.loadtxt('./text-files/mps-nonlin.txt', unpack = True)
plt.loglog(k, P, label='Linear')
plt.loglog(k_nl, P_nl,  label='Non-linear')
plt.xlabel('$k\;[h/Mpc]$', fontsize = fs)
plt.ylabel('$P(k)\;[Mpc/h]^3$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/mps-lin-vs-nonlin.pdf", bbox_inches='tight')
plt.show()
