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
## Comoving distance between z = 0 and some z
###############################################################################
def a_distance(var):
    return cd.comoving_distance(var, **cosmo)
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
    pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0= h * 100, omch2= omch2, ombh2= ombh2, YHe = YHe)
    pars.DarkEnergy.set_params(w=-1.13)
    pars.set_for_lmax(lmax = l_max)
    pars.InitPower.set_params(ns=ns, As = 2.196e-09)
    results = camb.get_background(pars)
    PK = get_matter_power_interpolator(pars, nonlinear=False, kmax =  k_max(z, l_max), k_hunit = True, hubble_units = True) 
    return PK.P(z, k)

# Non-linear
def nonlin_ps(z, l_max):
    pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=h * 100, omch2=omch2, ombh2=ombh2, YHe = YHe)
    pars_nl.DarkEnergy.set_params(w=-1.13)
    pars_nl.set_for_lmax(lmax = l_max)
    pars_nl.InitPower.set_params(ns=ns, As = 2.196e-09)
    results = camb.get_background(pars_nl)
    PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = k_max(z, l_max), k_hunit = True, hubble_units = True)
    return PK_nl.P(z, k)
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
omegam0 = 0.308
omegal = 0.692
ombh2 = 0.02214
omch2 = 0.1187
YHe = 0.24
ns = 0.961
fs = 14
mfs = 12

c = 3 * 10**8
h = 0.678
H0 = 100 * h * 1000 # h m s**(-1) / Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_plot_low_limit = 10
l_plot_upp_limit = 1000
#l_plot_upp_limit = 30000
redshift = 6.349
l_max = l_plot_upp_limit

z_array = np.arange(5.0, 9.0, 1.0)
#------------------------------------------------------------------------------


###############################################################################
# Store the data into text files
###############################################################################
for z in z_array:
    k = np.linspace(- 5, k_max(redshift, l_max), 1000)
    p = lin_ps(z, l_max)
    p_nl = nonlin_ps(z, l_max)
    z_file = open("./text-files/mps-lin-z-{}.txt".format(z), 'w')
    z_file_nl = open("./text-files/mps-nonlin-z-{}.txt".format(z), 'w')
    
    for i in tqdm(range(np.size(k))):
        z_file.write('{}   {}\n'.format(k[i], p[i]))
        z_file_nl.write('{}   {}\n'.format(k[i], p_nl[i]))
    z_file.close()
    z_file_nl.close()
#------------------------------------------------------------------------------

###############################################################################
# Read the text files and plot
###############################################################################
# Comment everything before this section if you just want to plot the previously
# generated data.

fig, ax = plt.subplots(figsize=(7,7))
for z in z_array:
    kx , py = np.loadtxt('./text-files/mps-lin-z-{}.txt'.format(z), unpack = True)
plt.loglog(kx, py, label='z = {}'.format(z))

plt.xlabel('k [h/Mpc]', fontsize = fs)
plt.ylabel('$P(k) [Mpc/h]^3$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/mps-lin.pdf", bbox_inches='tight')
# plt.show()

fig, ax = plt.subplots(figsize=(7,7))
for z in z_array:
    k_nl , p_nl = np.loadtxt('./text-files/mps-nonlin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(k_nl, p_nl,  label='z = {}'.format(z))
    
plt.xlabel('k [h/Mpc]', fontsize = fs)
plt.ylabel('$P(k) [Mpc/h]^3$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/mps-nonlin.pdf", bbox_inches='tight')
plt.show()
"""
fig, ax = plt.subplots(figsize=(6,6))
for z in z_array:
    kx , py = np.loadtxt('./text-files/mps-lin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(kx, py, label='z = {}'.format(z))

plt.xlabel('$k \; [h/Mpc]$', fontsize = fs)
plt.ylabel('$P(k) \; [Mpc/h]^3$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/mps-lin.pdf", bbox_inches='tight')
# plt.show()

fig, ax = plt.subplots(figsize=(6,6))
for z in z_array:
    k_nl , p_nl = np.loadtxt('./text-files/mps-nonlin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(k_nl, p_nl,  label='z = {}'.format(z))

plt.xlabel('$k \; [h/Mpc]$', fontsize = fs)
plt.ylabel('$P(k)\; [Mpc/h]^3$', fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)
plt.legend(fontsize = fs)
plt.savefig("./plots/mps-nonlin.pdf", bbox_inches='tight')
plt.show()
"""
#------------------------------------------------------------------------------
