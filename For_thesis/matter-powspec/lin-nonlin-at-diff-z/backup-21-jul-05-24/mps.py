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
## Ang dia distance between z = 0 and some z
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
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
    k = 10**np.linspace(-5, k_lim(z, l_max),1000)
    PK = get_matter_power_interpolator(pars, nonlinear=False, kmax =  k_max(z, l_max), k_hunit = True, hubble_units = True) 
    return PK.P(z, k)

# Non-linear
def nonlin_ps(z, l_max):
    pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
    pars_nl.DarkEnergy.set_params(w=-1.13)
    pars_nl.set_for_lmax(lmax = l_max)
    pars_nl.InitPower.set_params(ns=0.9603, As = 2.196e-09)
    results = camb.get_background(pars_nl)
    k = 10**np.linspace(-5, k_lim(z, l_max),1000)
    PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = k_max(z, l_max), k_hunit = True, hubble_units = True)
    return PK_nl.P(z, k)
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
z_array = np.arange(5.0, 9.0, 1.0)
#------------------------------------------------------------------------------


###############################################################################
# Store the data into text files
###############################################################################
for z in z_array:
    k = 10**np.linspace(-6, k_lim(z, l_max),1000)
    p = lin_ps(z, l_max)
    p_nl = nonlin_ps(z, l_max)
    #p = np.zeros(np.size(k))
    #p_nl = np.zeros(np.size(k))
    #p = PK
    #p_nl = PK_nl
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

plt.subplots()
for z in z_array:
    kx , py = np.loadtxt('./text-files/mps-lin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(kx, py, label='z = {}'.format(z))

plt.xlabel('k [h/Mpc]')
plt.ylabel(r'$P(k) [Mpc/h]^3$')
plt.xlim(1E-5, l_max/a_distance(z_array[-1]))
plt.legend()
plt.savefig("./plots/mps-lin.pdf")
# plt.show()

plt.subplots()
for z in z_array:
    k_nl , p_nl = np.loadtxt('./text-files/mps-nonlin-z-{}.txt'.format(z), unpack = True)
    plt.loglog(k_nl, p_nl,  label='z = {}'.format(z))
    
plt.xlabel('k [h/Mpc]')
plt.ylabel(r'$P(k) [Mpc/h]^3$')
plt.xlim(1E-5, l_max/a_distance(z_array[-1]))
plt.legend()
plt.savefig("./plots/mps-nonlin.pdf")
# plt.show()
#------------------------------------------------------------------------------
