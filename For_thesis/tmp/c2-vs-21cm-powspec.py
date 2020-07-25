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

def Tz(redshift):
    return 0.0596 * (1+redshift)**2/np.sqrt(omegam0 * (1+redshift)**3 + omegal)

def distance(var):
    return cd.comoving_distance(var, **cosmo)

redshift = 6

plt.subplots()

###############################################################################
# 21 cm
###############################################################################
omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 100 *  1000 # h m s**(-1) / Mpc units
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}
k_max = 10
l_max = int(k_max * distance(redshift))
k_lim = 1
k_min = -2
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(k_min,k_lim, 1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = k_max, k_hunit = True, hubble_units = True)

p = Tz(redshift)**2 * PK.P(redshift, k)
plt.plot(k, p, label = '21 cm')
plt.plot(k, PK.P(redshift, k), label = 'Ma')
#------------------------------------------------------------------------------

###############################################################################
# CII
###############################################################################
pk3_2pi_file, k_file = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6.txt', unpack=True)
p_file = pk3_2pi_file * 2 * 3.14**2 / k_file**3

plt.plot(k_file, p_file, label = 'CII')
plt.plot(k_file, pk3_2pi_file, label = r'CII$\times k^3 / 2 \pi^2$', color = 'black')
#------------------------------------------------------------------------------

plt.legend()
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k) $[Mpc/h]^3$')
plt.xscale('log')
plt.yscale('log')
plt.savefig("./plots/c2-vs-21cm-powspec.pdf")
#plt.savefig("./plots/c2-vs-21cm-powspec_in_h_mpc_2.pdf")
plt.show()
