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

###############################################################################
# Ang dia distance between z = 0 and some redshift
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Constants
###############################################################################
z = 2
l_min = 10
l_max = 1000
omegam0 = 0.315
omegal = 0.685
h = 0.673
omegak0 = 0.0
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': omegak0, 'h': h}
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
p_array = np.zeros(l_max - l_min)
l_array = np.zeros(l_max - l_min)
#------------------------------------------------------------------------------

###############################################################################
# CAMB
###############################################################################
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=
0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=20000)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-6,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 30)
#------------------------------------------------------------------------------

###############################################################################
# Plot
###############################################################################
f = open('./Text_files/powspec_at_k_z.txt', 'w')

for l in range(l_min, l_max):
    f.write('{}   {}\n'.format(l/a_distance(z), PK.P(z, l/a_distance(z))))
f.close()

k_f, p_f = np.loadtxt('./Text_files/powspec_at_k_z.txt', unpack = True)
plt.subplots()
plt.plot(k, PK.P(z, k), label = 'CAMB')
plt.plot(k_f, p_f, label = 'Interpolation')
plt.title('Power spectrum (CAMB)')
plt.xlabel('k/h (1/Mpc)')
plt.ylabel('Power Spectrum P(K) $(Mpc/h)^3$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("./Plots/powspec_at_k_z.pdf")
plt.show()
#------------------------------------------------------------------------------
