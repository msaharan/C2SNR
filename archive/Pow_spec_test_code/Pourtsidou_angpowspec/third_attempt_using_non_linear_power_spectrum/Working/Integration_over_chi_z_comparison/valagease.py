import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

import numpy as np

###############################################################################
## Ang dia distance between z = 0 and some redshift
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
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
def z_angpowspec_integrand(z, ell, dist_s):
    dist = a_distance(z)
    return 2 * 3.14 * (c / H0) *  (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist)/ hubble_ratio(z)

def z_angpowspec_integration(ell, redshift):
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand, 0.0001, redshift, args = (ell, dist_s), limit = 2000)[0]
#------------------------------------------------------------------------------

###############################################################################
# Constants
###############################################################################

omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 67.3 * 1000
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}
l_plot_low_limit = 10
l_plot_upp_limit = 550
two_pi_squared = 2 * 3.14 * 2 * 3.14
eta = 0
eta_D2_L = 5.94 * 10**12 / 10**eta
redshift = 1
l_max = l_plot_upp_limit

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k=np.exp(np.log(10)*np.linspace(-6,1,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################

chi_s = np.zeros(11)
L_array = np.arange(0, l_plot_upp_limit)
z_angpowspec = np.zeros(int(l_max))
#------------------------------------------------------------------------------

plt.subplots()

###############################################################################
# Plot from Pourtsidou et al. 2014 (with error bars)
###############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)

plt.plot(x_2,y_2,color='black', label='Pourtsidou et al. 2014 (z=2)')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
for L in tqdm(range(10, int(l_plot_upp_limit))):
    z_angpowspec[L] = z_angpowspec_integration(L, redshift)/ (2 * 3.14)


plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], z_angpowspec[l_plot_low_limit:l_plot_upp_limit], color='red', label='This work IoR'.format(redshift))

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Angular Power Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
#plt.ylim(1E-10,1E-7)
plt.savefig("./Plots/angpowspec_comparison_distance_redshift_test.pdf")
plt.show()

#------------------------------------------------------------------------------
