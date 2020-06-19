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

###############################################################################
# Constant factor in the expression of angular power spectrum
###############################################################################
def constantfactor(redshift):
    return 9 * (H0/c)**3 * omegam0**2 
#------------------------------------------------------------------------------

###############################################################################
## Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

###############################################################################
## To calculate the angular power spectrum using the linear matter power spectrum
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constf):
    dist = distance(z)
    return constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist)/ hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf))[0]
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
redshift = 2
l_max = l_plot_upp_limit

'''
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.02205, omch2=0.1199)
pars.InitPower.set_params(As=2.196e-9, ns=0.9624)
pars.set_for_lmax(2500, lens_potential_accuracy=1)
results = camb.get_background(pars)

k=np.exp(np.log(10)*np.linspace(-6,3,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
'''

nz = 100
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k=np.exp(np.log(10)*np.linspace(-6,3,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 30)

#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
chi_s = np.zeros(11)
L_array = np.arange(0, l_plot_upp_limit)
angpowspec_without_j = np.zeros(int(l_max))
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
# Comoving distance between z = 0 and z = source redshift
for L in tqdm(range(10, int(l_plot_upp_limit))):
    angpowspec_without_j[L] = angpowspec_integration_without_j(L, redshift)/ (2 * 3.14)

plt.plot(L_array[l_plot_low_limit:l_plot_upp_limit], angpowspec_without_j[l_plot_low_limit:l_plot_upp_limit], color='blue', label='This work (Non-linear PS)'.format(redshift))

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Angular Power Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
plt.ylim(1E-10,1E-7)
plt.savefig("./Plots/angpowspec_only_with_offline_camb_non_linear_powspec_test.pdf")
plt.show()

#------------------------------------------------------------------------------
