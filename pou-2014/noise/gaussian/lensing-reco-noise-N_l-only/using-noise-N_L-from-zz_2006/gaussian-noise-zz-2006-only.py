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
import p_2014_parameters as exp
import p_2014_cosmology as cos 

########################################################################
## Job specific parameters
########################################################################

### angular binning; max and min Fourier mode
l_ll = 10
l_ul = 50000

### Discretization of the observed volume along the line of site
#-- j=0 mode will be made useless due to foreground removal.
#-- Stated at the end of section 2 in P_2014.
j_min = 1 
j_max = 2

### source redshift
z_s = 2

### plot params
l_plot_min = l_ll
l_plot_max = 600
err_stepsize = exp.delta_l

#------------------------------------------------------------------------

########################################################################
## Comoving distance between z = 0 and some redshift
########################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------

########################################################################
## hubble ratio H(z)/H0
########################################################################
def hubble_ratio(var):
    return (omegam * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------

########################################################################
## angular power spectrum
########################################################################
def C_l_integrand(z, ell):
    dist = distance(z)
    return constfac * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) / hubble_ratio(z)

def C_l(ell):
    return integrate.quad(C_l_integrand, 0.0001, z_s, args = (ell), limit=20000)[0]
#------------------------------------------------------------------------

########################################################################
## constants
########################################################################

c = 3 * 10**8

### cosmology
h = cos.h
H0 = h * 100 * 1000
omegal = cos.omegal
omegam = cos.omegam
omegab = cos.omegab
omegac = cos.omegac
omegamh2 = omegam * h**2
omegabh2 = omegab * h**2
omegach2 = omegac * h**2
YHe = cos.YHe
As = cos.As
ns = cos.ns
cosmo = {'omega_M_0': omegam, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}

### constant factor ouside the integral in equation 20 of P_2014
constfac = (9/4) * (H0/c)**3 * omegam**2 

### distance between source and observer
dist_s = distance(z_s)
#-----------------------------------------------------------------------

########################################################################
## Arrays
########################################################################

### angular power spectrum 
#-- signal
C_l_array = np.zeros(int(2**0.5 * l_ul))
#------------------------------------------------------------------------

########################################################################
## CAMB
########################################################################
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2 = omegach2, ombh2 = omegabh2, YHe = YHe)
pars.DarkEnergy.set_params(w = -1)
pars.set_for_lmax(lmax = l_ul)
pars.InitPower.set_params(ns = ns, As = As)
#results = camb.get_background(pars)
results = camb.get_results(pars)
k = 10**np.linspace(-5,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = False, hubble_units = False)

print("SIGMA8")
print(results.get_sigma8())

#-----------------------------------------------------------------------
"""
########################################################################
## Plot from P_2014 (without error bars)
########################################################################
p_l = np.zeros(600)
p_c_l = np.zeros(600)
p_l, p_c_l, junk, junk, junk, junk = np.loadtxt("/mnt/storage/pdata/Utility/Git/dyskun/C2SNR/archive/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)
plt.loglog(p_l,p_c_l, label='Pourtsidou et al. 2014 (z=2)')
#-----------------------------------------------------------------------
"""

########################################################################
## write data
########################################################################
C_l_data = open('./text-files/c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max - 1), 'w')

### C_l
for L in tqdm(range(l_plot_min, l_plot_max)):
    C_l_data.write('{}    {}\n'.format(L, C_l(L)))
C_l_data.close()

#-----------------------------------------------------------------------

########################################################################
## write data
########################################################################
### canvas
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

### read data
#-- 'i' denotes 'input'
iL, iC_L = np.loadtxt('./text-files/c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max - 1), unpack=True)

p_l = np.zeros(600)
p_c_l = np.zeros(600)
p_l, p_c_l, junk, junk, junk, junk = np.loadtxt("/mnt/storage/pdata/Utility/Git/dyskun/C2SNR/archive/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)
plt.loglog(p_l,p_c_l, label='Pourtsidou et al. 2014 (z=2)')

### plot
plt.plot(iL, iC_L, label = '$C_L$', color = 'black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)
plt.ylabel('$C_L$', fontsize = fs)
plt.legend(fontsize = fs)
plt.xlim(l_plot_min, l_plot_max)
#plt.ylim(4E-10, 3E-8)
plt.savefig('./plots/c_l_z_{}_lmax_{}_j_max_{}.pdf'.format(z_s, l_ul, j_max - 1), bbox_inches='tight')
plt.show()

