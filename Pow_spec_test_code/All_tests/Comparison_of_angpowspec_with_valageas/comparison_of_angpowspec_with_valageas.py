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
# Constant factor; integration over distance
###############################################################################
def d_constantfactor(redshift):
    return 2 * 3.14 * (9/4) * (H0/c)**4 * omegam0**2 
#------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def z_constantfactor(redshift):
   return 2 * 3.14 * (9/4) * (H0/c)**3 * omegam0**2 
#------------------------------------------------------------------------------

###############################################################################
## Ang dia distance between z = 0 and some redshift
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
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
## Integration over distance
###############################################################################
def d_angpowspec_integrand(dist, ell, dist_s, constf):
    z = np.interp(dist, dist_red, red)
    return constf * (1 - dist/dist_s)**2 * (1 + z)**2 * PK.P(z, ell * (1 + z)/dist)

def d_angpowspec_integration(ell, redshift):
    constf = d_constantfactor(redshift)
    dist_s = c_distance(redshift)
    return integrate.quad(d_angpowspec_integrand, 0.00001, dist_s, args = (ell, dist_s, constf))[0]
#------------------------------------------------------------------------------

###############################################################################
## Integration over redshift
###############################################################################
def z_angpowspec_integrand(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * (1 + z)**2 * PK.P(z, ell/dist)/ hubble_ratio(z)

def z_angpowspec_integration(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand, 0.00001, redshift, args = (ell, dist_s, constf))[0]
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################

omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 67.3 * 1000
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_min = 10
l_max = 100000
redshift = 1

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k=np.exp(np.log(10)*np.linspace(-6,1,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 60, k_hunit = False, hubble_units = False)
#------------------------------------------------------------------------------

###############################################################################
# Distance vs redshift file (See d_angpowspec_integrand())
# comoving distance
###############################################################################
red, dist_red = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/Distances/comov_dist_vs_z.txt", unpack = True)
#------------------------------------------------------------------------------

plt.subplots()

###############################################################################
# To store the data
###############################################################################
d_file = open("./Text_files/angpowspec_integration_over_distance.txt", 'w')
z_file = open("./Text_files/angpowspec_integration_over_redshift.txt", 'w')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
for L in tqdm(range(10, l_max)):
    d_file.write('{}    {}\n'.format(L, d_angpowspec_integration(L, redshift)))
    z_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, redshift)))

d_file.close()
z_file.close()

L , d_CL = np.loadtxt('./Text_files/angpowspec_integration_over_distance.txt', unpack = True)
L , z_CL = np.loadtxt('./Text_files/angpowspec_integration_over_redshift.txt', unpack = True)

plt.plot(L, 2 * 3.14 * L**2 * d_CL, color='red', label='Distance')
plt.plot(L, 2 * 3.14 * L**2 * z_CL, color='blue', label='Redshift')

plt.xlabel('L')
plt.ylabel(r'$2 \pi \; L^2\; C_{L}$')
plt.suptitle("Angular Power Spectrum (z_source = 1)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_min, l_max)
plt.legend()
plt.savefig("./Plots/angpowspec_integration_over_distancs_vs_redshift.pdf")
plt.show()
#------------------------------------------------------------------------------
