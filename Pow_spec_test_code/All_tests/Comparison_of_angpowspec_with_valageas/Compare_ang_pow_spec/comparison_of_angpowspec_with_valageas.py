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

def d_angpowspec_integrand_nl(dist, ell, dist_s, constf):
    z = np.interp(dist, dist_red, red)
    return constf * (1 - dist/dist_s)**2 * (1 + z)**2 * PK_nl.P(z, ell * (1 + z)/dist)

def d_angpowspec_integration_nl(ell, redshift):
    constf = d_constantfactor(redshift)
    dist_s = c_distance(redshift)
    return integrate.quad(d_angpowspec_integrand_nl, 0.00001, dist_s, args = (ell, dist_s, constf))[0]
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

def z_angpowspec_integrand_nl(z, ell, dist_s, constf):
    dist = a_distance(z)
    return constf * (1 - dist/dist_s)**2 * (1 + z)**2 * PK_nl.P(z, ell/dist)/ hubble_ratio(z)

def z_angpowspec_integration_nl(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand_nl, 0.00001, redshift, args = (ell, dist_s, constf))[0]
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################

# Valageas
omegam0 = 0.238
omegal = 0.762
H0 = 73.2 * 1000
h = 0.732
c = 3 * 10**8

cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_min = 10
l_max = 5172 # max value in Valageas's plot
redshift = 1

pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0 = 73.2, omch2=0.1140, ombh2=0.02415)
pars.DarkEnergy.set_params(w=-1.0)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.958, As = 2.35e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-6,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = 4, k_hunit = True, hubble_units = True)


l_max_nl = 120000
pars_nl = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 73.2, omch2=0.1140, ombh2=0.02415)
pars_nl.DarkEnergy.set_params(w=-1.0)
pars_nl.set_for_lmax(lmax = l_max_nl)
pars_nl.InitPower.set_params(ns=0.958, As = 2.35e-09)
results = camb.get_background(pars_nl)
k_nl = 10**np.linspace(-6,2,1000)
PK_nl = get_matter_power_interpolator(pars_nl, nonlinear=True, kmax = 75, k_hunit = True, hubble_units = True)
#------------------------------------------------------------------------------

###############################################################################
# Read data files
###############################################################################
# Variation of comoving distance with redshift
red, dist_red = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/Distances/comov_dist_vs_z.txt", unpack = True)

# Plot from Valageas
x, y= np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/Valageas/linear_z_1.txt", unpack = True)

x_nl, y_nl= np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/Valageas/nonlinear_z_1.txt", unpack = True)
#------------------------------------------------------------------------------

plt.subplots()

###############################################################################
# To store the data
###############################################################################
d_file = open("./Text_files/linear_distance.txt", 'w')
z_file = open("./Text_files/linear_redshift.txt", 'w')

d_file_nl = open("./Text_files/nonlinear_distance.txt", 'w')
z_file_nl = open("./Text_files/nonlinear_redshift.txt", 'w')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
for L in tqdm(range(10, l_max)):
    d_file.write('{}    {}\n'.format(L, d_angpowspec_integration(L, redshift)))
    z_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, redshift)))

for L in tqdm(range(10, l_max_nl)):
    d_file_nl.write('{}    {}\n'.format(L, d_angpowspec_integration_nl(L, redshift)))
    z_file_nl.write('{}    {}\n'.format(L, z_angpowspec_integration_nl(L, redshift)))

d_file.close()
z_file.close()

d_file_nl.close()
z_file_nl.close()

L , d_CL = np.loadtxt('./Text_files/linear_distance.txt', unpack = True)
L , z_CL = np.loadtxt('./Text_files/linear_redshift.txt', unpack = True)

L_nl , d_CL_nl = np.loadtxt('./Text_files/nonlinear_distance.txt', unpack = True)
L_nl , z_CL_nl = np.loadtxt('./Text_files/nonlinear_redshift.txt', unpack = True)


plt.plot(L, 2 * 3.14 * L**2 * d_CL, label='L Distance')
plt.plot(L, 2 * 3.14 * L**2 * z_CL, label='L Redshift')

plt.plot(L_nl, 2 * 3.14 * L_nl**2 * d_CL_nl,  label='NL Distance')
plt.plot(L_nl, 2 * 3.14 * L_nl**2 * z_CL_nl,  label='NL Redshift')

plt.plot(x, y, label = "L Valageas et al.")
plt.plot(x_nl, y_nl, label = "NL Valageas et al.")
plt.xlabel('L')
plt.ylabel(r'$2 \pi \; L^2\; C_{L}$')
plt.suptitle("z = 1")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_min, x_nl.max())
plt.legend()
plt.savefig("./Plots/angpowspec_integration_over_distancs_vs_redshift.pdf")
plt.show()
#------------------------------------------------------------------------------
