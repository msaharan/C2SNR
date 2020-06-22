import numpy as np
import cosmolopy.distance as cd
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
    return 9 * (H0/c)**4 * omegam0**2 
#------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def z_constantfactor(redshift):
    return 9 * (H0/c)**3 * omegam0**2 
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
    # find z corresponding to dist
    z = np.interp(dist, dist_red, red)
    # Assume D is angular diameter distance and Chi is comoving distance
    # ell/D becomes ell/(a * Chi)
    # dist is comoving distance
#    return constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell * (1 + z)/dist)
    return constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist)

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
    return constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist)/ hubble_ratio(z)

def z_angpowspec_integration(ell, redshift):
    constf = z_constantfactor(redshift)
    dist_s = a_distance(redshift)
    return integrate.quad(z_angpowspec_integrand, 0.00001, redshift, args = (ell, dist_s, constf))[0]
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
redshift = 2
l_max = l_plot_upp_limit

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-6,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------

###############################################################################
# Distance vs redshift file (See d_angpowspec_integrand())
# comoving distance
###############################################################################
red, dist_red = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/Distances/ang_dia_dist_vs_z.txt", unpack = True)
#------------------------------------------------------------------------------

plt.subplots()

###############################################################################
# Plot from Pourtsidou et al. 2014
###############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)

plt.plot(x_2,y_2,color='black', label='Pourtsidou et al. 2014')
#------------------------------------------------------------------------------

###############################################################################
# To store the data
###############################################################################
d_file = open("./Text_files/angpowspec_integration_over_ang_dia_distance.txt", 'w')
z_file = open("./Text_files/angpowspec_integration_over_ang_dia_redshift.txt", 'w')
#------------------------------------------------------------------------------

###############################################################################
# Plot from this work
###############################################################################
for L in tqdm(range(10, int(l_plot_upp_limit))):
    d_file.write('{}    {}\n'.format(L, d_angpowspec_integration(L, redshift)/ (2 * 3.14)))
    z_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, redshift)/ (2 * 3.14)))

d_file.close()
z_file.close()

L , d_CL = np.loadtxt('./Text_files/angpowspec_integration_over_ang_dia_distance.txt', unpack = True)
L , z_CL = np.loadtxt('./Text_files/angpowspec_integration_over_ang_dia_redshift.txt', unpack = True)

plt.plot(L, d_CL, color='red', label='This work (integ. over dist.)')
plt.plot(L, z_CL, color='blue', label='This work (integ. over z)')

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Angular Power Spectrum (z_source = 2)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
plt.ylim(1E-10,1E-7)
plt.savefig("./Plots/angpowspec_integration_over_ang_dia_distancs_vs_redshift.pdf")
plt.show()
#------------------------------------------------------------------------------

