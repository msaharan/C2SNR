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
# Constant factor; Integration over redshift
###############################################################################
def z_constantfactor(redshift):
    return (9/4) * (H0/c)**3 * omegam0**2 
#------------------------------------------------------------------------------

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
def z_angpowspec_integrand(z, ell, dist_s, constf):
    dist = a_distance(z)
    return 2 * 3.14 * constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

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

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k=np.exp(np.log(10)*np.linspace(-6,1,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
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
z_file = open("./Text_files/Comparison_of_angpowspec_with_valageas.txt", 'w')
#------------------------------------------------------------------------------

###############################################################################
# Plot 
###############################################################################
for L in tqdm(range(10, int(l_plot_upp_limit))):
    z_file.write('{}    {}\n'.format(L, 2 * 3.14 * z_angpowspec_integration(L, redshift)))

z_file.close()

L , z_CL = np.loadtxt('./Text_files/Comparison_of_angpowspec_with_valageas.txt', unpack = True)

plt.plot(L, z_CL, color='blue')

plt.xlabel('L')
plt.ylabel(r'$2 \pi ell^2 P_{L} $')
plt.suptitle("Angular Power Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.legend()
#plt.ylim(1E-10,1E-7)
plt.savefig("./Plots/Comparison_of_angpowspec_with_valageas.pdf")
plt.show()
#------------------------------------------------------------------------------

