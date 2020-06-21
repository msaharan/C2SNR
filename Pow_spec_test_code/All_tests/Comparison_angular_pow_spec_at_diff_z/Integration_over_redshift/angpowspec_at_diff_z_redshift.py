import numpy as np                                                 
import cosmolopy.distance as cd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

###############################################################################
# Constant factor; Integration over z
###############################################################################
def z_constantfactor(z):
    return 9 * (H0/c)**3 * omegam0**2
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
    return constf * (1 - dist/dist_s)**2 * h**3 * PK.P(z, ell/dist)/ hubble_ratio(z)

def z_angpowspec_integration(ell, z):
    constf = z_constantfactor(z)
    dist_s = a_distance(z)
    # integrand blows up at z = 0
    return integrate.quad(z_angpowspec_integrand, 0.00001, z, args = (ell, dist_s,
 constf))[0]
#------------------------------------------------------------------------------

###############################################################################
# Constants
###############################################################################
# Compare the ang. pow. spec. at these redshifts
# Don's include 0.0 in this array

redshift_array = np.arange(0.5, 2.5, 0.5)

# Calculate the  ang. power spec. for these ells
# Don't set l_min = 0

l_min = 10
l_max = 1000

# Set cosmology 
# The cosmological parameters have been taken from Planck 2013

H0 = 67.3 * 1000
omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
h = 0.673
omegak0 = 0.0
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': omegak0, 'h': h}
#------------------------------------------------------------------------------

###############################################################################
# CAMB
###############################################################################
# Setting Do21cm and transder_21cm_cl to False gives the same result. Evidence in this directory: C2SNR/Pow_spec_test_code/All_tests/Camb_21cm_true_vs_false/ 
# I am working with the 21cm signal that's why I included them in the parameters. 
# I don't understand the functioning of Do21cm and transder_21cm_cl yet.
#
# Choose the power spectrum like this:
# Linear: NonLinear = 0
# Non-linear Matter Power (HALOFIT): NonLinear = 1
# Non-linear CMB Lensing (HALOFIT): NonLinear = 2
# Non-linear Matter Power and CMB Lensing (HALOFIT): NonLinear = 3
#
# Refer to the following URL to see the defaults of HaloFit: https://camb.readthedocs.io/en/latest/_modules/camb/nonlinear.html?highlight=halofit_default#
#
# Toggle transfer function with WantTransfer
#
# The cosmological parameters have been taken from Planck 2013

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)

# Maximum ell for which the power spectrum should be calculated

pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)

# Change the following values according to the chosen value of lmax
# np.linspace(-6, this_value, 1000)
# np.log(10)*np.linspace(-6, this_value,1000)
# and
# kmax = this_value
# Foe example, at z = 1 and lmax = 2500 we get kmax ~ 1.4. Therefore we can set np.linspace(-6, 1, 1000) and kmax = 2
# To calculate the kmax for any value of lmax, see C2SNR/Pow_spec_test_code/Frequently_used_snippets/Find_kmax_corresponding_to_lmax_for_camb_calculations/ 

k=np.exp(np.log(10)*np.linspace(-6,1,1000))
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------

###############################################################################
# Plot 
###############################################################################
for z in redshift_array:
    print("z = {}".format(z))
    z_file = open("./Text_files/integration_over_redshift_z_{}.txt".format(z), 'w')

    for L in tqdm(range(l_min, l_max)):
        z_file.write('{}    {}\n'.format(L, z_angpowspec_integration(L, z)/ (2 * 3.14)))

    z_file.close()
    L , z_CL = np.loadtxt('./Text_files/integration_over_redshift_z_{}.txt'.format(z), unpack = True)
    plt.plot(L, z_CL, label='z = {}'.format(z))

plt.xlabel('L')
plt.ylabel(r'$C_{L} L(L+1)/2\pi$')
plt.suptitle("Variation of Ang. Pow. Spec. with z")
plt.xscale("log")
plt.yscale("log")
plt.xlim(l_min, l_max)
plt.legend()
#plt.ylim(1E-10,1E-7)
plt.savefig("./Plots/integration_ove_redshift.pdf")
plt.show()
#------------------------------------------------------------------------------
