import numpy as np
import cosmolopy.distance as cd
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
# This works
# pars1 = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)

# This also works
# pars1 = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True)

# This also works
pars1 = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770)

# All the three options (of pars1) give the same result. 
# See: C2SNR/Pow_spec_test_code/All_tests/Camb_21cm_true_vs_false

pars1.DarkEnergy.set_params(w=-1.13)
pars1.set_for_lmax(lmax=2500)
pars1.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars1)
k = 10**np.linspace(-6,1,1000)
PK1 = get_matter_power_interpolator(pars1, nonlinear=True, kmax = 2)

# This doesn't work
#pars2 = model.CAMBparams(NonLinear = 2, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)

# This also doesn't work
#pars2 = model.CAMBparams(NonLinear = 2, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True)

# This works
pars2 = model.CAMBparams(NonLinear = 2, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770)

pars2.DarkEnergy.set_params(w=-1.13)
pars2.set_for_lmax(lmax=2500)
pars2.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars2)
k = 10**np.linspace(-6,1,1000)
PK2 = get_matter_power_interpolator(pars2, nonlinear=True, kmax = 2)

# This doesn't work
#pars3 = model.CAMBparams(NonLinear = 3, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)

# This also doesn't work
# pars3 = model.CAMBparams(NonLinear = 3, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770, Do21cm = True)

# This works
pars3 = model.CAMBparams(NonLinear = 3, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2= 0.02205, YHe = 0.24770)

pars3.DarkEnergy.set_params(w=-1.13)
pars3.set_for_lmax(lmax=2500)
pars3.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars3)
k = 10**np.linspace(-6,1,1000)
PK3 = get_matter_power_interpolator(pars3, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------

###############################################################################
# Plot
###############################################################################
plt.subplots()
plt.plot(k, PK1.P(z, k) - PK2.P(z, k), label = ('1 - 2'))
plt.plot(k, PK2.P(z, k) - PK3.P(z, k), label = ('1 - 3'))
plt.title('Power spectrum (CAMB)')
plt.xlabel('k/h (1/Mpc)')
plt.ylabel('Power Spectrum P(K) $(Mpc/h)^3$')
plt.legend()
plt.savefig("./Plots/comparison_of_powspec.pdf")
plt.show()
#------------------------------------------------------------------------------
