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

# Planck 2013
pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
pars.set_matter_power(redshifts=[0.], kmax=2.0)

results = camb.get_results(pars)
print(results.get_sigma8())


#k=np.exp(np.log(10)*np.linspace(-4,1,1000))
#PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = 1)
#plt.plot(k, PK.P(0, k))
#plt.show()



"""
# Documentation
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0.], kmax=2.0)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
print(results.get_sigma8())
"""

'''
def distance(var):
    return cd.angular_diameter_distance(var, **cosmo)

omegam0 = 0.315
omegal = 0.685
c = 3 * 10**8
H0 = 67.3 * 1000
h = 0.673

cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': 0.673}

print(40000/distance(2))
'''

