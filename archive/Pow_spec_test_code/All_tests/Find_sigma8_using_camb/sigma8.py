import numpy as np
import cosmolopy.distance as cd
import scipy.integrate as integrate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

# Planck 2013
pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=2500)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
pars.set_matter_power(redshifts=[0.], kmax=2.0)
results = camb.get_results(pars)
print('Without 21 cm: {}'.format(results.get_sigma8()))

pars_21 = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl = True)
pars_21.DarkEnergy.set_params(w=-1.13)
pars_21.set_for_lmax(lmax=2500)
pars_21.InitPower.set_params(ns=0.9603, As = 2.196e-09)
pars_21.set_matter_power(redshifts=[0.], kmax=2.0)
results_21 = camb.get_results(pars_21)
print('With 21 cm: {}'.format(results_21.get_sigma8()))


