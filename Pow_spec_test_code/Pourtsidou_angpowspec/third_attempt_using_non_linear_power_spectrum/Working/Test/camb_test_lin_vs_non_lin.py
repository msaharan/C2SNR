import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import camb
from camb import model, initialpower


pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.set_matter_power(redshifts=[0.], kmax=2.0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)


plt.loglog(kh, pk[0,:], color='k')
plt.loglog(kh_nonlin, pk_nonlin[0,:], color='r')

plt.show()

