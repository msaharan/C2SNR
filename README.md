# C2SNR

This file shows the general structure of the code to avoid the repetitive
comments.
 
# In the code related to the angular power spectrum
```
###############################################################################
# CAMB
###############################################################################
# Setting Do21cm and transder_21cm_cl to False gives the same result. Evidence
# in this directory: C2SNR/Pow_spec_test_code/All_tests/Camb_21cm_true_vs_false/
# I am working with the 21cm signal that's why I included them in the
# parameters.  I don't understand the functioning of Do21cm and transder_21cm_cl
# yet.
#
# Choose the power spectrum like this: Linear: NonLinear = 0 Non-linear Matter
# Power (HALOFIT): NonLinear = 1 Non-linear CMB Lensing (HALOFIT): NonLinear = 2
# Non-linear Matter Power and CMB Lensing (HALOFIT): NonLinear = 3
#
# Refer to the following URL to see the defaults of HaloFit:
# https://camb.readthedocs.io/en/latest/_modules/camb/nonlinear.html?highlight=halofit_default#
#
# Toggle transfer function with WantTransfer
#
# The cosmological parameters have been taken from Planck 2013

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3,
omch2=0.1199, ombh2=0.02205, YHe = 0.24770, Do21cm = True, transfer_21cm_cl =
True) pars.DarkEnergy.set_params(w=-1.13)

# Maximum ell for which the power spectrum should be calculated

pars.set_for_lmax(lmax=2500) pars.InitPower.set_params(ns=0.9603, As =
2.196e-09) results = camb.get_background(pars)

# Change the following values according to the chosen value of lmax
# np.linspace(-6, this_value, 1000) np.log(10)*np.linspace(-6, this_value,1000)
# and kmax = this_value Foe example, at z = 1 and lmax = 2500 we get kmax ~ 1.4.
# Therefore we can set np.linspace(-6, 1, 1000) and kmax = 2 To calculate the
# kmax for any value of lmax, see
# C2SNR/Pow_spec_test_code/Frequently_used_snippets/Find_kmax_corresponding_to_lmax_for_camb_calculations/ 

# Calculate power spectrum for these k values

k = 10**np.linspace(-6,1,1000))

# Read more about camb.get_matter_power_interpolator at:
# https://camb.readthedocs.io/en/latest/camb.html?highlight=PK.P#camb.get_matter_power_interpolator

PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 2)
#------------------------------------------------------------------------------
```


