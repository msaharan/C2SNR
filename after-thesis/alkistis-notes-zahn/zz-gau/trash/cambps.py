import mylibraries

###############################################################################
## CAMB parameters
###############################################################################

#-- The parameters other than omega(m/c/b) anh h have not been taken from ZZ 2006. Do not use these
#-- in the final calculations. Make sure to use the correct values. 
#-- TODO for Mohit: Try to find out if changing ns, As and YHe causes major change
#-- in the results. 
#-- From the previous experience with these numbers, I don't think
#-- that there will be big change in the results so I am proceeding with these values.
#--
#-- k_unit = True -> input of k will be in h/Mpc units
#-- hubble_units = True -> output power spectrum PK.P(k, z) will be in Mpc/h units.
#--
#-- This PK and k business is very confusing because ZZ_2006 or P_2014 might have used
#-- k and PK with or without h in the units. I once tried to find it out but my tests were poorly 
#-- structured so I couldn't quite find out which one they have used. I was trying 
#-- different h, true, false combinations to see which combination gives the best match 
#-- with the plot from P_2014.
#-- See the tests in: 
#-- C2SNR/tree/master/archive/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou
#-- or if you can't find them there, check the permanent location:
#-- C2SNR/tree/before_archive_20201002/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou

pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=omegach2, ombh2=omegabh2, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09) # ZZ_2006 took n = 1 and sigma_8 = 0.9
results = camb.get_background(pars)
k = 10**np.linspace(-5,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(k), k_hunit = True, hubble_units = True)

#------------------------ CAMB parameters end --------------------------------

