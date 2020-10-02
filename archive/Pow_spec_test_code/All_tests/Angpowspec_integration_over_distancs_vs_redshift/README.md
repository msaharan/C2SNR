# Converting distance integral into redshift integral and vice versa 
[Link](https://github.com/msaharan/C2SNR/blob/master/Pow_spec_test_code/All_tests/Angpowspec_integration_over_distancs_vs_redshift/References/Integral_conversion/intergral_conversion.pdf)

## UPDATE 2 Oct 2020
This code uses `angular_diameter_distance`. I and Girish discussed and decided that we should not use this quantity in our calculations.


## Notes about parts of the code that did/could cause errors
### The issue with the units of k and P(z, k)
This issue was resolved by adding `k_hunit = False` and `hubble_units = False`
arguments in `PK`. [Source](https://camb.readthedocs.io/en/latest/camb.html?highlight=get_matter_power_interpolator#camb.get_matter_power_interpolator)

* `k_hunit` - if true, matter power is a function of k/h, if false, just k (both Mpc^(-1) units)
* `hubble_units` - if true, output power spectrum in (Mpc/h)^3 units, otherwise Mpc^3.


```
k = 10**np.linspace(-6,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = 20, k_hunit = False, hubble_units = False)

```
### In the function `d_angpowspec_integrand(dist, ell, dist_s, constf)`

```
z = np.interp(dist, dist_red, red)
return constf * (1 - dist/dist_s)**2 * PK.P(z, ell * (1 + z)/ dist)
```

* I made sure that `dist_red` is the comoving distance. 
* Removed `h ** 3` from the return statement. 
* I made sure that doing `(1 + z)/ dist` is the right thing to do. 
 
### In the function `def d_angpowspec_integration(ell, redshift)`
* `dist_s = c_distance(redshift)` is correct. It should be comoving distance.

### In the CAMB section

* Made sure that `get_matter_power_interpolator` is working as expected. See
  [link](https://github.com/msaharan/C2SNR/tree/master/Pow_spec_test_code/All_tests/Find_the_pow_spec_at_k_z_using_interpol).
* Resolved the issue of `h ** 3` as discussed at the of this file. This is what
  we are going to use:
```
k = 10**np.linspace(-6,1,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = some_number, k_hunit = False, hubble_units = False)
```
* Made sure that 21cm arguments are not causing any problems. [Code](https://github.com/msaharan/C2SNR/tree/master/Pow_spec_test_code/All_tests/Camb_21cm_true_vs_false).

