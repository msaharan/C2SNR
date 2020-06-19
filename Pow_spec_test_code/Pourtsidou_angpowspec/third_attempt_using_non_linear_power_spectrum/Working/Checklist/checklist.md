# Note
- The angular power integration in `integrate.quad` goes from `0.0001` to `redshift`

# To-Do checklist
- Use cosmological parameters from P14. 

- Fix the units. Multiply `PK.P(k,z)` with h^3. Fixed in 

    ```
    $pot/Meeting_with_girish/angpowspec_only_with_offline_camb_meeting_with_girish_13_June.py
    ```
- Compare: Integration over redshift and distance

- Implement the changes to the full plot

- Find sigma8 using CAMB for Planck 2013 parameters

    ```
    $pot/Test/sigma8.py
    0.86605
    Planck 2013 - 0.829
    ```

# Extras
- Eq. 20 in P14 should have 1/a^2 in the integrand. 

    $pot/Checklist/Plots/angpowspec_with_a2_correction_in_integrand.pdf

    ![C2SNR/Pow_spec_test_code/Pourtsidou_angpowspec/third_attempt_using_non_linear_power_spectrum/Working/Checklist/Plots/angpowspec_with_a2_correction_in_integrand.pdf](/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Pourtsidou_angpowspec/third_attempt_using_non_linear_power_spectrum/Working/Checklist/Plots/angpowspec_with_a2_correction_in_integrand.png)



