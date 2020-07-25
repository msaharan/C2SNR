# Test 1
* Divide `PK.P(k, z)` by `h**3`
* `hubble_units = False`
* `k_hunit = False`

# Test 2
* Same as Test 1
* Divide `k` by `h**3`

# Test 3
* Same as Test 1
* Multiply `k` by `h**3`

# Test 4
* No division or multiplication with `h**3`
* `hubble_units = True` 
    * Output of `PK.P(k, z)` will be in MPc/h units
* `k_hunit = False`
    * Input of `k` will be in Mpc units
* The `h` factors of `H0` and `PK.P(k, z)` should cancel.

# Test 5
* Same as Test 4
* `k_hunit = True`

# Test 6
* Same as Test 5
* Multiply `k` with `h`

# Test 7
* Same as Test 6
* `hubble_units` = False

# Test 8
* Same as Test 7
* `hubble_units = True`
* `H0 = 100 * 1000`
    * Writing `H0` in ms^(-1)h/MPc units
    * Forcefully cancelling the `h` factor of `H0` with that of `PK.P(k,z)`

# Test 9
* Same as test 8
* `k = 10**np.linspace(-6,1,1000) * h` -> `k = 10**np.linspace(-6,1,1000)`
* And a bunch of other untracked stuff. 

# Test 10
* Same as test 9