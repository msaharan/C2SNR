##############################################################################
## Constants
###############################################################################

### cosmological constants (Zahn and Zaldarriaga 2006) 
omegam0 = 0.3
omegal = 0.7
omegab = 0.04
h = 0.7
omegabh2 = 0.04 * h**2
omegach2 = (omegam0 * h)**2 - omegabh2
H0 = 100 * h * 1000                       # ms**(-1)/Mpc units
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
c = 3 * 10**8

## the constant factor in the ang pow spec expression
constfac = 9 * (H0)**3 * omegam0**2 / c**3
#----------------------------- constants end ----------------------------------
