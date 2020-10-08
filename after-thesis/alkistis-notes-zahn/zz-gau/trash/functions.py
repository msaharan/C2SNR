#---------------------------- functions --------------------------------------#

##############################################################################
## Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)

###############################################################################
## Constant factor; Integration over redshift
###############################################################################
#def constantfactor(z):
#   return 9 * (H0)**3 * omegam0**2 / c**3
#-- Eqn 21 of ZZ has c^2 in denominator

###############################################################################
## Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5

###############################################################################
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constfac):
    dist = distance(z)
    return ( 1 /  ( ell * (ell + 1) ) ) * constfac * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def angpowspec_integration_without_j(ell, z):
  #dist_s = distance(z)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, z, args = (ell, dist_s, constfac), limit=20000)[0]

#------------------------- functions end -------------------------------------
