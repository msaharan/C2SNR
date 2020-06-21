import cosmolopy.distance as cd

###############################################################################
# Ang dia distance between z = 0 and some redshift
###############################################################################
def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Comoving distance between z = 0 and some redshift
###############################################################################
def c_distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Set cosmology for Cosmolopy
###############################################################################
omegam0 = 0.315
omegal = 0.685
omegak0 = 0.0
H0 = 67.3 * 1000
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': omegak0, 'h': h}
#------------------------------------------------------------------------------

###############################################################################
# kmax ~ lmax/distance
###############################################################################
# redshift = redshift of the source
# To find k using comoving distance: lmax/c_distance(redshift) 
# To find k using angular diameter distance: lmax/a_distance(redshift) 
print("Enter the source redshift (float):")
redshift = float(input())
print("Enter lmax (integer):")
lmax = int(input())
print("kmax using ang. dia. distance (kmax = lmax/angdiadistance): {}".format(lmax/a_distance(redshift)))
print("kmax using comoving distance (kmax = lmax/comovingdistance): {}".format(lmax/c_distance(redshift)))
#------------------------------------------------------------------------------
