import cosmolopy.distance as cd
import numpy as np

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
# Cosmology from Dumitru et al. 2019
omegam0 = 0.308
omegal = 0.692
omegak0 = 0.0
h = 0.678
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
l_max = int(input())
print("Using Ang. Dia. Dist.")
k_max = np.ceil(l_max/a_distance(redshift))
print(a_distance(redshift))
if k_max > 0 and k_max<=10:
    k_lim = 1
elif k_max>10 and k_max<=100:
    k_lim = 2
elif k_max>100 and k_max<=1000:
    k_lim = 3
elif k_max>1000 and k_max<=10000:
    k_lim = 4
print(l_max/a_distance(redshift))
print("kmax using ang. dia. distance: {}".format(k_max))
print("klim using ang. dia. distance: {}".format(k_lim))

print("Using Comov. Dist.")
k_max = np.ceil(l_max/c_distance(redshift))
print(c_distance(redshift))
if k_max > 0 and k_max<=10:
    k_lim = 1
elif k_max>10 and k_max<=100:
    k_lim = 2
elif k_max>100 and k_max<=1000:
    k_lim = 3
elif k_max>1000 and k_max<=10000:
    k_lim = 4
print(l_max/c_distance(redshift))
print("kmax using comov. distance: {}".format(k_max))
print("klim using comov. distance: {}".format(k_lim))

#------------------------------------------------------------------------------

"""
redshift  = 6.349
k = 0.034
print(0.034 * 0.67 * c_distance(redshift))
"""
