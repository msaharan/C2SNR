import numpy as np
import cosmolopy.distance as cd
cosmo = {'omega_M_0': 0.308,                                                                                                                                                                 
         'omega_lambda_0': 0.692,
         'omega_k_0': 0.0,
         'h': 0.678}
# Angular diameter distance 
L = cd.angular_diameter_distance(2, z0=0, **cosmo)

# (proper) width = height of the observed volume
D = 200 * L

#Volume 
V = L * D**2
print("L = " + str(L) + " Mpc")
print("D = "+ str(D) + " Mpc")
print("V = " + str(V) + "Mpc^3")


