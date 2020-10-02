import numpy as np
import scipy.integrate as integrate
"""
def N4(ell):
    return integrate.dblquad(lambda l1,l2: N4_integrand(ell,l1,l2), 0, 10, lambda l2: 0, lambda l2: 10)[0]

def N4_integrand(ell,l1,l2): 
    small_L = l1 + l2
    return ell + small_L

ell = 0
print(N4(ell))
"""


def N4(ell):
    return integrate.dblquad(N4_integrand(ell,l1,l2), 0, 10, 0, 10)[0]

def N4_integrand(ell,l1,l2): 
    small_L = l1 + l2
    return ell + small_L

ell = 0
print(N4(ell))

