import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

def integrand(l1, l2):
    return 1
l_max = 2
def integration(l_max):
    #return integrate.dblquad(lambda a, b: integrand(a, b), 0, l_max, lambda l1: 0, lambda l1: np.sqrt(l_max**2 - l1**2))[0]
    return integrate.quad(lambda a: integrand(a, 0), 0, l_max)[0]

print(integration(l_max))




