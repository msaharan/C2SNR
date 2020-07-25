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
"""
# Works
def itd(l1, l2):
    return 1

def itn(l_max):
    return integrate.dblquad(lambda l1, l2: itd(l1, l2), 0, l_max, lambda l1: 0, lambda l1: np.sqrt(l_max**2 - l1**2))[0]

print(itn(2))
"""



