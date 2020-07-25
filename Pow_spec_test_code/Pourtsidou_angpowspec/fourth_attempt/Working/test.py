import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm

def integrand(l1, l2):
    return 1

print(integrate.nquad(integrand, [(0, 1), (0, np.sqrt(1 - foo**2))]))
