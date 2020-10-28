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
import time
import p_2014_parameters as exp
import p_2014_cosmology as cos 

########################################################################
## Job specific parameters
########################################################################

### source redshift
z_s = np.arange(0, 10, 0.1)
#------------------------------------------------------------------------

########################################################################
## Comoving distance between z = 0 and some redshift
########################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------

########################################################################
## hubble ratio H(z)/H0
########################################################################
def hubble_ratio(var):
    return (omegam * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------

########################################################################
## temperature T(z)
########################################################################

def Tz(z): #-- in mK
    return 180 * omegaH1 * h * (1 + z)**2 / hubble_ratio(z)

########################################################################
## constants
########################################################################

### cosmology
h = cos.h
omegal = cos.omegal
omegam = cos.omegam


### experiment parameters
omegaH1 = exp.omegaH1
#-----------------------------------------------------------------------

########################################################################
## plot data
########################################################################
### canvas
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

### plot
plt.plot(z_s, Tz(z_s), color = 'black')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('z', fontsize = fs)
plt.ylabel('Tz (mK)', fontsize = fs)
plt.title('Mean $T_{b}^{HI}$ at redshift z', fontsize = fs)
plt.savefig('./plots/tz.pdf', bbox_inches='tight')
plt.show()

