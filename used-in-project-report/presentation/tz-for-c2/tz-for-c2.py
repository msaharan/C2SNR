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

def cps(z):
    pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
    pars.DarkEnergy.set_params(w=-1.13)
    #pars.set_for_lmax(lmax = l_max)
    pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
    results = camb.get_background(pars)
    PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = np.max(kc2), k_hunit = True, hubble_units = True)
    PK.P(z, kc2)
    return PK.P(z, kc2)

# CII Power Spectrum
pk3_2pi_file, kc2 = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6.txt', unpack=True)
pc2 = pk3_2pi_file * 2 * 3.14**2 / kc2**3

# CAMB power spectrum
pm = cps(6.349)

# Arrays

# Store the finer arrays to use later
fouttz = open('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6_tz.txt', 'w')

# Make the array finer
c = 0
for i in kc2f:
    pc2f[c] = np.interp(i, kc2, pc2)
    pmf[c] = np.interp(i, k, pm)
#    fout.write('{}  {}\n'.format(round(i**3 * pc2f[c] / (2 * 3.14**2), 3), i))
    fout.write('{}  {}\n'.format(i**3 * pc2f / (2 * 3.14**2), i))
    fouttz.write('{}  {}\n'.format((pc2f[c] / pmf[c])**2 * i**3 * pmf[c] / (2 * 3.14**2) , i))
    c = c + 1 

Tz = pc2f/pmf

fout.close()
fouttz.close()

# Plots
plt.subplots()
plt.plot(kc2, pc2, label = 'pc2')
plt.scatter(kc2f, pc2f, label = 'pc2f', marker='o')
plt.plot(kc2f, Tz, label = 'Div.')
plt.plot(kc2f, pmf, label = 'pmf', marker='o')
plt.plot(k, pm, label = 'pm')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
