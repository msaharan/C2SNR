import matplotlib
from matplotlib import pyplot as plt
import numpy as np 
import scipy.integrate as integrate
import cosmolopy.distance as cd
import sys
import platform
import os
import camb
from camb import model, initialpower, get_matter_power_interpolator
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
from scipy.constants import physical_constants
from scipy.constants import parsec as parsec_in_m
from scipy.constants import year as year_in_s 
from scipy.integrate import quad

# Sherwood parameters (Planck 2014)
##################################################################
## Mohit
##################################################################

import p_2014_cosmology as cos

#omega_m = 0.25 #-- Girish
omega_m = cos.omegam #-- now

#omega_l = 0.70
omega_l = cos.omegal

#omega_b = 0.05
omega_b = cos.omegab

#h = 0.700
h = cos.h

#ns = 0.961
ns = cos.ns

#sigma_8 = 0.9
sigma_8 = 0.829 

#-------------------------------------------------------------------

cosmo = {'omega_M_0' : omega_m, 'omega_lambda_0' : omega_l, 'h' : h}
cosmo = cd.set_omega_k_0(cosmo)
H0 = 1.023e-10 * h # yr^-1
c = physical_constants[u'speed of light in vacuum'][0] # m s^-1
Mpc = parsec_in_m * 1.0e6 # m 

def D_transverse(z):

    return cd.comoving_distance_transverse(z, **cosmo)/h # Mpc/h

def constfac(z):

    return 9.0 * omega_m**2 * H0**3 / (c/Mpc * year_in_s * h)**3 # (Mpc/h)^-3

pars = camb.CAMBparams()
pars.set_cosmology(H0=100.0*h, ombh2=omega_b*h*h, omch2=omega_l*h*h)
pars.InitPower.set_params(As=2.196e-9, ns=0.9624)
#pars.set_for_lmax(2500, lens_potential_accuracy=1)
pars.set_for_lmax(2500, lens_potential_accuracy=1)
results = camb.get_background(pars)

PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = 100) # (Mpc/h)^3
    
def integrand(z, ell, dist_s):

    t1 = 1.0/cd.e_z(z, **cosmo)
    
    dist = D_transverse(z)
    t2 = 1.0 - dist/dist_s

    t3 = (1+z)**2

    t4 = PK.P(z, ell/dist) # (Mpc/h)^3

    return constfac(z) * t1 * t2 * t3 * t4 # dimensionless

def c_ell(z, ell):

    d = D_transverse(z) 
    res = quad(integrand, 0, z, args=(ell, d))[0] # dimensionless

    return res 

ell = np.logspace(0,5,num=100)
z_source = 0.8
Cell = np.array([c_ell(z_source, x) for x in ell])

########################################################################
## Mohit
##
## Plot from Pourtsidou 2014 (without error bars)
########################################################################
p_l = np.zeros(600)
p_c_l = np.zeros(600)
p_l, p_c_l, junk, junk, junk, junk = np.loadtxt("/mnt/storage/pdata/Utility/Git/dyskun/C2SNR/archive/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)
#-----------------------------------------------------------------------

plt.figure(figsize=(5,5))
plt.plot(p_l,p_c_l, label='Pourtsidou et al. 2014 (z=2)')
plt.plot(ell, Cell)
plt.xscale('log')
plt.yscale('log')
plt.xlim(5, 1.0e5)
plt.ylim(1.0e-13, 1.0e-7)
plt.xlabel(r'ell')
plt.ylabel(r'C_ell^kk')
plt.savefig('./plots/ps.pdf', bbox_inches='tight')

plt.figure(figsize=(5,5))
plt.plot(p_l,p_c_l, label='Pourtsidou et al. 2014 (z=2)')
plt.plot(ell, Cell*2/np.pi)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10, 3000)
plt.ylim(1.0e-9, 1.0e-7)
plt.xlabel(r'ell')
plt.ylabel(r'C_ell^dthetadtheta')
plt.savefig('./plots/ps2.pdf', bbox_inches='tight')
