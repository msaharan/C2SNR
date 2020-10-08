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

########################################################################
## job specific constants
########################################################################

### redshift of the source (signal emitter)

#z_s = 8
z_array = [8, 6, 4, 2]

### Fourier modes corresponding to the angular binning

#-- This is the L of N(L) and C_(L, j). We observe the signal in this 
#-- particular mode. Observing the signal over a large L range should
#-- give better and reliable results. Mathematically, it can be seen from
#-- the equation 15 of Pourtsidou et al 2014. The instrumental noise will
#-- be less if we take a larger l_max.

l_min = 100
l_max = 100000
l_step = 100

#j_array = [1]
jn = 1
#j_array = [1, 10, 18]

### To control the plot range

l_plot_ll = 100 #-- lower limit
l_plot_ul = 100000 #-- upper limit


c = 3 * 10**8
#---------------------- job specific constants end ---------------------------

#---------------------------- functions --------------------------------------#

#########################################################################
## Comoving distance between z = 0 and some redshift
#########################################################################

def distance(z):
    return cd.comoving_distance(z, **cosmo)

########################################################################
## Hubble ratio H(z)/H0
########################################################################

def hubble_ratio(z):
    return (omegam0 * (1 + z)**3 + omegal)**0.5

########################################################################
## Temperature T(z)
#########################################################################

def Tz(z): #-- in mK #-- for z = 8 we get Tz ~ 23 mK
    return 180 * omegaH1 * h * (1 + z)**2 / hubble_ratio(z)

########################################################################
## kmax corresponding to lmax
########################################################################

def k_max(l, z):
    return np.ceil(l/distance(z))

########################################################################
## Angular power spectrum
########################################################################

"""
def angpowspec_l_integrand(z, ell, dist_s, constfac):
    dist = distance(z)
    return ( 1 /  ( ell * (ell + 1) ) ) * constfac * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def angpowspec_l_integration(ell, z):
  #dist_s = distance(z)
    return integrate.quad(angpowspec_l_integrand, 0.0001, z, args = (ell, dist_s, constfac), limit=20000)[0]
"""

def angpowspec_lj(ell, j, z):
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / Len)**2 )**0.5
    return Tz(z)**2 * PK.P(z, k_var) / D2xLen

#------------------------- functions end -------------------------------------

########################################################################
## Constants
#######################################################################

### cosmological constants (Zahn and Zaldarriaga 2006) 

h = 0.7
omegam0 = 0.3
omegamh2 = omegam0 * h**2
omegal = 0.7
omegab = 0.04
omegabh2 = 0.04 * h**2
omegach2 = omegamh2 - omegabh2
H0 = 100 * h * 1000                     # ms**(-1)/Mpc units
ns = 1
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
c = 3 * 10**8

### From Al. Pou's detailed notes; 21alk.pdf (this is other than the one we use to understand the derivation of lensing reco. noise.)

omegaH1 = 0.034

for z_s in z_array:

    ### (comoving distance between observer and source)**2 * radial depth of the observed volume. D^2*length => D2xLen

    bandwidth = 5   #-- MHz
    Len = 1.2 * (bandwidth / 0.1) * ((1 + z_s) / 10)**0.5 * (0.15 / omegamh2)**0.5 * h          #-- Mpc; ZZ_2006
    D2xLen = distance(z_s)**2 * Len

    ## the constant factor in the ang pow spec expression

    constfac = 9 * (H0)**3 * omegam0**2 / c**3

    ## Thermal noise

    C_lN = 3.1 * 10**(-7)

    #----------------------------- constants end ----------------------------------

    ########################################################################
    ## CAMB parameters
    ########################################################################

    #-- The parameters other than omega(m/c/b) and h have not been taken from ZZ_2006. Do not use these
    #-- in the final calculations. Make sure to use the correct values. 
    #-- TODO for Mohit: Try to find out if changing ns, As and YHe causes major change
    #-- in the results. 
    #-- From the previous experience with these numbers, I don't think
    #-- that there will be big change in the results so I am proceeding with these values.
    #--
    #-- k_unit = True -> input of k will be in h/Mpc units
    #-- hubble_units = True -> output power spectrum PK.P(k, z) will be in Mpc/h units.
    #--
    #-- The k/PK business is very confusing because ZZ_2006 or P_2014 might have used
    #-- k and PK with or without the h units. I once tried to find it out but my tests were poorly 
    #-- structured so I couldn't quite find out which one they have used. I was trying 
    #-- different h, true, false combinations to see which combination gives the best match 
    #-- with the plot from P_2014.
    #-- See the tests in: 
    #-- C2SNR/tree/master/archive/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou
    #-- or if you can't find them there, check the permanent location:
    #-- C2SNR/tree/before_archive_20201002/Pow_spec_test_code/All_tests/Comparison_angpowspec_with_pourtsidou

    pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = 100 * h, omch2=omegach2, ombh2=omegabh2) #-- H0 in kms^-1/Mpc as per the documentation
    pars.DarkEnergy.set_params(w=-1.13)
    pars.set_for_lmax(lmax = l_max)
    pars.InitPower.set_params(ns=ns) # ZZ_2006 took n = 1 and sigma_8 = 0.9
    results = camb.get_background(pars)
    k = 10**np.linspace(-5,k_max(l_max, z_s),1000)
    PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = k_max(l_max, z_s), k_hunit = False, hubble_units = False)

    #------------------------ CAMB parameters end --------------------------------

    ########################################################################
    ## compute and store data
    ########################################################################

    #-- dist_s = distance between the observer and the emitter (harmless => global parameter)

    dist_s = distance(z_s)

#for jn in j_array:
for z in z_array:

    #aps_l_data = open('./text-files/angpowspec-l-lmax-{}-h-false.txt'.format(l_max), 'w')
    aps_lj_data = open('./text-files/angpowspec-lj-lmax-{}-j-{}-z-{}-h-false.txt'.format(l_max, jn, z), 'w')

    for L in tqdm(range(l_min, l_max, l_step)):
    #    aps_l_data.write('{}    {}\n'.format(L, angpowspec_l_integration(L, z_s)))
        aps_lj_data.write('{}    {}\n'.format(L, angpowspec_lj(L, jn, z_s)))

    #aps_l_data.close()
    aps_lj_data.close()

#-------------------- compute and store data end -------------------------------

########################################################################
## Read data and plot
########################################################################

## canvas and fonts

fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

## read
#L, CL = np.loadtxt('./text-files/angpowspec-l-lmax-{}-z-{}-h-false.txt'.format(l_max, z), unpack=True)

#for jn in j_array:
for z in z_array:
    L, C_lj = np.loadtxt('./text-files/angpowspec-lj-lmax-{}-j-{}-z-{}-h-false.txt'.format(l_max, jn, z), unpack=True)
    plt.plot(L, C_lj * L * (L + 1), label = '$C_{lj}$' + '(j = {})'.format(jn), linestyle='dashed')


## plot and decorate

#plt.plot(L, CL * L * (L + 1) / (2 * 3.14), label = '$C_L$', color = 'black')

plt.plot(L, C_lN * L * (L + 1) / (2 * 3.14), label = '$C_{l}^N$', color = 'red')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)
#plt.ylabel('$C_l$', fontsize = fs)
plt.ylabel('$C_l\;l(l+1)/2\pi$', fontsize = fs)
plt.legend(fontsize = fs)
plt.xlim(l_plot_ll, l_plot_ul)
plt.title("Mpc units")
#plt.ylim(4E-10, 3E-8)
#plt.savefig('./plots/angpowspec-lmax-{}.pdf'.format(l_max), bbox_inches='tight')
plt.savefig('./plots/angpowspec-lj-lmax-{}-z{}-h-false.pdf'.format(l_max, z), bbox_inches='tight')
plt.show()

"""
plt.loglog(k, PK.P(z_s,k))
plt.xlabel('$k\;(1/Mpc)$', fontsize = fs)
plt.ylabel('$P(k)\;(Mpc^3)$', fontsize = fs)
plt.xlim(1E-5, k_max(l_max, z_s))
plt.legend(fontsize = fs)
plt.savefig('./plots/matter-angpowspec-lj-lmax-{}-j-{}.pdf'.format(l_max, j_max), bbox_inches='tight')
plt.show()

plt.plot(k, angpowspec_lj(k/distance(z_s), j_max, z_s))
plt.show()
"""
#------------------------ read data and plot end ----------------------
