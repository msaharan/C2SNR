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

### angular binning; max and min Fourier mode
l_ll = 1
l_ul = 10000

### Discretization of the observed volume along the line of site
#-- j=0 mode will be made useless due to foreground removal.
#-- Stated at the end of section 2 in P_2014.
#j = 1
j_array = [1, 5, 10, 20, 50, 100]

### source redshift
z_s = 2

### plot params
l_plot_min = l_ll
l_plot_max = l_ul

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
## kmax corresponding to lmax
########################################################################

def k_max(l, z):
    return np.ceil(l/distance(z))

########################################################################
## angular power spectrum
########################################################################
def C_lj(ell, j):
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / Len)**2 )**0.5
    return Tz(z_s)**2 * PK.P(z_s, k_var) / D2xLen
#------------------------------------------------------------------------

########################################################################
## constants
########################################################################

c = 3 * 10**8

### cosmology
h = cos.h
H0 = h * 100 * 1000
omegal = cos.omegal
omegam = cos.omegam
omegab = cos.omegab
omegac = cos.omegac
omegamh2 = omegam * h**2
omegabh2 = omegab * h**2
omegach2 = omegac * h**2
YHe = cos.YHe
As = cos.As
ns = cos.ns
cosmo = {'omega_M_0': omegam, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}


### experiment parameters
bandwidth = exp.bandwidth   #-- MHz
delta_l = exp.delta_l
f_sky = exp.f_sky
omegaH1 = exp.omegaH1
l_max = exp.l_max
T_sys = exp.T_sys
t_obs = exp.t_obs
f_cover = exp.f_cover

### depth (in Mpc) of the observed volume along the line of site
#-- source: ZZ_2006
Len = 1.2 * (bandwidth / 0.1) * ((1 + z_s) / 10)**0.5 * (0.15 / omegamh2)**0.5 * h

### the D^2 L factor in equation 4 of P_2014
D2xLen = distance(z_s)**2 * Len

### constant factor ouside the integral in equation 20 of P_2014
constfac = (9/4) * (H0/c)**3 * omegam**2 

### distance between source and observer
dist_s = distance(z_s)

#### thermal/instrumental noise
#-- C_l^N in eq. 14 of P_2014
#-- c_lj^tot = C_lj + C_l^N)
C_l_N = (2 * 3.14)**3 * T_sys**2 / (bandwidth * t_obs * f_cover**2 * l_max**2)


#-----------------------------------------------------------------------

########################################################################
## Arrays
########################################################################

### angular power spectrum 
#-- signal
C_l_array = np.zeros(int(2**0.5 * l_ul))
#------------------------------------------------------------------------

########################################################################
## CAMB
########################################################################
#print('********k_max**********{}'.format(k_max(l_ul, z_s)))

pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0 = 100 * h, omch2 = omegach2, ombh2 = omegabh2, YHe = YHe)
pars.DarkEnergy.set_params(w = -1)
pars.set_for_lmax(lmax = l_ul)
pars.InitPower.set_params(ns = ns, As = As)
#results = camb.get_background(pars)
results = camb.get_results(pars)
#print(results.get_sigma8()) #-- 0.84166148
k = np.linspace(10**(-5), k_max(l_ul, z_s) ,1000)
#print()
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = np.max(k), k_hunit = False, hubble_units = False)
#-----------------------------------------------------------------------

########################################################################
## compute and plot
########################################################################

### canvas
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

for j in tqdm(j_array):

    ########################################################################
    ## write data
    ########################################################################

    C_lj_data = open('./text-files/c_lj_and_c_l_n_data_z_{}_lmax_{}_j_{}.txt'.format(z_s, l_ul, j), 'w')

    ### C_lj
    for L in tqdm(range(l_plot_min, l_plot_max)):
        C_lj_data.write('{}    {}\n'.format(L, C_lj(L, j)))

    C_lj_data.close()

    #-----------------------------------------------------------------------

    ########################################################################
    ## read data and plot
    ########################################################################
    
    ### read data
    #-- 'i' denotes 'input'
    iL, iC_L = np.loadtxt('./text-files/c_lj_and_c_l_n_data_z_{}_lmax_{}_j_{}.txt'.format(z_s, l_ul, j), unpack=True)

    ### plot C_lj
    plt.plot(iL, iL * (iL + 1) * iC_L / (2 * 3.14), label = 'j = {}'.format(j), linestyle = 'dashed') #-- C_lj * l(l+1)/2pi
    #plt.plot(iL, iC_L, label = 'j = {}'.format(j)) #-- C_lj only


### plot C_l^N
plt.plot(iL, iL * (iL + 1) * C_l_N / (2 * 3.14), label = '$C_l^N$', color = 'black') #-- C_l^N * l(l+1)/2pi

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$l$', fontsize = fs)

plt.ylabel(r'$C_{lj} \times l(l+1)/2\pi$', fontsize = fs)
#plt.ylabel(r'$C_{lj}$', fontsize = fs)

plt.title('$z_{source} = $' + '{}'.format(z_s))
plt.legend(fontsize = fs)
plt.xlim(l_plot_min, l_plot_max)

plt.savefig('./plots/c_lj_and_c_l_n_z_{}_lmax_{}_j_{}.pdf'.format(z_s, l_ul, j), bbox_inches='tight')
#plt.savefig('./plots/c_lj_z_{}_lmax_{}_j_{}.pdf'.format(z_s, l_ul, j), bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------
