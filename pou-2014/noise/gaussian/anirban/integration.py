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
l_ul = 20000

### L range for plotting
l_plot_ll = 10
l_plot_ul = 2000

### Discretization of the observed volume along the line of site
#-- j=0 mode will be made useless due to foreground removal.
#-- Stated at the end of section 2 in P_2014.
#j = 1
j_min = 1
j_max = 40

### source redshift
z_s = 2
#-----------------------------------------------------------------------

########################################################################
## Comoving distance between z = 0 and some redshift
########################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#-----------------------------------------------------------------------

########################################################################
## hubble ratio H(z)/H0
########################################################################
def hubble_ratio(var):
    return (omegam * (1 + var)**3 + omegal)**0.5
#-----------------------------------------------------------------------

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
#-----------------------------------------------------------------------

########################################################################
## lensing reconstruction noise N_L for Gaussian distribution
########################################################################
#-- Equation 4 in Pourtsidou et al 2014 or Eq. 30 in Zahn and Zald. 2006

def noise_denominator_integrand(l1, phi_1, ell, j):
    l2 = (ell**2 + l1**2 - (2*ell*l1*np.cos(phi_1)))**0.5
    phi_2 = phi_1 + math.atan(ell * np.sin(phi_1) / (l1 - ell*np.cos(phi_1)))
    return l1 * (C_lj(l1, j) * ell * l1 + C_lj(abs(l2), j) * l2 * ell * np.cos(phi_2))**2 / (ell**2 * (2 * 3.14)**2 * 2 * (C_lj(l1, j) + C_l_N)  * (C_lj(abs(l2), j) + C_l_N))

"""
def noise_denominator(ell, j):
    return 2 * integrate.nquad(noise_denominator_integrand, [(1, l_ul), (0, np.pi)], args = (ell, j), opts = [{'limit': 2000}, {'limit': 2000}])[0]
"""
#-----------------------------------------------------------------------

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
Len = 1.2 * (bandwidth / 0.1) * ((1 + z_s) / 10)**0.5 * (0.15 / omegamh2)**0.5 / h

### the D^2 L factor in equation 4 of P_2014
D2xLen = distance(z_s)**2 * Len

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
C_lj_array = np.zeros((2 * l_ul, j_max))
#------------------------------------------------------------------------

########################################################################
## CAMB
########################################################################
pars = model.CAMBparams(NonLinear = 0, WantTransfer = True, H0 = 100 * h, omch2 = omegach2, ombh2 = omegabh2, YHe = YHe)
pars.DarkEnergy.set_params(w = -1)
pars.set_for_lmax(lmax = l_ul)
pars.InitPower.set_params(ns = ns, As = As)
#results = camb.get_background(pars)
results = camb.get_results(pars)
#print(results.get_sigma8()) #-- gives 0.84166148; 0.83 in Planck 2013
k = np.linspace(10**(-5), k_max(l_ul, z_s) ,1000)
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = np.max(k), k_hunit = False, hubble_units = False)
#-----------------------------------------------------------------------

########################################################################
## write data
########################################################################
dl1 = (l_ul - l_ll)/10
dp1 = 3.141/10
l1array = np.arange(l_ll, l_ul, dl1)
p1array = np.arange(0, 3.14, dp1)

### d2l_integration
print("Storing data")
len_reco_noise_data = open('./text-files/len_reco_noise_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max), 'w')
for large_L in tqdm(range(l_plot_ll, l_plot_ul)):
    noise_denominator_sum = 0
    for j in range(j_min, j_max):
        #-- approximating the "dl" "dtheta" integral by assuming "l" and "theta" as sides of a rectangle; l1 is just l1 integration variable and p1 means phi_1
        area = 0
        for l1 in l1array:
            for p1 in p1array:
                area = area + (noise_denominator_integrand(l1, p1, large_L, j) * dl1 * dp1)
    noise_denominator_sum = noise_denominator_sum + area
    len_reco_noise_data.write('{}  {}\n'.format(large_L, 1/(2 * noise_denominator_sum)))
len_reco_noise_data.close()


#-----------------------------------------------------------------------

########################################################################
## read data and plot
########################################################################

### canvas
fig, ax = plt.subplots(figsize=(7,7))
fs = 14
mfs = 12
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=mfs)

### read data
#-- 'i' denotes 'input'

print("Plotting data")

iLn, ilen_reco_noise = np.loadtxt('./text-files/len_reco_noise_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max), unpack=True)
plt.plot(iLn, ilen_reco_noise, label = 'N(L)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)

plt.ylabel('N(L)', fontsize = fs)

#plt.title('L = {};'.format(large_L) + ' $z_{source} = $' + '{}'.format(z_s))
plt.legend(fontsize = fs)

plt.savefig('./plots/len_reco_noise_z_{}_lmax_{}_jmax_{}.pdf'.format(z_s, l_ul, j_max), bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------

