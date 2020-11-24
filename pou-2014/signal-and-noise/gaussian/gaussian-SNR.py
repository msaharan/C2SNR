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
#l_ul = 1000
l_ul = 70000

### Discretization of the observed volume along the line of site
#-- j=0 mode will be made useless due to foreground removal.
#-- Stated at the end of section 2 in P_2014.
#j = 1
j_min = 1
#j_max = 10
j_max = 120

### source redshift
z_s = 2

### plot params
l_plot_min = 10
l_plot_max = 1000
#err_stepsize = exp.delta_l
err_stepsize = 200
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
def C_l_integrand(z, ell):
    dist = distance(z)
    return constfac * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) * (1 + z)**2 / hubble_ratio(z)

def C_l(ell):
    return (1 / (ell * (ell + 1) )) * integrate.quad(C_l_integrand, 0.0001, z_s, args = (ell), limit=20000)[0]

def C_lj(ell, j):
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / Len)**2 )**0.5
    return Tz(z_s)**2 * PK.P(z_s, k_var) / D2xLen
#------------------------------------------------------------------------

########################################################################
## lensing reconstruction noise N_L for Gaussian distribution
########################################################################
#-- Equation 4 in Pourtsidou et al 2014 or Eq. 30 in Zahn and Zald. 2006

def noise_denominator_integrand(l1, l2, ell, j):
    l_var = int(np.sqrt(l1**2 + l2**2))
    return (C_lj_array[ell, j] * ell * l_var + C_lj_array[abs(ell-l_var), j] * ell * (ell - l_var))**2 / (ell**2 * (2 * 3.14)**2 * 2 * ( C_lj_array[l_var, j] + C_l_N)  * ( C_lj_array[abs(ell - l_var), j] + C_l_N))
    
#------------------------------------------------------------------------

########################################################################
## error bars
########################################################################
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
Len = 1.2 * (bandwidth / 0.1) * ((1 + z_s) / 10)**0.5 * (0.15 / omegamh2)**0.5 / h

### the D^2 L factor in equation 4 of P_2014
D2xLen = distance(z_s)**2 * Len

### constant factor ouside the integral in equation 20 of P_2014
constfac = (9) * (H0/c)**3 * omegam**2 

### distance between source and observer
dist_s = distance(z_s)

#### thermal/instrumental noise
#-- C_l^N in eq. 14 of P_2014
#-- c_lj^tot = C_lj + C_l^N)
C_l_N = (2 * 3.14)**3 * T_sys**2 / (bandwidth * t_obs * f_cover**2 * l_max**2)

### number of error points to dsiplay on the SNR plot
n_err_points = 1 + int((l_plot_max - l_plot_min)/err_stepsize)

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

C_l_data = open('./text-files/c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max - 1), 'w')

N_l_data = open('./text-files/n_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max-1), 'w')

delta_C_l_data = open('./text-files/delta_c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max-1), 'w')

### C_l
for L in tqdm(range(l_plot_min, l_plot_max)):
    C_l_data.write('{}    {}\n'.format(L, C_l(L)))
C_l_data.close()

### C_lj
for L in range (l_plot_min, 2 * l_plot_max):
    for j in range(j_min, j_max):
        C_lj_array[L, j] = C_lj(L,j)

### N_l
for L in range(l_plot_min, l_plot_max, err_stepsize):
    j_sum = 0
    for j in tqdm(range(j_min, j_max)):
        d_square_l_sum = 0
        for l1 in range(1, l_ul, int(l_ul/100)):
            for l2 in range(1, l_ul, int(l_ul/100)):
                d_square_l_sum = d_square_l_sum + (noise_denominator_integrand(l1, l2, L, j) * (l_ul/100)**2)
        j_sum = j_sum + d_square_l_sum

    N_l = 1/j_sum
    
    N_l_data.write('{}  {}\n'.format(L, N_l))

    delta_C_l_data.write('{}    {}   {}\n'.format(L, C_l(L), np.sqrt(2 /( (2 * L + 1) * delta_l * f_sky)) * (C_l(L) + N_l)))

N_l_data.close()
delta_C_l_data.close()

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
iLc, iC_L = np.loadtxt('./text-files/c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max-1), unpack=True)

iLn, iN_L = np.loadtxt('./text-files/n_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max-1), unpack=True)


iLd, iC_Ld, idelta_C_L = np.loadtxt('./text-files/delta_c_l_data_z_{}_lmax_{}_jmax_{}.txt'.format(z_s, l_ul, j_max-1), unpack=True)

p_l = np.zeros(2 * l_ul)
p_c_l = np.zeros(2 * l_ul)
p_l, p_c_l, junk, junk, junk, junk = np.loadtxt("/mnt/storage/pdata/Utility/Git/dyskun/C2SNR/archive/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)
plt.loglog(p_l,p_c_l, label='Pourtsidou et al. 2014 (z=2)')

### plot
plt.plot(iLc, iC_L * iLc * (iLc + 1) / (2 * 3.14), label = 'This work', color='black')
for i in range(n_err_points):
    plt.errorbar(iLd[i],  iC_Ld[i] * iLd[i] * (iLd[i] + 1) / (2 * 3.14), yerr = idelta_C_L[i], capsize=3, ecolor='black')

plt.plot(iLn, iLn * (iLn + 1) * iN_L / (2 * 3.14), label = '$N_{L}$', linestyle = 'dashed') #-- N_l * l(l+1)/2pi

#plt.plot(iLn, iN_L, label = '$N_{L}$') #-- N_l only

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$', fontsize = fs)

#plt.ylim(1E-9, 1E-7)

#plt.ylabel(r'$N_{l} \times l(l+1)/2\pi$', fontsize = fs)
plt.ylabel(r'$C_{L} \times L (L+1) / 2\pi$', fontsize = fs)

plt.title('$z_{source} = $' + '{}'.format(z_s))
plt.legend(fontsize = fs)
plt.xlim(l_plot_min, l_plot_max)

#plt.savefig('./plots/n_l_z_{}_lmax_{}_j_{}.pdf'.format(z_s, l_ul, j_max), bbox_inches='tight')
plt.savefig('./plots/gaussian_z_{}_lmax_{}_j_{}.pdf'.format(z_s, l_ul, j_max), bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------

##################
## Temporary trash
##################

"""
#import sympy
#from scipy.integrate import simps


l_array = np.arange(0, l_ul, 1)
def noise_denominator_integration(ell,j):
    #return number * integrate.nquad(noise_denominator_integrand, [(0, l_ul),( 0, l_ul)], args = (ell, j), opts = [{'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}, {'limit' : 5000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}])[0]
#return integrate.nquad(noise_denominator_integrand, [(0, l_ul),( 0, l_ul)], args = (ell, j))[0]
    
    l1, l2 = sympy.symbols('l1 l2')
    #print(sympy.integrate(noise_denominator_integrand(l1, l2, ell, j), (l1, 0, l_ul), (l2, 0, l_ul))[0]
    return(sympy.integrate(noise_denominator_integrand(l1, l2, ell, j), (l1, 0, l_ul), (l2, 0, l_ul)))


def midpoint_double1(f, a, b, c, d, nx, ny, foo1, foo2):
#-- source: https://link.springer.com/chapter/10.1007/978-3-030-16877-3_6#Sec20
    hx = (b - a)/nx
    hy = (d - c)/ny
    I = 0
    for i in range(nx):
        for j in range(ny):
            xi = a + hx/2 + i*hx
            yj = c + hy/2 + j*hy
            I = I + hx*hy*f(xi, yj)
    return I
"""

