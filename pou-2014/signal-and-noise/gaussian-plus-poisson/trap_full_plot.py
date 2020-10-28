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
l_ll = 10
l_ul = 10000

### Discretization of the observed volume along the line of site
#-- j=0 mode will be made useless due to foreground removal.
#-- Stated at the end of section 2 in P_2014.
j_min = 1
j_max = 2

### source redshift
z_s = 2

### plot params
l_plot_min = l_ll
l_plot_max = 600
err_stepsize = exp.delta_l

n_err_points = 1 + int((l_plot_max - l_plot_min)/err_stepsize)

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
    return ell * (ell + 1) * (0.5/3.14) * integrate.quad(C_l_integrand, 0.0001, z_s, args = (ell), limit=20000)[0]

def C_lj(ell, j):
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / Len)**2 )**0.5
    return Tz(z_s)**2 * PK.P(z_s, k_var) / D2xLen

#------------------------------------------------------------------------

###############################################################################
## To calculate the noise moments (equation 14 in Pourtsidou et al. 2014)
###############################################################################
def N4_integrand(l1, l2, ell, j):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2 * (C_l_array[ell, j] + (C_l_N)) * (C_l_array[abs(ell-small_l), j] + (C_l_N)) / (two_pi_squared * Tz(z_s)**4)

def N4(ell, j):
    #print("-------------N4----------------")
    return integrate.nquad(N4_integrand, [(0, l_ul), (0, l_ul)], args = (ell, j), opts = [{'limit' : 20000}, {'limit' : 20000}])[0] * Tz(z_s)**4

def N3_integrand(l1, l2, ell, j):
    small_l = int(np.sqrt(l1**2 + l2**2))
    """ 
    print('C_l_array[ell, j] {}'.format(C_l_array[ell, j]))
    print('C_l_array[abs(ell-small_l),j] {}'.format(C_l_array[abs(ell-small_l),j]))
    print('(C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N){}'.format((C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N)))
    print('Cshot * 2 * ((C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N)) / two_pi_squared {}'.format(Cshot * 2 * ((C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N)) / two_pi_squared))
    """
    #print(Cshot * 2 * ((C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N)) / (two_pi_squared * Tz(z_s)**2))
    return Cshot * 2 * ((C_l_array[ell, j] + C_l_N) + (C_l_array[abs(ell-small_l),j]  + C_l_N)) / (two_pi_squared * Tz(z_s)**2)

def N3(ell, j):
    return Tz(z_s)**2 * integrate.nquad(N3_integrand, [(0, l_ul), (0, l_ul)], args = (ell, j), opts = [{'limit' : 20000, 'epsabs' : 1.49e-02, 'epsrel' : 1.49e-02}, {'limit' : 20000, 'epsabs' : 1.49e-02, 'epsrel' : 1.49e-02}])[0]
    #opts = [{'limit' : 20000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}, {'limit' : 20000, 'epsabs' : 1.49e-05, 'epsrel' : 1.49e-05}]

def N2(ell, j, N3_for_this_L_j):
    #print("-------------N2----------------")
    #return Tz(z_s)**2 * mass_moment_2**2 * N3_for_this_L_j / (2 * eta_D2_L**2 * mass_moment_1**4 * Cshot)
    return Tz(z_s)**2 * mass_moment_2**2 * N3_for_this_L_j / (2 * eta_D2_L**2 * mass_moment_1**4 * Cshot)

def N1(ell, j, N3_for_this_L_j):
    #print("-------------N1----------------")
    #return Tz(z_s)**2 * j_max * mass_moment_3 * (l_ul**2 / two_pi_squared) * N3_for_this_L_j / (eta_D2_L**2 * mass_moment_1**3 * Cshot)
    return Tz(z_s)**2 * j_max * mass_moment_3 * (l_ul**2 / two_pi_squared) * N3_for_this_L_j / (eta_D2_L**2 * mass_moment_1**3 * Cshot)

def N0(ell):
    #print("-------------N0----------------")
    #return Tz(z_s)**4 * j_max**2 * mass_moment_4 * (l_ul**2 / two_pi_squared)**2 / (eta_D2_L**3 * mass_moment_1**4)
    return Tz(z_s)**4 * j_max**2 * mass_moment_4 * (l_ul**2 / two_pi_squared)**2 / (eta_D2_L**3 * mass_moment_1**4)
#------------------------------------------------------------------------------

###############################################################################
# To calculate the denominator of the lensing recosntruction noise term (eq 14 in Pourtsidou et al. 2014)
###############################################################################

def noise_denominator_integrand(l1, l2, ell,j):
    small_l = int(np.sqrt(l1**2 + l2**2))
    #print('Den Integrand {}'.format((C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * abs(ell - small_l) + ell**2 * Cshot) / (two_pi_squared * Tz(z_s)**2)))
    #print('C(l) {}   C(L - l) {}'.format(C_l_array[ell, j]/Tz(z_s)**2, C_l_array[abs(ell-small_l), j]/Tz(z_s)**2))
    #time.sleep(0.05)
    #return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * abs(ell - small_l) + ell**2 * Cshot) / (two_pi_squared * Tz(z_s)**2)
    #return 2.75196 * 10**(-6)
    #return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * abs(ell - small_l) + ell**2 * Cshot) / (two_pi_squared * Tz(z_s)**2)
    return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * abs(ell - small_l) + ell**2 * Cshot) / (two_pi_squared)
"""
def noise_denominator_integration(ell,j, redshift):
    #return Tz(z_s)**2 * integrate.dblquad(lambda l1, l2: noise_denominator_integrand(l1, l2, ell, j, redshift), 0, l_max, lambda l1: 0, lambda l1: np.sqrt(l_max**2 - l1**2))[0]
    #return Tz(z_s)**2 * integrate.nquad(noise_denominator_integrand, [(ell, l_max), (ell, l_max)] , args = (ell, j, redshift), opts = [{'limit' : 50, 'epsabs' : 1.49e-11, 'epsrel' : 1.49e-11}, {'limit' : 50, 'epsabs' : 1.49e-11, 'epsrel' : 1.49e-11}])[0] / 100000
    #print (Tz(z_s)**2 * integrate.nquad(noise_denominator_integrand, [(0, l_max), (0, l_max)] , args = (ell, j, redshift))[0])
    #return Tz(z_s)**2 * integrate.nquad(noise_denominator_integrand, [(0, l_max), (0, l_max)] , args = (ell, j, redshift))[0]

    return Tz(z_s)**2 * noise_denominator_integrand(0,0 , ell, j, redshift) * l_max**2
"""

"""
# Single Integration
def noise_denominator_integrand(l1, ell,j, redshift):
    small_l = int(l1)
    #print('Den Integrand {}'.format((C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * Cshot) / (two_pi_squared * Tz(z_s)**2)))
    return (C_l_array[ell, j]*ell*small_l + C_l_array[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * Cshot) / (two_pi_squared * Tz(z_s)**2)

def noise_denominator_integration(ell,j, redshift):
    return Tz(z_s)**2 * integrate.quad(noise_denominator_integrand, 0, l_max, args=(ell, j,  redshift))[0]
    #return Tz(z_s)**2 * integrate.nquad(noise_denominator_integrand, [(ell, l_max), (ell, l_max)] , args = (ell, j, redshift), opts = [{'limit' : 50, 'epsabs' : 1.49e-11, 'epsrel' : 1.49e-11}, {'limit' : 50, 'epsabs' : 1.49e-11, 'epsrel' : 1.49e-11}])[0] / 100000
"""
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
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

mass_moment_1 = exp.mass_moment_1
mass_moment_2 = exp.mass_moment_2
mass_moment_3 = exp.mass_moment_3
mass_moment_4 = exp.mass_moment_4
eta_bar = exp.eta_bar


### depth (in Mpc) of the observed volume along the line of site
#-- source: ZZ_2006
Len = 1.2 * (bandwidth / 0.1) * ((1 + z_s) / 10)**0.5 * (0.15 / omegamh2)**0.5 * h

### the D^2 L factor in equation 4 of P_2014
D2xLen = distance(z_s)**2 * Len
eta_D2_L = eta_bar * D2xLen

### constant factor ouside the integral in equation 20 of P_2014
constfac = (9/4) * (H0/c)**3 * omegam**2

### distance between source and observer
dist_s = distance(z_s)

#### thermal/instrumental noise
#-- C_l^N in eq. 14 of P_2014
#-- c_lj^tot = C_lj + C_l^N)
C_l_N = (2 * 3.14)**3 * T_sys**2 / (bandwidth * t_obs * f_cover**2 * l_max**2)


Cshot = (4 * 10**6 * 2 * 3.14**2 / 0.98**3) / D2xLen


two_pi_squared = (2 * 3.14)**2
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.arange(l_plot_min, l_plot_max)
N_L = np.zeros(n_err_points)

noise_denominator_sum = np.zeros(n_err_points)
N0_sum = np.zeros(n_err_points)
N1_sum = np.zeros(n_err_points)
N2_sum = np.zeros(n_err_points)
N3_sum = np.zeros(n_err_points)
N4_sum = np.zeros(n_err_points)

C_l_array = np.zeros((int(2**0.5 * l_ul), j_max))
N4_array = np.zeros(n_err_points)
delta_C_L = np.zeros(n_err_points)

#------------------------------------------------------------------------------

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
k = np.linspace(10**(-5), k_max(l_ul, z_s) ,50000)
#print()
PK = get_matter_power_interpolator(pars, nonlinear=False, kmax = np.max(k), k_hunit = False, hubble_units = False)
#-----------------------------------------------------------------------

###############################################################################
# Write data
###############################################################################
plot_err = open('./text-files/plt_err_j_max_{}_lmax_{}.txt'.format(j_max-1, l_ul), 'w')
plot_curve = open('./text-files/plt_j_max_{}_lmax_{}.txt'.format(j_max-1, l_ul), 'w')

igd = open('./text-files/igd_j_max_{}_lmax_{}.txt'.format(j_max-1, l_ul), 'w')

for L in tqdm(range(l_plot_min, l_plot_max)):
    plot_curve.write('{}    {}\n'.format(L, C_l(L)))
plot_curve.close()

for L in tqdm(range(l_plot_min, int(l_ul * 2 **0.5))):
    for j in range(j_min, j_max):
        C_l_array[L, j] = C_lj(L, j)

L_counter = 0
for L in range(l_plot_min, l_plot_max, err_stepsize):

    res_j = 0
    for j in tqdm(range(j_min, j_max)):
        res_sml = 0
        for l1 in tqdm(range(0, l_ul)):
            for l2 in range(0, l_ul):
                res_sml = res_sml + noise_denominator_integrand(l1, l2, L, j)
        res_j = res_j + res_sml

        #noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j)
        N3_for_this_L_j = N3(L, j)
        N1_sum[L_counter] = N1_sum[L_counter] + N1(L, j, N3_for_this_L_j)
        N2_sum[L_counter] = N2_sum[L_counter] + N2(L, j, N3_for_this_L_j)
        N3_sum[L_counter] = N3_sum[L_counter] + N3_for_this_L_j
        N4_sum[L_counter] = N4_sum[L_counter] + N4(L, j)

    igd.write('L {} N_den {}'.format(L, res_j**2))
    print('L {} N_den {}'.format(L, res_j**2))

    noise_denominator_sum[L_counter] = res_j

    # lensing reconstruction noise N_L(L); equation 14 in Pourtsidou et al. 2014
    N_L[L_counter] = (L**2 * (N0(L) + N1_sum[L_counter] + N2_sum[L_counter] + N3_sum[L_counter] + N4_sum[L_counter]) ) / (noise_denominator_sum[L_counter]**2)
    
    # error bars: delta_C_L(L); equation 21 in Pourtsidou et al. 2014
    delta_C_L[L_counter] = ( 2 / ((2*L + 1) * delta_l * f_sky) )**0.5 * (C_l(L) + N_L[L_counter])

    plot_err.write('{}  {}  {}  {}\n'.format(L, C_l(L), delta_C_L[L_counter], N_L[L_counter]))

    L_counter = L_counter + 1
plot_err.close()
igd.close()
#------------------------------------------------------------------------------

L, CL = np.loadtxt('./text-files/plt_j_max_{}_lmax_{}.txt'.format(j_max - 1, l_ul), unpack=True)
plt.plot(L, CL, label = 'Convergence APS (z = {})'.format(z_s), color = 'blue')

L_err, CL_err, delta_CL, N_L = np.loadtxt('./text-files/plt_err_j_max_{}_lmax_{}.txt'.format(j_max - 1, l_ul), unpack=True)

plt.plot(L_err, N_L, label='N(L)')

for i in range(n_err_points):
    plt.errorbar(L_err[i],  CL_err[i], yerr = delta_CL[i], capsize=3, ecolor='black')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$')
plt.ylabel(r'$C_L$')
plt.legend()
plt.xlim(l_plot_min, l_plot_max)
plt.savefig('./plots/full_plot_j_max_{}_lmax_{}.pdf'.format(j_max - 1, l_ul))
plt.show()


