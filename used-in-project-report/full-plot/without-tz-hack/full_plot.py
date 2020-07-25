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


##############################################################################
# Comoving distance between z = 0 and some redshift
###############################################################################
def distance(var):
    return cd.comoving_distance(var, **cosmo)
#------------------------------------------------------------------------------

###############################################################################
# Constant factor; Integration over redshift
###############################################################################
def constantfactor(redshift):
   return (9/4) * (H0/c)**3 * omegam0**2
#------------------------------------------------------------------------------

###############################################################################
# Hubble ratio H(z)/H0
###############################################################################
def hubble_ratio(var):
    return (omegam0 * (1 + var)**3 + omegal)**0.5
#------------------------------------------------------------------------------

###############################################################################
# cshot
###############################################################################
"""
def CShot(ell, redshift):
    # cshot * k**3 / (2 Pi**2)  = 4 * 10**6 at k = 0.98 h/Mpc
    return 4 * 10**6 * 2 * 3.14**2 / 0.98**3
"""
cshot = 4 * 10**6 * 2 * 3.14**2 / 0.98**3
#------------------------------------------------------------------------------

###############################################################################
## P_[CII]^N
###############################################################################
def pc2tot_s2( PS,k, z):
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=6.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=10.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=400.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.031 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1000.*3600. #1500hr CONCERTO - 1000hr Stage2

    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    #ptot=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))
    #ptot = (PS + PN_CII) * k**3 / (2 * 3.14**2)
    ptot = (PS + PN_CII)
    return ptot

def pc2tot_s1( PS,k, z):

    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=12.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=2.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=1500.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.155 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1500.*3600. #1500hr CONCERTO - 1000hr Stage2

    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    #ptot=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))
    #ptot = (PS + PN_CII) * k**3 / (2 * 3.14**2)
    ptot = (PS + PN_CII)

    return ptot

def pc2n_s2( PS,k, z):
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=6.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=10.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=400.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.031 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1000.*3600. #1500hr CONCERTO - 1000hr Stage2

    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    return PN_CII

def pc2n_s2( PS,k, z):
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=12.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=2.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=1500.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.155 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1500.*3600. #1500hr CONCERTO - 1000hr Stage2

    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    return PN_CII
#------------------------------------------------------------------------------

###############################################################################
## Integration over redshift
###############################################################################
def angpowspec_integrand_without_j(z, ell, dist_s, constf):
    dist = distance(z)
    return constf * (1 - dist/dist_s)**2 * PK.P(z, ell/dist) / hubble_ratio(z)

def angpowspec_integration_without_j(ell, redshift):
    constf = constantfactor(redshift)
    dist_s = distance(redshift)
    return integrate.quad(angpowspec_integrand_without_j, 0.0001, redshift, args = (ell, dist_s, constf), limit=5000)[0]

def c2_lj(z, ell, j):
    dist_s = distance(z)
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return np.interp(k_var, k_file, p_file) / D2_L

def c2_tot_lj(z, ell, j):
    dist_s = distance(z)
    k_var = ((ell/dist_s)**2 + (2 * 3.14 * j / dist_s)**2 )**0.5
    return pc2tot_s1(np.interp(k_var, k_file, p_file), k_var, z) / D2_L
#------------------------------------------------------------------------------

###############################################################################
## To calculate the noise moments (equation 14 in Pourtsidou et al. 2014)
###############################################################################
def N4_integrand(l1, l2, ell, j):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2 * c2_tot_lj_array[ell, j] * c2_tot_lj_array[abs(ell-small_l), j] / two_pi_squared

def N4(ell, j):
    return integrate.nquad(N4_integrand, [(0, l_max), (0, l_max)], args = (ell, j), opts = [{'limit' : 40000}, {'limit' : 40000}])[0]

def N3_integrand(l1, l2, ell, j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return cshot * 2 * (c2_tot_lj_array[ell, j] + c2_tot_lj_array[abs(ell-small_l),j]) / two_pi_squared

def N3(ell, j, redshift):
    return integrate.nquad(N3_integrand, [(0, l_max), (0, l_max)], args = (ell, j, redshift), opts = [{'limit' : 40000}, {'limit' : 40000}])[0]

def N2(ell, j, N3_for_this_L_j, redshift):
    return mass_moment_2**2 * N3_for_this_L_j / (2 * eta_D2_L**2 * mass_moment_1**4 * cshot)

def N1(ell, j, N3_for_this_L_j, redshift):
    return j_max * mass_moment_3 * (l_max**2 / two_pi_squared) * N3_for_this_L_j / (eta_D2_L**2 * mass_moment_1**3 * cshot)

def N0(ell, redshift):
    return j_max**2 * mass_moment_4 * (l_max**2 / two_pi_squared)**2 / (eta_D2_L**3 * mass_moment_1**4)
#------------------------------------------------------------------------------

###############################################################################
# To calculate the denominator of the lensing recosntruction noise term (eq 14 in Pourtsidou et al. 2014)
###############################################################################
def noise_denominator_integrand(l1, l2, ell,j, redshift):
    small_l = int(np.sqrt(l1**2 + l2**2))                                                    
    #return (c2_lj_array[ell, j] * ell * small_l + c2_lj_array[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * cshot(ell/distance(redshift))) / two_pi_squared
    return (c2_lj_array[ell, j] * ell * small_l + c2_lj_array[abs(ell-small_l), j] * ell * (ell - small_l) + ell**2 * cshot) / two_pi_squared

def noise_denominator_integration(ell,j, redshift):
    return integrate.nquad(noise_denominator_integrand, [(0, l_max), (0, l_max)] , args = (ell, j, redshift), opts = [{'limit' : 40000}, {'limit' : 40000}])[0] 
#------------------------------------------------------------------------------

##############################################################################
# Constants
###############################################################################
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8
h = 0.678
H0 = 100 * h *  1000 
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}
l_plot_low_limit = 10
l_plot_upp_limit = 600
err_stepsize = 36
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)
mass_moment_1 = 0.3
mass_moment_2 = 0.21
mass_moment_3 = 0.357
mass_moment_4 = 46.64
j_low_limit = 1
j_upp_limit = 41
#j_upp_limit = 2
j_max = j_upp_limit
two_pi_squared = 2 * 3.14 * 2 * 3.14
redshift = 6.349
delta_z = 0.5
D2_L = distance(redshift)**2 * abs(distance(redshift  + (delta_z / 2)) - distance(redshift - (delta_z / 2)))
eta_D2_L = 0.88 * h**3 * D2_L
f_sky = 0.2
delta_l = 36

#k_max = 10
#l_max = int(10 * h * distance(redshift))
l_max = 5000
k_max = np.ceil(l_max/distance(redshift))
k_lim = 1
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0 = h * 100, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax = l_max)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)
results = camb.get_background(pars)
k = 10**np.linspace(-2,k_lim,1000)
PK = get_matter_power_interpolator(pars, nonlinear=True, kmax = k_max, k_hunit = True, hubble_units = True)
#------------------------------------------------------------------------------

###############################################################################
# Arrays
###############################################################################
L_array = np.arange(0, l_plot_upp_limit)
angpowspec_without_j = np.zeros(int(l_max))
N_L = np.zeros(n_err_points)

noise_denominator_sum = np.zeros(n_err_points)
N0_sum = np.zeros(n_err_points)
N1_sum = np.zeros(n_err_points)
N2_sum = np.zeros(n_err_points)
N3_sum = np.zeros(n_err_points)
N4_sum = np.zeros(n_err_points)

angpowspec_without_j = np.zeros(int(2**0.5 * l_max))

angpowspec_without_j_signal_in_final_plot = np.zeros(l_plot_upp_limit)

c2_lj_array = np.zeros((int(2**0.5 * l_max), j_upp_limit))
c2_tot_lj_array = np.zeros((int(2**0.5 * l_max), j_upp_limit))
N4_array = np.zeros(n_err_points)
delta_C_L = np.zeros(n_err_points)
#------------------------------------------------------------------------------

###############################################################################
# CII power spectrum
###############################################################################
pk3_2pi_file, k_file = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6.txt', unpack=True)
p_file = pk3_2pi_file * 2 * 3.14**2 / k_file**3
#------------------------------------------------------------------------------

###############################################################################
# Write data
###############################################################################
plot_err = open("./text-files/plt_err_j_upp_limit_{}_lmax_{}.txt".format(j_upp_limit-1, l_max), "w")
plot_curve = open("./text-files/plt_j_upp_limit_{}_lmax_{}.txt".format(j_upp_limit-1, l_max), "w")
plot_cshot = open("./text-files/cshot.txt", "w")

for L in tqdm(range(1, int(l_max * 2**0.5))):
    for j in range(j_low_limit, j_upp_limit):
        c2_lj_array[L, j] = c2_lj(redshift, L, j)
        c2_tot_lj_array[L, j] = c2_tot_lj(redshift, L, j)

for L in tqdm(range(1, l_plot_upp_limit)):
    angpowspec_without_j[L] = angpowspec_integration_without_j(L, redshift) 
    plot_curve.write('{}    {}\n'.format(L, angpowspec_without_j[L]))
plot_curve.close()

L_counter = 0
for L in range(l_plot_low_limit + 8, l_plot_upp_limit, err_stepsize):
    for j in tqdm(range(j_low_limit, j_upp_limit)):

        noise_denominator_sum[L_counter] = noise_denominator_sum[L_counter] + noise_denominator_integration(L, j, redshift)
        print('before')
        N3_for_this_L_j = N3(L, j, redshift)
        print('after')
        N1_sum[L_counter] = N1_sum[L_counter] + N1(L, j, N3_for_this_L_j, redshift)
        N2_sum[L_counter] = N2_sum[L_counter] + N2(L, j, N3_for_this_L_j, redshift)
        N3_sum[L_counter] = N3_sum[L_counter] + N3_for_this_L_j
        N4_sum[L_counter] = N4_sum[L_counter] + N4(L, j)

    # lensing reconstruction noise N_L(L); equation 14 in Pourtsidou et al. 2014
    N_L[L_counter] = (L**2 * (N0(L, redshift) + N1_sum[L_counter] + N2_sum[L_counter] + N3_sum[L_counter] + N4_sum[L_counter]) ) / (noise_denominator_sum[L_counter]**2)
    
    # error bars: delta_C_L(L); equation 21 in Pourtsidou et al. 2014
    delta_C_L[L_counter] = ( 2 / ((2*L + 1) * delta_l * f_sky) )**0.5 * (angpowspec_without_j[L] + N_L[L_counter]) 

    plot_err.write('{}  {}  {}  {}\n'.format(L, angpowspec_without_j[L], delta_C_L[L_counter], N_L[L_counter]))

    plot_cshot.write('{}    {}\n'.format(L, cshot))
    L_counter = L_counter + 1
plot_err.close()
plot_cshot.close()
#------------------------------------------------------------------------------

###############################################################################
# Plot
###############################################################################
L, CL = np.loadtxt('./text-files/plt_j_upp_limit_{}_lmax_{}.txt'.format(j_upp_limit - 1, l_max), unpack=True)
plt.plot(L, CL, label = 'Convergence APS (z = 6.349)', color = 'blue')

L_err, CL_err, delta_CL, N_L = np.loadtxt('./text-files/plt_err_j_upp_limit_{}_lmax_{}.txt'.format(j_upp_limit - 1, l_max), unpack=True)

plt.plot(L_err, N_L, label="N(L)")

for i in range(n_err_points):
    plt.errorbar(L_err[i],  CL_err[i], yerr = delta_CL[i], capsize=3, ecolor='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$')
plt.ylabel(r'$C_L(L)$')
plt.legend()
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.savefig('./plots/full_plot_j_upp_limit_{}_lmax_{}.pdf'.format(j_upp_limit - 1, l_max))
plt.show()

