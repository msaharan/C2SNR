### P_2014 refers to Pourtsidou et al. 2014

#-- parameters are taken from
#-- Planck 2013
#-- table 2; Planck+WP; 68% limit

from scipy.special import gamma

h = 0.673
omegab = 0.02205 / h**2
omegac = 0.1199 / h**2
omegam = 0.315
omegal = 0.685
ns = 0.9603
As = 2.196 * 10**(-9)
sigma_8 = 0.829
YHe = 0.24770 #-- Y_p in the paper


#-- parameters stated below equation in Pourtsidou et al. 2014

T_sys = 50 #-- kelvin 
t_obs = 2 * 365 * 24 * 3600 #-- seconds
bandwidth = 40 #-- MHz
l_max = 19900
f_cover = 0.06
delta_l = 36
f_sky = 0.2
omegaH1 = 0.7

## To calculate Tz(z)
#-- Refer to eq. 1 and eq. 17-19 of P_2014
alpha = -1.3
phi_star = 0.0204 * h**3 #-- Mpc^(-3)
m_star = 3.47 * 10**9 / h**2 #-- solar mass
rho_c = 2.7755 * h**2 * 10**11

### mean galaxy number density 
#-- defined near equation 9 of P_2014
eta_bar = phi_star * gamma(alpha + 1)

### mean and relative HI galaxy mass density 
#-- equation 18 in P_2014
#-- mean
rhoHI = phi_star * m_star * gamma(alpha + 2)
#-- relative
omegaH1 = rhoHI / rho_c

### mass moments
#-- defined near eq. 9 of P_2014 
#-- Required in equation 14 of P_2014
mass_moment_1 = phi_star * m_star * gamma(alpha + 2) / eta_bar
mass_moment_2 = phi_star * m_star**2 * gamma(alpha + 3) / eta_bar
mass_moment_3 = phi_star * m_star**3 * gamma(alpha + 4) / eta_bar
mass_moment_4 = phi_star * m_star**4 * gamma(alpha + 5) / eta_bar

"""
import matplotlib.pyplot as plt
import numpy as np

#-- behaviour of gamma function
x = np.arange(-5, 5, 0.1)
print(x)
print(gamma(x))
plt.plot(x, gamma(x))
plt.ylim(-20, 50)
plt.show()

print(gamma(alpha + 2))
print(gamma(alpha + 3))
print(gamma(alpha + 4))
print(gamma(alpha + 5))

print("eta_bar = {}".format(eta_bar))
print("rhoHI = {}".format(rhoHI))
print("omegaH1 = {}".format(omegaH1))

### values of mass moments
print("MM1 = {}".format(mass_moment_1))
print("MM2 = {}".format(mass_moment_2))
print("MM3 = {}".format(mass_moment_3))
print("MM4 = {}".format(mass_moment_4))

### ratio of mass moments
print("MM4 / MM1**4 = {}".format(mass_moment_4 / mass_moment_1**4))
print("MM3 / MM1**3 = {}".format(mass_moment_3 / mass_moment_1**3))
print("MM2 / MM1**2 = {}".format(mass_moment_2 / mass_moment_1**2))
"""
