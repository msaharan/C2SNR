### P_2014 refers to Pourtsidou et al. 2014

#-- parameters are taken from
#-- Planck 2013
#-- table 2; Planck+WP; 68% limit

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
m_halo_min = 10**8 #-- solar mass; integration variable; minimum
m_halo_max = 10**12 #-- solar mass; integration variable; maximum
"""
mass_moment_1
mass_moment_2
mass_moment_3
mass_moment_4
"""
alpha = -1.3
phi_star = 0.0204 * h**3 #-- Mpc^(-3)
m_star = 3.47 * 10**9 / h**2 #-- solar mass

