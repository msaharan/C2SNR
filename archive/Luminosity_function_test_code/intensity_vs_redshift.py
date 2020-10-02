import matplotlib.pyplot as plt
import numpy as np
import hmf
from hmf.integrate_hmf import hmf_integral_gtm

def hubble_parameter(z):
    return H0 * (omega_m * (1 + z)**3 + omega_lambda)**0.5

######################
# Constants
#####################
luminosity_of_sun = 3.8 * 10**26 # Watt
volume = 160**3 # (c Mpc / h)^3
c = 3 * 10**8 
c2_line_frequency  = c / (158 * 10**-6) # s^(-1)
omega_m = 0.308
omega_lambda = 0.692
H0 = 67.8 * 10**3 / (3.086 * 10**22) # s^(-1)
#--------------------

#########################################################
# Array definitions
#########################################################
redshift = np.arange(6,10.5,0.1)
halo_mass_function = hmf.MassFunction(Mmin = 8, Mmax = 13, sigma_8 = 0.829, n = 0.961)
star_formation_rate_density_at_z = np.zeros(len(redshift))
star_formation_rate_at_z =  np.zeros(len(redshift))
log_luminosity_at_z = np.zeros(len(redshift))
luminosity_at_z = np.zeros(len(redshift))
intensity_at_z = np.zeros(len(redshift))
#------------------------------------------------------------

###############################################################
# Data points from the top panel of figure 1 of Dumitru et al. 2018
###############################################################
data_redshift, data_sfrd = np.loadtxt("sfrd_data_points_dumitru_fig_1.txt", unpack=True)
data_sfrd = 10**data_sfrd
#---------------------------------------------------------------

##################################################################
# Finding f-star
###################################################################
for n in range(0, len(redshift)):
    halo_mass_function.update(z = redshift[n])
    halo_mass = halo_mass_function.m
    star_formation_rate_density_at_z[n] = (10**-12)*np.trapz(halo_mass * halo_mass_function.dndm, halo_mass)
    star_formation_rate_at_z[n] = star_formation_rate_density_at_z[n] * volume

"""
plt.subplots()
plt.plot(redshift, star_formation_rate_density_at_z)
plt.scatter(data_redshift, data_sfrd, color='red')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('SFRD ($ M_\odot yr^{-1}Mpc^{-3} $)')
plt.ylim(1*10**-4.0, 1*10**-1.5)
plt.savefig("sfrd_vs_redshift.pdf")
plt.show()
"""
#--------------------------------------------------------------------

####################################################################
# Variation of Luminosity and Intensity with redshift
#####################################################################
for n in range(0,len(redshift)):
    log_luminosity_at_z[n] = (1.4 - 0.07 * redshift[n])*np.log10(star_formation_rate_at_z[n]) + 7.1 - 0.007 * redshift[n]
    luminosity_at_z[n] = 10**log_luminosity_at_z[n] * luminosity_of_sun
    intensity_at_z[n] = c * luminosity_at_z[n] / (4 * 3.14 * c2_line_frequency * hubble_parameter(redshift[n]))

plt.plot(redshift, intensity_at_z)
plt.yscale('log')
plt.xlabel('$z$')
plt.ylabel('Intensity (SI units)')
plt.savefig('intensity_vs_redshift.pdf')
plt.show()
#--------------------------------------------------------------------



