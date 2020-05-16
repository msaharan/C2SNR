import matplotlib.pyplot as plt
import numpy as np
import hmf
from hmf.integrate_hmf import hmf_integral_gtm

data_redshift, data_sfrd = np.loadtxt("sfrd_data_points_dumitru_fig_1.txt", unpack=True)
data_sfrd = 10**data_sfrd
redshift = np.arange(6,10.5,0.1)
halo_mass_function = hmf.MassFunction(Mmin = 8, Mmax = 13, sigma_8 = 0.829, n = 0.961)
star_formation_rate_density_at_z = np.zeros(len(redshift))

for n in range(0, len(redshift)):
    halo_mass_function.update(z = redshift[n])
    halo_mass = halo_mass_function.m
    star_formation_rate_density_at_z[n] = (10**-12)*np.trapz(halo_mass * halo_mass_function.dndm, halo_mass)
"""
plt.subplots()
plt.plot(redshift, star_formation_rate_density_at_z)
plt.scatter(data_redshift, data_sfrd, color='red')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('SFRD ($ M_\odot yr^{-1}Mpc^{-3} $)')
plt.ylim(1*10**-4.0, 1*10**-1.5)
plt.savefig("sfrd.pdf")
plt.show()
"""

luminosity_at_z = np.zeros(len(halo_mass))

for n in range(0,len(halo_mass)):
    luminosity_at_z[n] = (10**(1.4 - 0.07*redshift[0] + 7.1 - 0.07*redshift[0]) * (10**-12)) * halo_mass[n]

plt.plot(halo_mass, luminosity_at_z)
plt.show()

