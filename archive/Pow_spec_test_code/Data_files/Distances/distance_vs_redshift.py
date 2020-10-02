import numpy as np                                                                           
import cosmolopy.distance as cd
import matplotlib.pyplot as plt
import matplotlib
from tqdm.auto import tqdm

# Angular diameter distance (Units: Mpc)

def a_distance(var):
    return cd.angular_diameter_distance(var, **cosmo)

# Comoving distance (Units: Mpc)

def c_distance(var):
    return cd.comoving_distance(var, **cosmo)


# Cosmology 
# Parameters used are taken from Planck 2013

omegam0 = 0.315
omegal = 0.685
h = 0.673
cosmo = {'omega_M_0': omegam0, 'omega_lambda_0': omegal, 'omega_k_0': 0.0, 'h': h}

# Write the data into files so that we can read them later to make the plot

f_a = open("ang_dia_dist_vs_z.txt", 'w')
f_c = open("comov_dist_vs_z.txt", 'w')

z = np.arange(0, 10, 0.001)
for n in tqdm(z):
   f_a.write('{}  {}\n'.format(n, a_distance(n))) 
   f_c.write('{}  {}\n'.format(n, c_distance(n))) 
f_a.close()
f_c.close()

'''
# Read files and make the plots

red, a_dist = np.loadtxt('ang_dia_dist_vs_z.txt', unpack = True)
red, c_dist = np.loadtxt('comov_dist_vs_z.txt', unpack = True)

fig, ax = plt.subplots()
plt.xscale('log')
plt.yscale('log')


plt.plot(red, a_dist, label = 'Ang. Dia. Distance')
plt.plot(red, c_dist, label = 'Comoving Distance')
plt.xlabel("Redshift (z)")
plt.ylabel("Distance in Mpc")
plt.title("Comparison of distance measures")
plt.grid(True)
plt.savefig("distances_in_cosmology.pdf")
plt.legend()
plt.show()
'''
