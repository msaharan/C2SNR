import numpy as np
import cosmolopy.distance as cd
from tqdm.auto import tqdm

def distance(redshift):
    return cd.comoving_distance(redshift, **cosmo)

fileout = open("./comoving_distance_between_redshift_0_and_2.txt", 'w')

z_array = np.arange(0, 2, 0.0001)
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678}

for z in tqdm(z_array):
    fileout.write('{}   {}\n'.format(z, distance(z)))
fileout.close()
