import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
"""
jmax = 61
lmax = 50000
z = 6.349
L, j, clj = np.loadtxt('./text-files/clfile_j_{}_l_{}_z_{}.txt'.format(jmax - 1, lmax, z), unpack=True)
inp = interpolate.interp2d(L, j, clj, kind='cubic')
"""
l_fil, j_fil, clj = np.loadtxt('./text-files/clfile_j_61_l_50001_z_6.349.txt', unpack=True)
inp = interpolate.interp2d(l_fil, j_fil, clj, kind='cubic')
for i in range(0, 1):
    print('L {} j {} clj {}\n'.format(l_fil[i], j_fil[i], clj[i]) )
print('')
print(inp(400, 6))
