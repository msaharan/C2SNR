import numpy as np                                                                           
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm



def delta_C_L(L):
    #return  (2 / ((2*L + 1) * delta_L * f_sky) )**0.5 * np.interp(L, x, y) 
    #return  L * (L + 1) * (2 / ((2*L + 1) * delta_L * f_sky) )**0.5 * np.interp(L, x, y) / (2 * 3.14)
    return  4 * (2 / ((2*L + 1) * delta_L * f_sky) )**0.5 * np.interp(L, x, y) / (2 * 3.14)

delta_L = 36
f_sky = 0.2

x = np.zeros(50)
y = np.zeros(50)
x, y, junk, junk, junk, junk = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)


for n in range(18, 523, 36):
    print('L = {}, C_L = {}, delta_C_L = {}'.format(n, np.interp(n, x, y), delta_C_L(n)))

"""
plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
plt.show()
"""
