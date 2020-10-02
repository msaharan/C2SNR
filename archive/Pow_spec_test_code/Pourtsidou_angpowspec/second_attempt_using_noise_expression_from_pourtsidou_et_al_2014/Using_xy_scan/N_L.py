import numpy as np                                                                           
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm

def N_L(L, counter):
    return  (error[counter] *  2 * 3.14 / (L * (L + 1) * (2 / ((2*L + 1) * delta_L * f_sky) )**0.5 )) - (y[counter] * 2 * 3.14 / (L * (L + 1)))

delta_L = 36
f_sky = 0.2

x = np.zeros(50)
y = np.zeros(50)
x, y, junk, junk, error, junk = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_curve_z_2.txt", unpack=True)

counter = 0
for n in range(18, 523, 36):
    print('L = {}, C_L (L(L+1)/2Pi) = {}, N_L = {}'.format(n, y[counter], N_L(n, counter)))
    counter = counter + 1



