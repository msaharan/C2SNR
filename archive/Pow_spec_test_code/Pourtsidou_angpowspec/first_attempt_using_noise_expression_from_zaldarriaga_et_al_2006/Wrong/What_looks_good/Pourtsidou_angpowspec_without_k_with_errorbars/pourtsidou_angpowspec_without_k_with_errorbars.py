import numpy as np
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth  
from scipy import interpolate

def integration(i,redshift):
    return integrate.quad(lambda x: np.power((xs[redshift] - x)/xs[redshift], 2)* np.interp(l[i]/x, kn,dkn),0,xs[redshift])[0]

def dblintegrand(i, l1, l2):
    small_l = int(np.sqrt(l1**2 + l2**2))
    
    return ((result[i]*i*small_l) + (result[abs(i-small_l)] *i* (i - small_l)))**2/((result[i] + (C_l/constantfactor[redshift])) * (result[abs(i-small_l)] + (C_l/constantfactor[redshift])))
"""
    if small_l < i:
        return ((result[i]*i*small_l) + (result[i-small_l] *i* (i - small_l)))**2/((result[i] + (C_l/constantfactor[redshift])) * (result[i-small_l] + (C_l/constantfactor[redshift])))

    elif small_l>=i:
        return 1
"""  
def dblintegration(i):
    return integrate.dblquad(lambda l1, l2: dblintegrand(i, l1, l2), 1, 14000, lambda l2: 1,lambda l2:14000)[0]


#constants
omegam0 = 0.308
omegal = 0.692
c = 3 * 10**8                                                                                
H0 = 73.8 * 1000                                                                      
cosmo = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'omega_k_0': 0.0, 'h': 0.678} 
C_l = 1.6*10**(-16)
delta_l = 36
f_sky = 0.2
l_upper_limit = 20000
l_plot_ll = 10
l_plot_ul = 700

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,l_upper_limit)
constantfactor = np.zeros(11)
result = np.arange(0,l_upper_limit) 
N_L = np.zeros(l_upper_limit)
dC_L = np.zeros(l_upper_limit)

#reading the data file
kn,dkn = np.loadtxt("../Data_files/CAMB_linear.txt", unpack=True)

fileout = open("pourtsidou_angpowspec_without_k_with_errorbars.txt","a")
plt.subplots()
for redshift in range(2,3):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration
    
    for i in range (l_plot_ll, l_plot_ul):
        print("--------------------------------result[]"+str(i))
        result[i] = integration(i, redshift)

    for i in range(20,700,50):
        print("-------------------------------dC_l"+str(i))
        N_L[i] = 2* (i**2) * (2*np.pi)**2 / dblintegration(i)
        dC_L[i] = np.sqrt(2/((2*i + 1)*delta_l* f_sky)) *((constantfactor[redshift]*result[i]) +  N_L[i]) 
        fileout.write("{}    {}\n".format(i,dC_L[i]))
        print("This is dC_L {}".format(dC_L[i]))
        print("")

    plt.errorbar(l[l_plot_ll:l_plot_ul],constantfactor[redshift]*result[l_plot_ll:l_plot_ul], yerr=dC_L[l_plot_ll:l_plot_ul], label='Mohit z = {}'.format(redshift)) 
    #plt.plot(l[l_plot_ll:l_plot_ul],constantfactor[redshift]*result[l_plot_ll:l_plot_ul], label='Mohit z = {}'.format(redshift)) 

fileout.close()
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("../Data_files/pourtsidou_xyscan_curve_z_2.txt", unpack=True)
"""
xn_2 = np.zeros(50)
yn_2 = np.zeros(50)
dxln_2 = np.zeros(50)
dxhn_2 = np.zeros(50)
dyln_2 = np.zeros(50)                                                   
dyhn_2 = np.zeros(50)
xn_2, yn_2, dxln_2, dxhn_2, dyln_2, dyhn_2 = np.loadtxt("pourtsidou_xyscan_noise_z_2.txt", unpack=True)
"""

xplot_2 = np.arange(10, x_2[int(np.size(x_2))-1], x_2[int(np.size(x_2))-1]/2000)
tck_2 = interpolate.splrep(x_2,y_2, s=0)
tckerr_2 = interpolate.splrep(x_2,dyh_2,s=0)
yploterr_2 = interpolate.splev(xplot_2, tckerr_2, der=0)
yplot_2 = interpolate.splev(xplot_2, tck_2, der=0)
plt.errorbar(xplot_2,yplot_2, yerr=yploterr_2,color='black', ecolor='yellow', label='Pourtsidou (z=2)')
#plt.plot(xplot_2,yplot_2,color='black', label='Pourtsidou z=2')



plt.xlabel('l')
plt.ylabel(r'$C_{l} L(L+1)/2\pi$')
plt.suptitle(r"Angular Power Spectrum (Using Linear Matter Pow. Spec)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(1E-9,1E-7)
plt.xlim(10,1000)
plt.savefig("pourtsidou_angpowspec_without_k_with_errorbars.pdf")
plt.show()
