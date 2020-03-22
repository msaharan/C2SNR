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

def integrationk(i, k,redshift): 
    return integrate.quad(lambda x: np.power((xs[redshift] - x)/xs[redshift], 2)* np.interp(np.sqrt(2*((l[i]/x)**2) + k**2), kn,dkn),0,xs[redshift])[0]

def dblintegrand(i,k, l1, l2):
    small_l = int(np.sqrt(l1**2 + l2**2))
    return ((resultk[i,k]*i*small_l) + (resultk[abs(i-small_l),k] *i* (i - small_l)))**2/((resultk[i,k] + (C_l/constantfactor[redshift])) * (resultk[abs(i-small_l),k] + (C_l/constantfactor[redshift])))

def dblintegration(i,k):
    return integrate.dblquad(lambda l1,l2: dblintegrand(i,k, l1, l2),0 ,14000, lambda l2: 0,lambda l2:14000)[0]

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
err_stepsize = 200
n_err_points = 1 + int((l_plot_ul - l_plot_ll)/err_stepsize)
k_ll = 1
k_ul = 2

#array definitions
xs = np.zeros(11)
fgrow = np.zeros(11)
l = np.arange(0,l_upper_limit)
x_junk = np.zeros(l_upper_limit)
y_junk = np.zeros(l_upper_limit)
constantfactor = np.zeros(11)
result = np.arange(0,l_upper_limit) 
N_L = np.zeros(l_upper_limit)
dC_L = np.zeros(l_upper_limit)
resultk = np.zeros((l_upper_limit, k_ul))

#reading the data file
kn,dkn = np.loadtxt("../../Data_files/CAMB_linear.txt", unpack=True)

fileout = open("integration_with_error_plot_k_upto_2_l_upto_14000.txt", "a")

plt.subplots()

x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("../../Data_files/pourtsidou_xyscan_curve_z_2.txt", unpack=True)

xplot_2 = np.arange(10, x_2[int(np.size(x_2))-1], x_2[int(np.size(x_2))-1]/2000)
tck_2 = interpolate.splrep(x_2,y_2, s=0)
tckerr_2 = interpolate.splrep(x_2,dyh_2,s=0)
yploterr_2 = interpolate.splev(xplot_2, tckerr_2, der=0)
yplot_2 = interpolate.splev(xplot_2, tck_2, der=0)
plt.errorbar(xplot_2,yplot_2, yerr=yploterr_2,color='black', ecolor='yellow', label='Pourtsidou et al. 2014 (z=2)')


for redshift in range(2,3):
    constantfactor[redshift] = (9/4)* np.power(H0/c,4) * np.power(omegam0,2)*(fgrowth(redshift, 0.308, unnormed=False) * (1 + redshift))**2
    xs[redshift] = cd.comoving_distance(redshift, **cosmo) # upper limit of integration
    
    for i in range (l_plot_ll, l_plot_ul):
        print("--------------------------------result[]"+str(i))
        result[i] = integration(i, redshift)
        for k in range(k_ll, k_ul):
            resultk[i,k] = integrationk(i,k,redshift)

    for i in range(20,700,50):
        integ = 0
        integ_sum = 0
        print("-------------------------------i "+str(i))
        for k in range(k_ll, k_ul):
            integ = dblintegration(i,k)
            integ_sum = integ_sum + integ
        N_L[i] = 2* (i**2) * (2*np.pi)**2 / integ_sum
        dC_L[i] = np.sqrt(2/((2*i + 1)*delta_l* f_sky)) *((constantfactor[redshift]*result[i]) + N_L[i]) 
        plt.errorbar(l[i], constantfactor[redshift]*result[i], yerr=dC_L[i], capsize=3, ecolor='blue')
        fileout.write("{}   {}\n".format(i,dC_L[i]))
        print("This is dC_L {}\n".format(dC_L[i]))

    
plt.plot(l[l_plot_ll:l_plot_ul], constantfactor[redshift]*result[l_plot_ll:l_plot_ul], color='blue', label='This work (z = {})'.format(redshift))

fileout.close()

plt.xlabel('l')
plt.ylabel(r'$C_{l} L(L+1)/2\pi$')
plt.suptitle(r"Angular Power Spectrum (Using Linear Matter Pow. Spec)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(1E-9,1E-7)
plt.xlim(10,1000)
plt.savefig("integration_with_error_plot_k_upto_2_l_upto_14000.pdf")
plt.show()
