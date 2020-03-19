import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("pourtsidou_xyscan_curve_z_2.txt", unpack=True)

xn_2 = np.zeros(50)
yn_2 = np.zeros(50)
dxln_2 = np.zeros(50)
dxhn_2 = np.zeros(50)
dyln_2 = np.zeros(50)
dyhn_2 = np.zeros(50)
xn_2, yn_2, dxln_2, dxhn_2, dyln_2, dyhn_2 = np.loadtxt("pourtsidou_xyscan_noise_z_2.txt", unpack=True)


x_3 = np.zeros(50)
y_3 = np.zeros(50)
dxl_3 = np.zeros(50)
dxh_3 = np.zeros(50)
dyl_3 = np.zeros(50)
dyh_3 = np.zeros(50)
x_3, y_3, dxl_3, dxh_3, dyl_3, dyh_3 = np.loadtxt("pourtsidou_xyscan_curve_z_3.txt", unpack=True)

xn_3 = np.zeros(50)
yn_3 = np.zeros(50)
dxln_3 = np.zeros(50)
dxhn_3 = np.zeros(50)
dyln_3 = np.zeros(50)
dyhn_3 = np.zeros(50)
xn_3, yn_3, dxln_3, dxhn_3, dyln_3, dyhn_3 = np.loadtxt("pourtsidou_xyscan_noise_z_3.txt", unpack=True)




xplot_2 = np.arange(10, x_2[int(np.size(x_2))-1], x_2[int(np.size(x_2))-1]/2000)
tck_2 = interpolate.splrep(x_2,y_2, s=0)
tckerr_2 = interpolate.splrep(x_2,dyh_2,s=0)
yploterr_2 = interpolate.splev(xplot_2, tckerr_2, der=0)
yplot_2 = interpolate.splev(xplot_2, tck_2, der=0)
plt.errorbar(xplot_2,yplot_2, yerr=yploterr_2,color='black', ecolor='gray', label='z=2')


plt.plot(xn_2, yn_2, color = 'black', label=r'z = 2; $L^2 \times N_l$')

xplot_3 = np.arange(10, x_3[int(np.size(x_3))-1], x_3[int(np.size(x_3))-1]/2000)
tck_3 = interpolate.splrep(x_3,y_3, s=0)
tckerr_3 = interpolate.splrep(x_3,dyh_3,s=0)
yploterr_3 = interpolate.splev(xplot_3, tckerr_3, der=0)
yplot_3 = interpolate.splev(xplot_3, tck_3, der=0)
plt.errorbar(xplot_3,yplot_3, yerr=yploterr_3,color='red', ecolor='yellow', label='z=3')

plt.plot(xn_3, yn_3, color = 'red', label=r'z = 3; $L^2 \times N_l$')

plt.legend()
plt.xscale("log")
plt.xlabel(r'$l$')
plt.xlim(10,3000)
plt.ylim(1E-9, 1E-7)
plt.yscale("log")
plt.ylabel(r'$C_l\; l(l+1)/ (2\pi)$')
plt.suptitle("Pourtsidou's plot; reproduced using xyscan")
plt.savefig("pourtsidou_xyscan_plot.pdf")
plt.show()
