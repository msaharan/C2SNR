import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Plot from Pourtsidou et al. 2014 (with error bars)
###############################################################################
x_2 = np.zeros(50)
y_2 = np.zeros(50)
dxl_2 = np.zeros(50)
dxh_2 = np.zeros(50)
dyl_2 = np.zeros(50)
dyh_2 = np.zeros(50)
x_2, y_2, dxl_2, dxh_2, dyl_2, dyh_2 = np.loadtxt("/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/Pow_spec_test_code/Data_files/pourtsidou_xyscan_z_2_no_errors.txt", unpack=True)

plt.plot(x_2,y_2,color='black', label='Pourtsidou et al. 2014 (z=2)')
#------------------------------------------------------------------------------

###############################################################################
# Cshot
###############################################################################
L, Cshot  = np.loadtxt("./Text_files/cshot.txt", unpack=True)

plt.plot(L, Cshot,color='black', label='Cshot')
#------------------------------------------------------------------------------

l_plot_low_limit = 10
l_plot_upp_limit = 600
err_stepsize = 36
n_err_points = 1 + int((l_plot_upp_limit - l_plot_low_limit)/err_stepsize)

j_upp_limit = 2
l_max = 600

L, CL = np.loadtxt('./Text_files/plt_integration_over_redshift_j_upp_limit_{}_lmax_{}_eta_0.txt'.format(j_upp_limit - 1, l_max), unpack=True)
plt.plot(L, CL, label = 'This Work (z = 2)', color = 'blue')

L_err, CL_err, delta_CL, N_L = np.loadtxt('./Text_files/plt_err_integration_over_redshift_j_upp_limit_{}_lmax_{}_eta_0.txt'.format(j_upp_limit - 1, l_max), unpack=True)

plt.plot(L_err, N_L, label="N(L)")

"""
for i in range(n_err_points):
    plt.errorbar(L_err[i], CL_err[i], yerr = delta_CL[i], capsize=3, ecolor='blue')
"""
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$L$')
plt.ylabel(r'$(L(L + 1)/2\pi)\; C_L$')
plt.legend()
plt.xlim(l_plot_low_limit, l_plot_upp_limit)
plt.show()
