import numpy as np
import matplotlib.pyplot as plt

L_g_700, N_L_g_700 = np.loadtxt("../Gaussian/J_files/Without_legend/gaussian_integration_over_distance_j_upp_limit_1_lmax_700.txt", unpack=True)
#L_g_10000, N_L_g_10000 = np.loadtxt("../Gaussian/J_files/Without_legend/gaussian_integration_over_distance_j_upp_limit_1_lmax_10000.txt", unpack=True)


L_p_eta0, N_L_p_eta0 = np.loadtxt("../Poisson/poisson_noise_eta_0.txt", unpack = True)
#L_p_eta2, N_L_p_eta2 = np.loadtxt("../Poisson/poisson_noise_eta_2.txt", unpack = True)

L_gp_700_eta0, N_L_gp_700_eta0 = np.loadtxt("../Gaussian_plus_poisson/J_files/gaussian_plus_poisson_integration_over_distance_j_upp_limit_1_lmax_700_eta_0.txt", unpack=True)
#L_gp_10000_eta0, N_L_gp_10000_eta0 = np.loadtxt("../Gaussian_plus_poisson/J_files/gaussian_plus_poisson_integration_over_distance_j_upp_limit_1_lmax_10000_eta_0.txt", unpack=True)

noise_array_min = [np.amin(N_L_g_700), np.amin(N_L_p_eta0), np.amin(N_L_gp_700_eta0)]
noise_array_max = [np.amax(N_L_g_700), np.amax(N_L_p_eta0), np.amax(N_L_gp_700_eta0)]

plt.subplots()
plt.ylim(np.amin(noise_array_min)/10, np.amax(noise_array_max)*10)
plt.plot(L_g_700, N_L_g_700, label = '(Gaussian)', marker = 'o')
plt.plot(L_p_eta0, N_L_p_eta0, label = '(Poisson)', marker = '+')
plt.plot(L_gp_700_eta0, N_L_gp_700_eta0, label = '(Gaussian + Poisson)', marker = 6)
plt.legend()
plt.title("Lensing Reconstruction Noise")
plt.xlabel("$L$")
plt.ylabel('$N_L(L)$')
plt.yscale('log')
plt.xscale('log')
plt.savefig('error_comparison_to_demonstrate.pdf')
plt.show()
