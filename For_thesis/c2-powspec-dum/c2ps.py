import numpy as np
import matplotlib.pyplot as plt

p6, k6 = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6.txt', unpack = True)
p7, k7 = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z7.txt', unpack = True)
p8, k8 = np.loadtxt('/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z8.txt', unpack = True)

p6 = p6 * 2 * 3.14**2 / k6**3
p7 = p7 * 2 * 3.14**2 / k7**3
p8 = p8 * 2 * 3.14**2 / k8**3
fs = 14
fig, ax = plt.subplots(figsize=(7,7))
plt.xlabel('$k [h/Mpc]$', fontsize = fs)
plt.ylabel('$P(k)\; [h^3\;Jy^2/ Mpc^3\;Sr^2]$', fontsize = fs)
plt.loglog(k6, p6, label = 'z = 6.349')
plt.loglog(k7, p7, label = 'z = 7.139')
plt.loglog(k8, p8, label = 'z = 8.15')
ax.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(fontsize = fs)
plt.savefig("./plots/c2dum.pdf", bbox_inches='tight')
plt.show()
