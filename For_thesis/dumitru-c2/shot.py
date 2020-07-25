import numpy as np
import matplotlib.pyplot as plt

p_sh6, k_sh6 = np.loadtxt("./shot_noise_z6.txt", unpack=True)
p_sh7, k_sh7 = np.loadtxt("./shot_noise_z7.txt", unpack=True)
p_sh8, k_sh8 = np.loadtxt("./shot_noise_z8.txt", unpack=True)
p_sh9, k_sh9 = np.loadtxt("./shot_noise_z9.txt", unpack=True)


p_si6, k_si6 = np.loadtxt("./CII_z6_sigma.txt", unpack=True)
p_si8, k_si8 = np.loadtxt("./CII_z8_sigma.txt", unpack=True)
p_si9, k_si9 = np.loadtxt("./CII_z9_sigma.txt", unpack=True)

plt.loglog(k_si6, p_si6, label = "CII PS z = 7")
plt.loglog(k_si8, p_si8, label = "CII PS z = 8")
plt.loglog(k_si9, p_si9, label = "CII PS z = 9")
plt.xlabel(r'$k[h/cMpc]$', size=13)
plt.ylabel(r'$\Delta^2_{CII}(k)[Jy/Sr]^2$', size=13)
#plt.loglog(k_sh6, k_sh6**3 * p_sh6, label = "Shot noise")
plt.legend()
plt.savefig("./c2-dum.pdf")
plt.show()
