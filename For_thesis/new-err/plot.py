import numpy as np
import matplotlib.pyplot as plt

lp, nlp = np.loadtxt('./poisson/noise.txt', unpack=True)
lg, nlg = np.loadtxt('./gaussian/noise.txt', unpack=True)
#lb, nlb = np.loadtxt('./both/noise.txt', unpack=True)

plt.xlabel('L')
plt.ylabel('N(L)')
plt.loglog(lp, nlp, label='Poisson')
plt.loglog(lg, nlg, label='Gaussian')
#plt.loglog(lb, nlb, label='Gaussian + Poisson')
plt.legend()
plt.show()

