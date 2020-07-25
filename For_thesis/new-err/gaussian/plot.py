import numpy as np
import matplotlib.pyplot as plt

l, nl = np.loadtxt('noise.txt', unpack=True)

plt.xlabel('L')
plt.ylabel('N(L)')
plt.loglog(l, nl)
plt.show()

