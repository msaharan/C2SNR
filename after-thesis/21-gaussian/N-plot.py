import numpy as np
import matplotlib.pyplot as plt

L, dCL, CL, NL = np.loadtxt("./text-files/plt_err_j_upp_limit_60_lmax_50000.txt", unpack=True)
plt.plot(L, NL)
plt.xlabel("L")
plt.ylabel("$N_L$")
#plt.xscale("log")
#plt.yscale("log")
plt.title("z $\sim$ 6")
plt.savefig("./plots/NL-plot.pdf")
plt.show()
