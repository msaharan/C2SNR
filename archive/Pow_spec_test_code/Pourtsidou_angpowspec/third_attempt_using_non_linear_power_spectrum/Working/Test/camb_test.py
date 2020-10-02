import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import camb
from camb import model, initialpower, get_matter_power_interpolator

K, PS = np.loadtxt("/home/dyskun/Documents/Utility/Academics/Cosmology_project/C2SNR/Pow_spec_test_code/Data_files/CAMB_non_linear.txt", unpack = True)

nz = 100
kmax = 2
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.02205, omch2=0.1199)
pars.InitPower.set_params(As=2.196e-9, ns=0.9624)
pars.set_for_lmax(2500, lens_potential_accuracy=1);

results = camb.get_background(pars)
PK = get_matter_power_interpolator(pars, nonlinear = True, kmax = 2)

plt.figure(figsize=(8,5))
#k=np.exp(np.log(10)*np.linspace(-4,2,200))
k=np.exp(np.log(10)*np.linspace(-4,2,200))
zplot = [0]

for z in zplot:
    plt.loglog(k, PK.P(z,k), label = "Offline, z = {}".format(z))
#print(PK.P(0, k)[0])
plt.loglog(K, PS, label = "Online")
plt.xlim([1e-4,kmax])
plt.xlabel('k Mpc')
plt.ylabel('$P(K), Mpc^{-3}$') # Check Units
plt.legend()

plt.show()
