# To compare the evolution of Growth Factor and 'a' with 'z'

import numpy as np                                                                           
import math
import cosmolopy.distance as cd
import cosmolopy.perturbation as cp
import matplotlib.pyplot as plt
from cosmolopy.perturbation import fgrowth

zmin = 0
zmax = 20
redshift = np.arange(zmin,zmax)
a = np.zeros(zmax-zmin)
fgrow = np.zeros(zmax-zmin)

plt.subplots()
for z in redshift:
    a[redshift] = 1/ (1 + redshift)
    fgrow[redshift] = fgrowth(redshift, 0.308, unnormed=True)

# uncomment this section if you want to plt a(z) and fgrowth(z) vs z
'''
plt.plot(redshift, a, label = "a(z)")
plt.plot(redshift, fgrow, label = 'fgrowth(z)')
plt.xlabel("z")
plt.ylabel(" 'a' and Growth factor")
plt.suptitle("Evolution of 'a' and Growth factor with z \n Command: fgrowth(redshift, 0.308, unnormed=True)")
plt.legend()
plt.savefig('fgrowth_and_a_vs_z_True.pdf')
plt.show()
'''

#uncomment this section if you want to plot fgrowth(z) - a(z) vs z
'''
plt.plot(redshift, fgrow - a, label = 'fgrowth(z) - a(z)')
plt.xlabel("z")
plt.ylabel(" fgrowth(z) - a(z)")
plt.suptitle("Evolution of 'a' and Growth factor with z \n Command: fgrowth(redshift, 0.308, unnormed=True)")
plt.legend()
plt.savefig('fgrowth_and_a_vs_z_True_diff.pdf')
plt.show()
'''
