import numpy as np
import cosmolopy.distance as cd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm
import camb
from camb import model, initialpower, get_matter_power_interpolator

f = open("camb_parameters_pars.txt", 'w')
f1 = open("camb_parameters_results.txt", 'w')
pars = model.CAMBparams(NonLinear = 1, WantTransfer = True, H0=67.3, omch2=0.1199, ombh2=0.02205, YHe = 0.24770)
pars.DarkEnergy.set_params(w=-1.13)
pars.set_for_lmax(lmax=20000)
pars.InitPower.set_params(ns=0.9603, As = 2.196e-09)

f.write('print(pars)\n\n')
f.write(str(pars))
f.write('\n\n')

f.write('------------------------------------------------------------------\n')

results = camb.get_background(pars)
f1.write('print(results)\n\n')
f1.write(str(results))

f.close()
f1.close()

