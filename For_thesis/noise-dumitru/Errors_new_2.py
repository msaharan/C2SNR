import numpy as np
import matplotlib.pyplot as plt
#import Sensitivity
#import interspec
from scipy.interpolate import interp1d

def calc_Error_2( PS,k, z):
  
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=6.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=10.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=400.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.031 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1000.*3600. #1500hr CONCERTO - 1000hr Stage2
    
    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
#    delta_k=k[2]-k[1]
    delta_k = k 
    N_m=(V_s*delta_k*(k**2))/(4.*3.141*3.141)
    error=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))/np.sqrt(N_m)
    print(error)
    return error

def calc_Error_1( PS,k, z):
  
    V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
    aperture=12.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=2.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=1500.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.155 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1500.*3600. #1500hr CONCERTO - 1000hr Stage2
    
    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    sigma_pix=NEFD/Omega_beam
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    delta_k=k[2]-k[1]
    #delta_k = k 
    N_m=(V_s*delta_k*(k**2))/(4.*3.141*3.141)
    error=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))/np.sqrt(N_m)
    print(error)
    return error
    
pc2, kc2 = np.loadtxt("/home/dyskun/Documents/Utility/Git/Msaharan/C2SNR/C2/data-files/ps_CII_z6.txt", unpack = True) 
"""
for i in range(np.size(kc2)):
    print('k = {}, P = {}'.format(kc2[i], pc2[i]))
"""

####  shot noise
# ~ 4 * 10**6 at k = 0.92 for z = 6.349
shot = 4 * 10**6 * 2 * 3.14 * 3.14 / 0.92**3
plt.figure(figsize = (6,6))
plt.errorbar(kc2,  pc2 , yerr=calc_Error_1(pc2, kc2, 6.349))
plt.plot(kc2, kc2**3 * shot / (2 * 3.14 * 3.14))
plt.xlim(1E-1, 1E1)
plt.ylim(1E3, 1E9)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k[h/cMpc]')
plt.ylabel('$\Delta^2_{[CII]}(k) \;[Jy/Sr]^2$')
plt.savefig('./plots/c2-powspec.pdf', bbox_inches='tight')
plt.show()


