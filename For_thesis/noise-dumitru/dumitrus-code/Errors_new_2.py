import numpy as np
import matplotlib.pyplot as plt
#import Sensitivity
#import interspec
from scipy.interpolate import interp1d


def calc_Error( PS,k, z, flag1, flag2, PS_CII=0.0, PS_21=0.0):
  if flag1=='SKA':
    lamda1=21000.
    t_int1=6000.
    N_survey=866
    delta_nu1=0.00025
    baseline=40000.
    Aeff = 962.0
    d_antenna=962.
    bmax,plm = 40286.83, 10**5
    theta1= 3.14*0.21/(baseline*180)
  if flag1=='LOFAR':
    lamda1=21000.
    B_nu1=0.008
    Omega_S1=25.
    delta_nu1=0.00025
    t_int1=6000.
    Aeff = 526.0
    baseline=3400.
    N_survey=48.
    bmax,plm = 3475.584,10**6
    theta1= 3.14*0.21/(baseline*180)

  V_s=3.3*10**7*(10./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
  
  if flag2=='CII':
    aperture=6.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=10.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=400.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.031 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1000.*3600. #1500hr CONCERTO - 1000hr Stage2
    
    # aperture=12.0  #12m CONCERTO - 6m Stage 2
    # transmission=0.3
    # A_survey=2.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    # d_nu=1500.  #1.5GHz CONCERTO - 0.4 Stage 2
    # NEFD=0.155 #155mJy CONCERTO 31mJy Stage 2
    # N_pix=1500.
    # t_int=1500.*3600. #1500hr CONCERTO - 1000hr Stage2
    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    print(Omega_beam)
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    print(t_pix)
    sigma_pix=NEFD/Omega_beam
    print(sigma_pix)
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    print(V_pix_CII)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    print(PN_CII)
    #delta_k=k[2]-k[1]
    delta_k = k 
    N_m=(V_s*delta_k*(k**2))/(4.*3.141*3.141)
    error=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))/np.sqrt(N_m)

  if flag2=='cross':
    aperture=12.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=2. #2 deg^2 CONCERTO - 10 deg^2 Stage 2
    d_nu=1500.  #1.5GHz CONCERTO - 0.4 Stage 2
    NEFD=0.155 #155mJy CONCERTO 31mJy Stage 2
    N_pix=1500.
    t_int=1500.*3600. #1500hr CONCERTO - 1000hr Stage2
    theta_beam=1.22*158.0*(1+z)/(aperture*10**6)
    Omega_beam=2.*3.14*(theta_beam/2.355)**2
    print(Omega_beam)
    t_pix=t_int*N_pix*Omega_beam/(A_survey*3.14*3.14/(180.**2))
    print(t_pix)
    sigma_pix=NEFD/Omega_beam
    print(sigma_pix)
    V_pix_CII= 1.1*(10**3)*(((1.+z)/8.)**0.5)*(((theta_beam*180*60.)/(10.*3.14))**2)*(d_nu/400.)
    print(V_pix_CII)
    PN_CII=(sigma_pix**2)*V_pix_CII/t_pix
    print(PN_CII)
    delta_k=k[2]-k[1]
    N_m=(V_s*delta_k*(k**2))/(4.*3.141*3.141)
    bmin= 14.0# m
    wl = 0.21 * (1.0+z) # m
    umin, umax = bmin/wl, bmax/wl
    umax = bmax/wl
    c = 3.0e8# m
    nu = c/wl # Hz
    nu *= 1.0e-6 # MHz
    Tsys = 60.0*(300.0/nu)**2.55 # K
    sens_hera =Sensitivity.sens_Parsons2012(k, t_days=180.0, B=6.0, Tsys=Tsys,
                                N=N_survey, u_max=umax, Omega=wl**2/Aeff)
    error=np.sqrt(PS**2+(PS_21+sens_hera*np.sqrt(N_m))*(PS_CII+(PN_CII*V_pix_CII*(k**3)/(2.*3.14*3.14))))/np.sqrt(N_m)
    

  if flag2=='21':
    V_pix1=1.1*(10**3)*(21000./158.)*(((1.+z)/8)**0.5)*((theta1/10.)**2)*(delta_nu1/400)  #cMpc/h
    delta_k=k[3]-k[2]
    N_m=(V_s*delta_k*(k**2))/(4*3.141*3.141)
    bmin= 14.0 # m
    wl = 0.21 * (1.0+z) # m
    umin, umax = bmin/wl, bmax/wl
    umax = bmax/wl
    c = 3.0e8 # m
    nu = c/wl # Hz
    nu *= 1.0e-6 # MHz
    Tsys = 60.0*(300.0/nu)**2.55 # K
    sens_hera =Sensitivity.sens_Parsons2012(k, t_days=180.0, B=6.0, Tsys=Tsys,
                                N=N_survey, u_max=umax, Omega=wl**2/Aeff)
    error=(PS/np.sqrt(N_m) + sens_hera)

  return error
    

print('error=',calc_Error(5512.993, 0.1, 6.349, 'foo', 'CII'))
#print('error=',calc_Error(4.58e3, 0.039270, 6.0, 'foo', 'CII'))    
    
    
    
    
    

   
   
   

