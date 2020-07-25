import numpy as np
import matplotlib.pyplot as plt


def calc_Error( PS,k, z):
    V_s=3.3*10**7*(2./16.0)*(80.0/20.0)*((1.+z)/8.0)**0.5 #cMpc/h
  
    aperture=12.0  #12m CONCERTO - 6m Stage 2
    transmission=0.3
    A_survey=2.0 #2 deg^2 CONCERTO - 10 deg^2 Stage 2
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
    error=(PS + (PN_CII*(k**3)/(2.*3.14*3.14)))/np.sqrt(N_m)

    return error

calc_Error(1E6, 1., 7.139)
    
    
    
    
    

   
   
   

