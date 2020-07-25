import numpy as np
import matplotlib.pyplot as plt

# Equation 19 of D19
# P_CII^N = V_pix  * sigma_pix^2 / t_pix

######################### t_pix #############################

def t_pix():
    t_survey * N_pix * omega_beam / A

# dependencies of t_pix

t_survey = 1500    # hours
A = 2 # degree squared
N_pix = 1500 # mentioned below eq. 23 in D19
D = 12 # metres
"""
lambda_obs =  # should be in metres. D in theta_beam() is in metres.
"""
def omega_beam():
    return 2 * 3.14 * (theta_beam / 2.355)**2

def theta_beam():
    return 1.22 * lambda_obs / D

###################### sigma_pix #############################
def sigma_pix():
    return NEI_diff**2 / N_pix

# dependencies of sigma_pix()
NEI = 155 * N_pix**0.5  # Fix units
NEI_diff = NEI * 10**(-19) / omega_beam