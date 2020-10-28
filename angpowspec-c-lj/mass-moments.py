import numpy as np
import scipy.integrate as integrate
import math
import p_2014_parameters as exp

alpha = exp.alpha
phi_star = exp.phi_star
m_star = exp.m_star
m_halo_min = exp.m_halo_min
m_halo_max = exp.m_halo_max

integration_var = np.arange(m_halo_min / m_star, m_halo_max / m_star)
print(integration_var)
#mass_moment_1 = m_star * integrate.tra




