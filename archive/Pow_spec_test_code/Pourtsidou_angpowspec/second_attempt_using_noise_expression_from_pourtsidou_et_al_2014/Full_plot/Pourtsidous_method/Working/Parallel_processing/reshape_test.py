import numpy as np

junk1, junk2, ang = np.loadtxt('./J_text_files/gaussian_plus_poisson_integration_over_redshift_angpowspec_with_j_j_upp_limit_1_lmax_700_eta_0.txt', unpack = True)

ang.reshape(len(junk1), 3)
print(ang)
