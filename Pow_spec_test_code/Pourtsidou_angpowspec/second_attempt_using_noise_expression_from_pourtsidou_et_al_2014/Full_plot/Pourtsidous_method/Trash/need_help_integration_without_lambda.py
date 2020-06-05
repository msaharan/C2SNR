
# The term 'paper' refers to Pourtsidou et al. 2014

######################################################
# Problem
#####################################################
# I don't know how to implement the integration without using 'lambda' for the following functions.
# It's because of the way I have defined 'small_L' in the code.
#---------------------------------------------------


# Integration
# Read this first.
def N4(ell,redshift):
    #####################################################
    # About l1, l2
    ######################################################
    # Look at the double-ell integrations in equation 14 of the paper. 
    # In the equation, the integration is written as "Integration d^2 ell Integrand(ell)".
    # In the code, I have written it as "Integration d ell_1 Integration d ell_2 Integrand(ell_1, ell_2) ", where both ell_1 and ell_2 run from 0 to l_max = 19900.
    # Take a look at the return statement and move on to the integrand function
    #----------------------------------------------------
    return integrate.dblquad(lambda l1,l2: N4_integrand(ell,l1,l2), 0, l_max, lambda l2: 0, lambda l2: l_max)[0]

# Integrand
def N4_integrand(ell,l1,l2, redshift): 
    ###################################################
    # About 'small_L'
    ###################################################
    # In the N4 term of equation 14 of the paper you will see the following term: C^{tot}_{|ell - L|}. 
    # I have denoted this 'ell' as 'small_L' in the code.
    # The way I have defined the small_l below might be wrong but let's not focus on that for the moment.
    # Take a look at the definition of small_L and the return statememt and move on.
    #----------------------------------------------------
    small_l = int(np.sqrt(l1**2 + l2**2))
    return 2*(angpowspec_without_j[ell] + (C_l)) * (angpowspec_without_j[abs(ell-small_l)] + (C_l)) / two_pi_squared


######################################################
# Questions
######################################################
# The lambda function defines the integration variables l1 and l2.
# That's why I can define 'small_L' the way I have defined it in the integrand function.
# Question 1: 
#   How can we define small_L without using lambda?
# 
# Assuming that we get the answer to Question 1.
# Question 2: 
#   How do we write the double integration without using lambda?
#
#
# I googled to find a way to write the integration function without using lambda but couldn't 
# get much help. It would be great if you can tell me how to do so by writing this integration 
# function without using lambda. 
#---------------------------------------------------

