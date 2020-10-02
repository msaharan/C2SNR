import numpy as np
import scipy.integrate as integrate

def integrand(para1, para2, para3):
    var = para2 + para3
    return para1 + var

def function(para1):
    return integrate.quad(integrand, 0, 10, 0, 10)

para1  = 1

print(function(para1))
