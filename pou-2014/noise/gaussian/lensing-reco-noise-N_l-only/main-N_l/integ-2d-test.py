import numpy as np
import sympy
l1, l2 = sympy.symbols('l1 l2')
print(sympy.integrate(l1  + l2, (l1, 0, 1000), (l2, 0, 1000)))

