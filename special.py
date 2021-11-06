import numpy as np
from numba import jit

p = np.array([676.5203681218851
    ,-1259.1392167224028
    ,771.32342877765313
    ,-176.61502916214059
    ,12.507343278686905
    ,-0.13857109526572012
    ,9.9843695780195716e-6
    ,1.5056327351493116e-7
    ])

@jit(nopython=True)
def gamma(z):
    z = complex(z)
    if z.real < 0.5:
        y = np.pi / (np.sin(np.pi * z) * gamma(1 - z))  # Reflection formula
    else:
        z -= 1
        x = 0.99999999999980993
        for (i, pval) in enumerate(p):
            x += pval / (z + i + 1)
        t = z + len(p) - 0.5
        y = np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x
    return y

@jit(nopython=True)
def beta(a, b):
    return gamma(a) * gamma(b) / gamma(a + b)