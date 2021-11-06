import sys,os,time
import numpy as np
from numba import jit
from params import CF
from mpmath import fp

import alphaS
import mellin

get_psi= lambda i,_N: fp.psi(i,complex(_N.real,_N.imag))
get_S1 = lambda _N: fp.euler + get_psi(0,_N+1)
get_S2 = lambda _N: zeta2 - get_psi(1,_N+1)

N=mellin.N


Nf=3
S1 = np.array([get_S1(n) for n in N])
P=CF*(3+2/N/(N+1)-4*S1)
R=P/alphaS.beta[Nf,0]
Q02=1

@jit(nopython=True)
def evolve(pdf0,Q2):
    a = alphaS.get_a(Q2)
    a0= alphaS.get_a(Q02)
    U=np.power(a/a0,-R)
    return U*pdf0    
