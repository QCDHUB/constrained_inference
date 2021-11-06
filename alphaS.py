import sys,os,time
import numpy as np
from numba import jit

#--mpmath
from mpmath import fp

from params import mc2,mb2,mZ2,alphaSMZ,order


@jit(nopython=True)
def get_Nf(Q2):
    Nf=3
    if Q2>=(mc2): Nf+=1
    if Q2>=(mb2): Nf+=1
    return Nf

#--setup the beta function
beta=np.zeros((7,3))
for Nf in range(3,7): 
    beta[Nf,0]=11-2/3*Nf 
    beta[Nf,1]=102-38/3*Nf 
    beta[Nf,2]=2857/2-5033/18*Nf+325/54*Nf**2 

@jit(nopython=True)
def beta_func(a,Nf,order):
    betaf = -beta[Nf,0]
    if order>=1: betaf+=-a*beta[Nf,1]
    if order>=2: betaf+=-a*beta[Nf,2]
    return betaf*a**2


@jit(nopython=True)
def evolve_a(Q20,a,Q2,Nf,order):
    # Runge-Kutta 
    LR = np.log(Q2/Q20)/20
    for k in range(20):
        XK0 = LR * beta_func(a,Nf,order)
        XK1 = LR * beta_func(a + 0.5 * XK0,Nf,order)
        XK2 = LR * beta_func(a + 0.5 * XK1,Nf,order)
        XK3 = LR * beta_func(a + XK2,Nf,order)
        a+= (XK0 + 2.* XK1 + 2.* XK2 + XK3) * 0.166666666666666
    return a


order=0
aZ=alphaSMZ/(4*np.pi)
ab=evolve_a(mZ2,aZ,mb2,5,order)
ac=evolve_a(mb2,ab,mc2,4,order)
a0=evolve_a(mc2,ac,  1,3,order)


@jit(nopython=True)
def get_a(Q2):
    if mb2<=Q2:
        return evolve_a(mb2,ab,Q2,5,order)
    elif mc2<=Q2 and Q2<mb2: 
        return evolve_a(mc2,ac,Q2,4,order)
    elif Q2<mc2:
        return evolve_a(1,a0,Q2,3,order)

@jit(nopython=True)
def get_alphaS(Q2):
    return get_a(Q2)*4*np.pi


if __name__=='__main__':
    
    print(get_alphaS(10.0))

