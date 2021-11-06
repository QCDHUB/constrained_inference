import numpy as np
from numba import jit 

#--countour setup
npts=8
c=1.9
phi=3.0/4.0*np.pi
  
#--gen z and w values along coutour
x,w=np.polynomial.legendre.leggauss(npts)
znodes=[0,0.1,0.3,0.6,1.0,1.6,2.4,3.5,5,7,10,14,19,25,32,40,50,63]
  
Z,W,JAC=[],[],[]
for i in range(len(znodes)-1):
    a,b=znodes[i],znodes[i+1]
    Z.extend(0.5*(b-a)*x+0.5*(a+b))
    W.extend(w)
    JAC.extend([0.5*(b-a) for j in range(x.size)])

#--globalize
W  =np.array(W)
Z  =np.array(Z)
JAC=np.array(JAC)

#--gen mellin contour
N=c+Z*np.exp(complex(0,phi)) 
phase= np.exp(complex(0,phi))
      
@jit(nopython=True)
def invert(x,F):
    return np.sum(np.imag(phase * x**(-N) * F)/np.pi * W * JAC)