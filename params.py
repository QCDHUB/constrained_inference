import sys,os,time
import numpy as np

#--mpmath
from mpmath import fp


#--set_constants 

CA=3
CF=4/3
TR=1/2
TF=1/2
euler=fp.euler 

#--set_masses

me   = 0.000511
mm   = 0.105658
mt   = 1.77684
mu   = 0.055
md   = 0.055
ms   = 0.2
mc   = 1.28
mb   = 4.18

mZ   = 91.1876
mW   = 80.398
M    = 0.93891897
Mpi  = 0.13803
Mk   = 0.493677
Mdelta = 1.232

me2   = me**2 
mm2   = mm**2 
mt2   = mt**2
mu2   = mu**2  
md2   = md**2  
ms2   = ms**2  
mc2   = mc**2  
mb2   = mb**2  
mZ2   = mZ**2  
mW2   = mW**2  
M2    = M**2  
Mpi2  = Mpi**2  
Mdelta2=Mdelta**2

#--set_couplings       

c2w   = mW2/mZ2
s2w   = 1.0-c2w
s2wMZ = 0.23116
c2wMZ = 1.0 - s2wMZ
alfa  = 1/137.036
alphaSMZ = 0.118
GF = 1.1663787e-5   # 1/GeV^2

#--pQCD setup
order=1
