import numpy as np
from numba import jit
import special

lam2=0.4
Q02=1

par=[1,0.1,0.1] #A
par=np.append(par,[-0.5,0.1,0.1]) # a
par=np.append(par,[3,0.1,0.1]) #b
par=np.append(par,[0,0.1,0.1]) #c
#par=np.append(par,[0,0.1,0.1]) #d
#par=np.append(par,[0,0.1,0.1]) #e


@jit(nopython=True)
def get_s(Q2):
    return np.log(np.log(Q2/lam2)/np.log(Q02/lam2))

@jit(nopython=True)
def get_parQ2(par,Q2):
    s=get_s(Q2)
    return par[0]+par[1]*s+par[2]*s**2

@jit(nopython=True)
def get_pdf(x,Q2,par):
    A=get_parQ2(par[:3],Q2)
    a=get_parQ2(par[3:6],Q2)
    b=get_parQ2(par[6:9],Q2)
    c=get_parQ2(par[9:12],Q2)
    #d=get_parQ2(par[12:15],Q2)
    #e=get_parQ2(par[15:18],Q2)
    return A*x**a*(1-x)**b#*(1+c*x+d*x**2+e*x**3)


@jit(nopython=True)
def get_pdf_N(N,Q2,par):
    A=get_parQ2(par[:3],Q2)
    a=get_parQ2(par[3:6],Q2)
    b=get_parQ2(par[6:9],Q2)
    c=get_parQ2(par[9:12],Q2)
    d=get_parQ2(par[12:15],Q2)
    e=get_parQ2(par[15:18],Q2)
    pdf=special.beta(N+a,b+1)+c*special.beta(N+a+1,b+1)+d*special.beta(N+a+2,b+1)+e*special.beta(N+a+3,b+1)
    return A*pdf

@jit(nopython=True)
def get_ds_dQ2(par,Q2):
    p1,p2,p3=par
    return p1/(Q2*np.log(Q2/lam2)) + 2*p2*np.log(np.log(Q2/lam2)/np.log(Q02/lam2))/(Q2*np.log(Q2/lam2))

@jit(nopython=True)
def get_dpdf_dQ2(x,Q2,par):

    A=get_parQ2(par[:3],Q2)
    a=get_parQ2(par[3:6],Q2)
    b=get_parQ2(par[6:9],Q2)
    c=get_parQ2(par[9:12],Q2)
    d=0#get_parQ2(par[12:15],Q2)
    e=0#get_parQ2(par[15:18],Q2)

    dpdfdA= x**a*(1 - x)**b*(c*x + d*x**2 + e*x**3 + 1)
    dpdfda= A     *x**a*(1 - x)**b*(c*x + d*x**2 + e*x**3 + 1)*np.log(x)
    dpdfdb= A     *x**a*(1 - x)**b*(c*x + d*x**2 + e*x**3 + 1)*np.log(1 - x)
    dpdfdc= A*x   *x**a*(1 - x)**b
    #dpdfdd= A*x**2*x**a*(1 - x)**b
    #dpdfde= A*x**3*x**a*(1 - x)**b
    
    dAdQ2=get_ds_dQ2(par[:3],Q2)
    dadQ2=get_ds_dQ2(par[3:6],Q2)
    dbdQ2=get_ds_dQ2(par[6:9],Q2)
    dcdQ2=get_ds_dQ2(par[9:12],Q2)
    #dddQ2=get_ds_dQ2(par[12:15],Q2)
    #dedQ2=get_ds_dQ2(par[15:18],Q2)
    
    return   dpdfdA*dAdQ2 + dpdfda*dadQ2 + dpdfdb*dbdQ2 + dpdfdc*dcdQ2# + dpdfdd*dddQ2 + dpdfde*dedQ2
    
    
    


