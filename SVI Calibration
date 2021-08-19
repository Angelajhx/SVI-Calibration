import numpy as np
import scipy as sp
import scipy.optimize as optimize
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time
import pandas as pd
import matplotlib.pyplot as plt


#SVI raw model
def svi_raw(par, k):
    a, b, rho, m, tau = par
    y = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + tau ** 2))
    if y>=0:
        return y
    else:
        return 0
#SVI first derivative to k
def sviCurveDerivate(par, k):
    a, b, rho, m, tau = par
    return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + tau ** 2))

def sviCurve2ndDerivate(par, k):
    a, b, rho, m, tau = par
    return b * tau * tau / np.power((k - m) ** 2 + tau ** 2, 1.5)

#Check if butterfly arb, create penalty exp(10) 
def butterflyArbitrage(par,moneyness):
    a, b, rho, m, tau = par
    g = []
    for k in moneyness:
        gi = gfun(par,k)
        g.append(gi)  
    return np.exp([0 if gi>=0 else 10 for gi in g])
# gfun to calculate whether there is butterfly arb
def gfun(par, k):
    a, b, rho, m, tau = par
    w = svi_raw(par, k)
    wd = sviCurveDerivate(par, k) 
    wdd = sviCurve2ndDerivate(par, k)
    if w!=0:
        g = (1 - k * wd / (2 * w)) ** 2 - (wd * wd / 4) * (1 / w + 0.25) + wdd / 2
    else:
        g= -1
    return g

def get_dens(par, moneyness):
    a, b, rho, m, tau = par
    dens = []
    for k in moneyness:
        densi=density(par, k)
        dens.append(densi)  
    return dens

def d2(par, k):
    v = np.sqrt(svi_raw(par, k))
    return -k/v - 0.5*v
    
#Calculate risk neutral density
def density(par, k):
    g = gfun(par, k)
    w = svi_raw(par, k)
    dtwo = d2(par, k)
    dens = (g / np.sqrt(2 * np.pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens

def gSVI_raw(par, moneyness):
    a, b, rho, m, tau = par
    SVI = []
    for k in moneyness:
        svi = svi_raw(par, k)
        SVI.append(svi)  
    return SVI

#within bid ask penalty, ext is the penalty para
def envelopeCondition(pAdjusted, bid, ask, ext):
    return np.exp(np.clip([0 if (pi>=bi and pi<=ai) else ext*2*np.fabs(pi-(ai+bi)/2)/(ai-bi) 
                           for pi, bi, ai in zip(pAdjusted, bid, ask)],a_min=None,a_max=10))

#weight here is to assign higher weight to around ATM and not focus much on wing calibration

def globalOptimization(X, volMarket,weight,bid,ask,moneyness):
    vol_SVI = gSVI_raw(X, moneyness) 
    consButterfly = butterflyArbitrage(X, moneyness)
    consEnvelope = envelopeCondition(vol_SVI, bid, ask,10)
    OF = np.array([(vol/volMar - 1)*bi*ei*w for vol,volMar,bi,ei,w 
                   in zip(vol_SVI, volMarket, consButterfly, consEnvelope,weight)])
    return np.sum(OF**2)
    
## follow-up function
def callbackFunction(X, convergence):
    xProgress.append(X) # global variable containing progress of the optimization process

    
#bring in object to optimize
#a, b, rho, m, tau
#max(a)->max(Wsvi)
# b positive<1
IV = pd.read_csv("Total Variance1.csv",delimiter=',')
Maturities=['8/11/2021','8/13/2021','8/20/2021','8/27/2021','9/24/2021','10/29/2021','12/31/2021','3/25/2022','6/24/2022']
Optimization=[]
print("Optimization begins...")
xProgress=[]
for maturity in Maturities:
    opt=IV[(IV['Maturity']== maturity)].sort_values('Moneyness').dropna()
    w_max=opt['MaxTV'].max()
    args = (opt['MidConsensus'],
            opt['Weight'],
            opt['MinTV'],
            opt['MaxTV'],
            opt['Moneyness'])
    boundsParameters = [(0.000001,w_max), (0.001,1), (-1,1), (-0.5,0.5), (0.01,1)]
    OptimizationPara = differential_evolution(globalOptimization,
                                              args = args,
                                              callback=callbackFunction,
                                              bounds = boundsParameters,
                                              polish=True,
                                              maxiter=1000,
                                              tol = 1E-7).x
    Optimization.append(OptimizationPara)
    
print(Optimization)
