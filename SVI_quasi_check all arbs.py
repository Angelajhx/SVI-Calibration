import numpy as np
import scipy as sp
import scipy.optimize as optimize
from scipy.optimize import minimize
import time
import pandas as pd
import matplotlib.pyplot as plt
import time
startTime = time.time()

# two step first optimize a,d,c
#algo is pretty easy and fast to treat linear gradiabe from a,d,c presentation of SVI
#after optimize a,d,c. optimize m,sigma

def svi_2steps(iv,x,init_msigma,weight,bid,ask,par_,ext=10,maxiter=50,exit=1e-12,verbose=True):
    opt_rmse=1

    #redefine first local optimization function, y is function of log moneyness
    def svi_quasi(y,a,d,c):
        return a+d*y+c*np.sqrt(np.square(y)+1)
    
    #local mean difference for first step calibration
    #iv is still total variance
    def svi_quasi_rmse(iv,y,a,d,c):
        return np.sqrt(np.mean(np.square(svi_quasi(y,a,d,c)-iv)))
    
    def svi_raw(par, k):
        a, b, rho, m, tau = par
        y = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + tau ** 2))
        if y>=0:
            return y
        else:
            return 0
    
    def sviCurveDerivate(par, k):
        a, b, rho, m, tau = par
        return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + tau ** 2))
    
    def sviCurve2ndDerivate(par, k):
        a, b, rho, m, tau = par
        return b * tau * tau / np.power((k - m) ** 2 + tau ** 2, 1.5)

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
    
    # calculated a,d,c
    #additional boundaries have to be considered: fab(d)<=c&fab(d)<=4*sigma-c\
    #boundary is wrong
    #use 45 degree np.sqrt(2)/2*(y+z),np.sqrt(2)/2*(-y+z)
    #np.sqrt(2)/2*(d-c),np.sqrt(2)/2*(d+c)
    def SVI_adc(iv,x,_m,_sigma):
        y = (x-_m)/_sigma
        s = max(_sigma,1e-6)
        bnd = ((0.00001,-4*s,0),(max(iv.max(),1e-4),4*s,4*s))
        z = np.sqrt(np.square(y)+1)
        A = np.column_stack([np.ones(len(iv)),y,z])
        a,d,c = optimize.lsq_linear(A,iv,bnd,tol=1e-12,verbose=False).x
        return a,d,c
    
    def envelopeCondition(pAdjusted, bid, ask, ext):
        return np.exp(np.clip([0 if (pi>=bi and pi<=ai) else ext*2*np.fabs(pi-(ai+bi)/2)/(ai-bi) 
                               for pi, bi, ai in zip(pAdjusted, bid, ask)],a_min=None,a_max=10))
    

    def butterflyArbitrage(par,moneyness):
        a, b, rho, m, tau = par
        g = []
        for k in moneyness:
            gi = gfun(par,k)
            g.append(gi)  
        return np.exp([0 if gi>=0 else 10 for gi in g])
    
    def calendarspreadArb(par,moneyness,par_):
        a, b, rho, m, tau = par
        
        calendar=[]
        for k in moneyness:
            svi_T=svi_raw(par, k)
            svi_t=svi_raw(par_, k)
            cali=svi_T-svi_t
            calendar.append(cali) 
            return np.exp([0 if cali>=0 else 10 for cali in calendar])
    
    def opt_msigma(msigma,iv,weight,bid,ask,x,par_):
        _m,_sigma = msigma
        _y = (x-_m)/_sigma 
        _a,_d,_c = SVI_adc(iv,x,_m,_sigma)
        par=[_a,_c/_sigma,_d/_c,_m,_sigma]
        vol_SVI = svi_quasi(_y,_a,_d,_c)
        consButterfly = butterflyArbitrage(par,x)
        consEnvelope = envelopeCondition(vol_SVI, bid, ask,10)
        consCalendar = calendarspreadArb(par,x,par_)
        OF = np.array([(vol/volMar - 1)*bi*ei*ci*w for vol,volMar,bi,ei,ci,w 
                       in zip(vol_SVI, iv, consButterfly, consEnvelope,consCalendar,weight)])
        return np.sum(OF**2)
    

    for i in range(1,maxiter+1):
        #a_star,d_star,c_star = SVI_adc(iv,x,init_msigma) 
        args=(iv,weight,bid,ask,x,par_)
        m_star,sigma_star = optimize.minimize(opt_msigma,init_msigma,args=args,method='Nelder-Mead',
                                              bounds=((2*min(x.min(),0), 2*max(x.max(),0)),(1e-2,1)),tol=1e-12).x
        a_star,d_star,c_star = SVI_adc(iv,x,m_star,sigma_star)
        opt_rmse1 = svi_quasi_rmse(iv,(x-m_star)/sigma_star,a_star,d_star,c_star)
        if verbose:
            print(f"round {i}: RMSE={opt_rmse1} para={[a_star,d_star,c_star,m_star,sigma_star]}")
        if i>1 and opt_rmse-opt_rmse1<exit and np.fabs(d_star)<=c_star and np.fabs(d_star)<=4*sigma_star-c_star and c_star/sigma_star*(1+np.fabs(d_star/c_star))<=4 :
            break
        opt_rmse = opt_rmse1
        #init_msigma = [m_star+np.random.random(1)*0.1,sigma_star+np.random.random(1)/5*min(sigma_star,1-sigma_star)]
        init_msigma = [m_star,sigma_star]
    result = np.array([a_star,d_star,c_star,m_star,sigma_star,opt_rmse1])
    if verbose:
        print(f"\nfinished. params = {result[:5].round(10)}")
    return result


def gfunction(par,k):
    a,b,rho,m,tau = par
    discr = np.sqrt((k-m)*(k-m) + tau*tau)
    w = a + b *(rho*(k-m)+ discr)
    dw = b*rho + b *(k-m)/discr
    d2w = b*tau**2/(discr*discr*discr)
    return 1 - k*dw/w + dw*dw/4*(-1/w+k*k/(w*w)-4) +d2w/2   

def quasi2raw(a,d,c,m,sigma):
    return a,c/sigma,d/c,m,sigma

def svi_raw(x,a,b,rho,m,sigma):
    centered = x-m
    return a+b*(rho*centered+np.sqrt(np.square(centered)+np.square(sigma)))

def svi_quas_cal(x,a,d,c,m,sigma):
    y = (x-m)/sigma
    return a+d*y+c*np.sqrt(np.square(y)+1)

class svi_quasi_model:
    def __init__(self,a,d,c,m,sigma):
        self.a = a
        self.d = d
        self.c = c
        self.m = m
        self.sigma = sigma
    def __call__(self,x):
        return svi_quasi(x,self.a,self.d,self.c,self.m,self.sigma)

def plot_tv(logm,tv,model,extend=0.5,n=100):
    scale = (max(logm)-min(logm))*extend
    lmax,lmin = min(logm)-scale,max(logm)+scale
    lin = np.linspace(lmin,lmax,n)
    plt.figure(figsize=(8, 4))
    plt.plot(logm, tv, '+', markersize=12)
    plt.plot(lin,model(lin),linewidth=1)
    plt.title("Total Variance Curve")
    plt.xlabel("Log-Moneyness", fontsize=12)
    plt.legend()
    
def plot_iv(logm,tv,t,model,extend=0.1,n=100):
    scale = (max(logm)-min(logm))*extend
    lmax,lmin = min(logm)-scale,max(logm)+scale
    lin = np.linspace(lmin,lmax,n)
    plt.figure(figsize=(8, 4))
    plt.plot(np.exp(logm), np.sqrt(tv/t), '+', markersize=12)
    plt.plot(np.exp(lin),np.sqrt(model(lin)/t),linewidth=1)
    plt.title("Implied Volatility Curve")
    plt.xlabel("Moneyness", fontsize=12)
    plt.legend()

#check arbitrage
def raw_svi(par, k):
    w = par[0] + par[1] * (par[2] * (k - par[3]) + ((k - par[3]) ** 2 + par[4] ** 2) ** 0.5)
    return w
def diff_svi(par, k):
    a, b, rho, m, sigma = par
    return b*(rho+(k-m)/(np.sqrt((k-m)**2+sigma**2)))

def diff2_svi(par, k):
    a, b, rho, m, sigma = par
    disc = (k-m)**2 + sigma**2
    return (b*sigma**2)/((disc)**(3/2))

#g(x)to make sure probability density always positive to avoid butterfly arb
def d2(par, k):
    v = np.sqrt(raw_svi(par, k))
    return -k/v - 0.5*v

#get probablity density from g(x)
def density(par, k):
    g = gfun(par, k)
    w = raw_svi(par, k)
    dtwo = d2(par, k)
    dens = (g / np.sqrt(2 * np.pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens


#plot certain maturity
IV = pd.read_csv("Total Variance1.csv",delimiter=',')
Maturities=['8/11/2021','8/13/2021','8/20/2021','8/27/2021','9/24/2021','10/29/2021','12/31/2021','3/25/2022','6/24/2022']
TimetoMaturities=[0.002968037,0.008447489,0.02739726,0.046575342,0.123287671,0.219178082,
                        0.391780822,0.621917808,0.871232877]
#initiation of m, sigma
msigma=[[0.003,0.01],[0.02,0.06],[-0.03,0.25],[-0.06,0.3],[-0.29,0.5],[-0.25,0.4],[-0.3,0.7],[-0.3,0.7],[-0.25,0.55]]
Optimization=[]
OptimizationPara=([[0,0,0,0,0]])
lmax,lmin = 0.7,-1
lin = np.linspace(lmin,lmax,100)
print("Optimization begins...")
for i in range(9):
    t=TimetoMaturities[i]
    maturity=Maturities[i] 
    opt=IV[(IV['Maturity']== maturity)].sort_values('Moneyness').dropna()
    w_max=opt['MaxTV'].max()
    a,d,c,m,sigma,rmse = svi_2steps(opt['MidConsensus'],opt['Moneyness'],msigma[i],opt['Weight'],opt['MinTV'],opt['MaxTV'],OptimizationPara[i])

    OptimizationParai=[a,c/(sigma),d/c,m,sigma]
    OptimizationPara.append(OptimizationParai)
    Optimization.append((OptimizationParai,rmse))              
    model_svi = svi_quas_cal(lin,a,d,c,m,sigma)
    g_values= gfunction(OptimizationParai,lin)
    logm=opt['Moneyness'].values
    tv=opt['MidConsensus'].values
    fig,ax= plt.subplots(nrows=1,ncols=2,figsize=(8,4))
    plt.figure(0)
    plt.plot(np.exp(logm), np.sqrt(tv/t), '+', markersize=12)
    plt.plot(np.exp(lin),np.sqrt(model_svi/t),linewidth=1)
    plt.title("Implied Volatility Curve")
    plt.xlabel("Moneyness", fontsize=12)
    plt.figure(1)
    plt.plot(np.exp(lin),g_values,linewidth=1)
    plt.plot(np.exp(lin),np.zeros(100),linewidth=1,color="black")
    plt.title("Density")
    plt.xlabel("Moneyness", fontsize=12)
    plt.show()

print(Optimization)
#init_=[0.0001,0.1,-0.4,-0.1,0.2]
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
