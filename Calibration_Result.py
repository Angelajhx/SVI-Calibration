Maturities_=['8/11/2021','8/13/2021','8/20/2021','8/27/2021','9/24/2021','10/29/2021','12/31/2021','3/25/2022','6/24/2022']
TimetoMaturities=[0.002968037,0.008447489,0.02739726,0.046575342,0.123287671,0.219178082,
                        0.391780822,0.621917808,0.871232877]
IV_SVI=[]
G_density=[]
for i in range(9):
    par=Optimization[i]
    t=TimetoMaturities[i]
    maturity=Maturities[i]
    lmax,lmin = 1,-0.8
    lin = np.linspace(lmin,lmax,100)
    opt=IV[(IV['Maturity']== maturity)].sort_values('Moneyness').dropna()
    moneyness=opt['Moneyness']
    model= gSVI_raw(par,lin)
    model_raw=np.array(gSVI_raw(par,moneyness))/t
    g_density=np.array(get_dens(par,lin))
    model=np.array(model)/t
    logm= moneyness.values
    tv= opt['MidConsensus'].values
    tv_bid= opt['MinTV'].values
    tv_ask= opt['MaxTV'].values
    IV_SVI.append(np.sqrt(model_raw))
    G_density.append(g_density)
    fig,ax= plt.subplots(nrows=1,ncols=3,figsize=(8,4))
    plt.figure(0)
    plt.plot(np.exp(logm), np.sqrt(tv/t), '+', markersize=12)
    plt.plot(np.exp(logm), np.sqrt(tv_bid/t), linewidth=0.5,color='black')
    plt.plot(np.exp(logm), np.sqrt(tv_ask/t), linewidth=0.5)
    plt.plot(np.exp(lin),np.sqrt(model),linewidth=2)
    plt.title("Implied Volatility Curve")
    plt.xlabel("Moneyness", fontsize=12)
    plt.figure(1)
    plt.plot(np.exp(lin), g_density,linewidth=1)
    plt.plot(np.exp(lin),np.zeros(100),linewidth=1,color="black")
    plt.title("Density")
    plt.xlabel("Moneyness", fontsize=12)
    plt.figure(2)
    tv_spread1=np.sqrt(model_raw)-np.sqrt(tv_bid/t)
    tv_spread2=np.sqrt(tv_ask/t)-np.sqrt(model_raw)
    plt.plot(np.exp(logm),tv_spread1, linewidth=0.4)
    plt.plot(np.exp(logm), tv_spread2, linewidth=0.4)
    plt.plot(np.exp(lin),np.zeros(100),linewidth=1,color="black")
    plt.show()
    print(tv_spread1.min(),tv_spread2.min(),g_density.min(),tv_spread1)
print("IV_SVI:",IV_SVI)
print("Density",G_density)
