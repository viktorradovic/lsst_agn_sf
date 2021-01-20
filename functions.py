

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from scipy.stats import binned_statistic
from ipywidgets import widgets
from IPython.display import display
import time

# LSST libraries, MAF metrics


def LC_conti(long, deltatc=1, oscillations=True, A=0.2, P=500):
    """ 
    Parameters:
    -----------
    long: int
        Duration of the survey in days.
    deltatc: float or int, default=1
        Cadence (time interval between two observations or samplings of the light curve in days).
    oscillations: bool, default=True
        If True, light curve simulation will take an oscillatory signal into account.
    P: float or int, default=500
        Period of the oscillatory signal in days.
    A: float or int, default=0.2
        Amplitude of the oscillatory signal in magnitudes.
     
    Returns:
    --------
    yy: np.array
        Obesrved light curve points in magnitudes.
    tt: np.array
        Survey days when the light curve points were sampled.
    """
    
    # Generating survey days 
    tt = np.arange(0, long, int(deltatc))
    times = tt.shape[0]
    
    # Constants
    const1 = 0.455*1.25*1e38
    const2 = np.sqrt(1e09)
    SFCONST2 = 0.18
    meanmag = 23
    
    # Generating log L_bol
    loglumbol = np.random.uniform(42.2,47,1)
    lumbol = np.power(10,loglumbol)

    # Calculate M_{SMBH}
    msmbh=np.power((lumbol*const2/const1),2/3.)
    
    # Calculate damping time scale
    tau = 80.4*np.power(lumbol/1e45,-0.42)*np.power(msmbh/1e08,1.03)
    
    # Calculate  log sigma^2 - an amplitude of correlation decay
    logsig2=-3.83-0.09*np.log10(lumbol/1e45)-0.25*np.log10(msmbh/1e08)
    sig2=np.power(10,logsig2)
    
    # Calculating light curve points
    ss = np.zeros(times)
    ss[0] = np.random.normal(meanmag, np.sqrt(sig2), 1)
    ratio = -deltatc/tau
    ssigma1=np.sqrt(sig2)*np.sqrt((1-np.exp(2*ratio)))
    
    for i in range(1, times):
        ss[i] = np.random.normal(ss[i-1]*np.exp(ratio) + meanmag*(1-np.exp(ratio)),
                                     np.sqrt(np.abs(0.5*SFCONST2*SFCONST2*(1-np.exp(-2*ratio)))),1)
        
    # Calculating error (Ivezic et al. 2019) --> https://iopscience.iop.org/article/10.3847/1538-4357/ab042c/pdf
    gamma=0.039
    m5=24.7
    x=np.zeros(len(ss))
    x=np.power(10, 0.4*(ss-m5))

    greska2=(0.005**2)+(0.04-gamma)*x+gamma*x*x
    
    # Final light curve with oscillations
    if oscillations == True:
        sinus=A*np.sin(2*np.pi*tt/P)
        yy = np.zeros(times)
        for i in range(times):
            yy[i]=ss[i]+np.sqrt(greska2[i]) + sinus[i]
    
        return yy, tt
    
    # Final light curve without oscillations
    if oscillations == False:
        yy = np.zeros(times)
        for i in range(times):
            yy[i]=ss[i]+np.sqrt(greska2[i])
    
        return yy, tt

def LC_conti_old(long, sigma, tau, deltatc=1, oscillations=True, A=0.2, P=500, noise=0.05):
    """ 
    Parameters:
    -----------
    long: int
        Duration of the survey in days.
    sigma: float
        Standard deviation for Dumped Random Walk (DRW).
    tau: float or int
        Time lag.
    deltatc: float or int, default=1
        Cadence (time interval between two observations or samplings of the light curve in days).
    oscillations: bool, default=True
        If True, light curve simulation will take the oscilatory signal into account.
    P: float or int, default=500
        Period of the oscilatory signal in days.
    A: float or int, default=0.2
        Amplitude of the oscilatory signal in magnitudes.
    noise: float, default=0.05
        Add noise to the simulated light curve.
     
    Returns:
    --------
    yy: np.array
        Obesrved light curve points (Eq. 17 in Kovacevic+2020).
    tt: np.array
        Days during the survey on which we had an observation.
    """
    # Generating survey days
    tt = np.arange(0, long, int(deltatc))
    times = tt.shape[0]
    
    # Calculating light curve points (Kovacevic+2020)
    ss = np.zeros(times)
    ss[0] = np.random.normal(0, sigma, 1)
    ssigma1=sigma*np.sqrt((1-np.exp(-2*deltatc/tau)))
    
    # Light curve with oscillations
    if oscillations == True:
        sinus=np.zeros(times)
        for i in range(1, times):
            sinus[i]=A*np.sin(2*np.pi*tt[i]/P)
            ss[i] = ss[i-1]*np.exp(-deltatc/tau) + np.random.normal(0, ssigma1,1)
            
        yy = np.zeros(times)
        for i in range(times):
            yy[i]=ss[i]+np.random.normal(0,(noise*np.abs(ss[i])),1)+sinus[i]
        
        return yy, tt
    
    # Light curve without oscillations
    elif oscillations == False:
        for i in range(1, times): 
            ss[i] = ss[i-1]*np.exp(-deltatc/tau) + np.random.normal(0, ssigma1,1)
        yy=np.zeros(times)
        for i in range(times):
            yy[i]=ss[i]+np.random.normal(0,(noise*np.abs(ss[i])),1)
        
        return yy, tt


def LC_opsim(mjd,t,y):
    """ 
    Parameters:
    -----------
    mjd: np.array
        Modified Julian Date obtained from OpSim. It is the time of each sampling for the light curve.
    t: np.array
        Days during the survey on which we had an observation for continuous simulated light curve (cadence = 1 day).
    y: np.array
        Obesrved light curve points (Eq. 17 in Kovacevic+2020) for continuous simulated light curve (cadence = 1 day).
     
    Returns:
    --------
    top: np.array
        Days during the survey on which we had an observation (sampling) in a given OpSim.
    yop: np.array
        Observed light curve points taken from the continuous light curve on days we had an observation (sampling) in a 
        given OpSim.
    """
    # Racunamo duzinu surveya (ALI SE NE KORISTI NIGDE, izbrisati?)
    #long=np.int(mjd.max()-mjd.min()+1)

    top=np.ceil(mjd-mjd.min())
    
    yop=[]
    for i in range(len(top)):
        abs_vals = np.abs(t-top[i])
        
        # Find matching days and their index
        bool_arr = abs_vals < 1
        if bool_arr.sum() != 0:
            index = (np.where(bool_arr)[0])[0]
            yop.append(y[index])
            
        # Case when we don't have a match
        elif bool_arr.sum() == 0:
            yop.append(-999)
    yop=np.asarray(yop)
    
    # Drop placeholder values (-999) coresponding to cases when we don't have a match between cont. LC and OpSim LC days.
    top = top[yop!=-999]
    yop = yop[yop!=-999]
    
    return top,yop
  

def sf(rol1t,rol1):
    """
    Parameters:
    -----------
    rol1t: np.array
        Days during the survey on which we had an observation (sampling) in a given OpSim.
    rol1: np.array
        Observed light curve points taken from the continuous light curve on days we had an observation (sampling) in a 
        given OpSim.

    Returns:
    --------
    srol: np.array
        Mean of the squared flux difference between consecutive light curve points in a bin with edges 
        defined by edgesrol. Used for plotting the y-axis of the structure function visualization.
    edgesrol: np.array
        Bin edges for the range of intervals between consecutive observation times for a given OpSim. Used for plotting the
        x-axis in the structure function visualization.
    """

    #KK=len(rol1t) # broj tacaka (mozda suvisno jer se ponovo definise kasnije, a pre toga se ne koristi)
    dtr=[]
    dy1r=[]
    KK=np.asarray(rol1.shape)[0]
    for i in range(KK-1):
        dtr.append(rol1t[i+1:KK] - rol1t[i])
        dy1r.append((rol1[i+1:KK] - rol1[i])*(rol1[i+1:KK] - rol1[i]))

    dtr = np.concatenate(dtr, axis=0)
    dy1r = np.concatenate(dy1r, axis=0)
    
    srol, edgesrol, _ = binned_statistic(dtr, dy1r, statistic='mean', bins=np.logspace(0,4,100))
    
    return srol,edgesrol
    
    
def LC_SF_viz(deltatc, long, opsims, labels, A=0.2, noise=0.05,  oscilations=True, P=500):

    # Every time the LC_conti function is called, a random simulated light curve is generated. We do this only once because
    # we want to evaluate OpSim light curves on the same referent continuous light curve.
    yy, tt = LC_conti(long, deltatc, oscillations, A, P)
    
    # Now, we calculate structure function for the continuous light curve.
    
    sc, edgesc = sf(tt, yy)
    
    # We create a figure where all the light curves will be stored.
    fig1 = plt.figure(figsize=(15,10))
    
    # Continuous light curve is plotted first.
    ax11 = fig1.add_subplot(len(opsims)+1,1,1)
    ax11.plot(tt,yy, 'ko', markersize = 1, label='1 day cadence')
    plt.setp(ax11.get_xticklabels(), visible=False)
    custom_xlim = (0, long)
    custom_ylim = (yy.min()-0.3, yy.max()+0.3)
    ax11.set_ylabel('mag [ar.un.]', fontsize = 14)
    ax11.tick_params(direction='in', pad = 5)
    plt.setp(ax11, xlim=custom_xlim, ylim=custom_ylim)
    ax11.legend()
    ax11.grid(True)
    
    # We will store the results from each OpSim light curve and later use those to calculate their structure functions.
    tops = []
    yops = []
    # Now we plot all OpSim light curves that user provided.
    for i, mjd in enumerate(opsims):
        start_time = time.time()
        top, yop = LC_opsim(mjd, tt, yy)
        tops.append(top)
        yops.append(yop)
        
        ax = fig1.add_subplot(len(opsims)+1,1,i+2)
        ax.plot(top, yop, 'ko', markersize = 1, label='%s' %labels[i])
        
        if (i+1) == len(opsims):
            ax.set_xlabel('t [days]', fontsize=14)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
    
        ax.set_ylabel('magnitude', fontsize = 14)
        ax.tick_params(direction='in', pad = 5)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        ax.legend()
        ax.grid(True)    
    
    tops = np.asarray(tops)
    yops = np.asarray(yops)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.05)
    fig1.suptitle('Light curves', y=0.92, fontsize=17)
    
    plt.show()
    
    
    # We create a second figure to store structure function plots.
    fig2 = plt.figure(figsize=(15,6))
    # First, we plot the structure function of the continuous light curve.
    ax21 = fig2.add_subplot(121)
    ax21.plot(np.log10(edgesc[:-1]+np.diff(edgesc)/2), np.log10(np.sqrt(sc)), 'ko-',linewidth=1, markersize=3,label='1 day cadence')
    ax21.set_xlabel(r'$\log_{10}(\Delta t)$', fontsize = 14)
    ax21.set_ylabel(r'$\log_{10} \ SF$', fontsize = 14)
    
    ax22 = fig2.add_subplot(122)
    ax22.plot(np.log10(edgesc[:-1]+np.diff(edgesc)/2), np.sqrt(sc), 'ko-',linewidth=1, markersize=3,label='1 day cadence')
    ax22.set_xlabel(r'$\log_{10}(\Delta t)$', fontsize = 14)
    ax22.set_ylabel('SF', fontsize = 14)
    
    # We now plot the structure functions of OpSim light curves over the structure function of the continuous light curve.
    color=iter(plt.cm.cool(np.linspace(0,1,len(opsims))))
    i = 0 # counter
    for topp, yopp in zip(tops, yops):
        start_time = time.time()
        s, edge = sf(topp, yopp)
        c=next(color)
        ax21.plot(np.log10(edge[:-1]+np.diff(edge)/2), np.log10(np.sqrt(s)), c=c, linewidth=1, marker='o', markersize=3, label='%s'%labels[i])
        ax22.plot(np.log10(edge[:-1]+np.diff(edge)/2), np.sqrt(s), c=c, linewidth=1, marker='o', markersize=3, label='%s'%labels[i])
        i = i+1
    
    axs = [ax21, ax22]
    
    for ax in axs:
        ax.tick_params(direction='in', pad = 5)
        ax.legend()
        ax.grid(True)
        
    fig2.suptitle('Structure functions', y=0.96, fontsize=17)
    
    
    
def var_cad(yy, tt, m1=0,m2=0,m3=0,m4=0,m5=0,m6=0,m7=0,m8=0,m9=0,m10=0,m11=0,m12=0):
    """
    Parameters:
    -----------
    yy: np.array
        Flux values for the referent light curve (recommendation: if possible, use light curves with 1 day cadence for the entire duration of the survey)
    tt: np.array
        Days during the survey on which we had an observation for the referent light curve.
    m1; m2; ... ; m12: int in range [0,30], default=0 (no obseravtions)
        Cadence for each month in a year of the survey. The same combination of monthly cadences is used for each year of the survey.
        
    Returns:
    --------
    ym: np.array
        Flux values for the resulting light curve with user defined variable cadence.
    tm: np.array
        Days during the survey on which we had an observation for resulting light curve with user defined variable cadence.
    """

    
    month_cad = [m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12]
    bounds = [(0,31),(31,59),(59,90),(90,120),(120,151),(151,181),(181,212),(212,243),(243,273),(273,304),(304,334),(334,365)]

    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    l10 = []
    l11 = []
    l12 = []
    
    ls = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]
    
    for mc, b, l in zip(month_cad, bounds, ls):
        if mc != 0:
            lb, ub = b
            tm = np.arange(lb, ub, int(mc))
            l.append(tm)
            for yr in range(9):
                tm = tm + 365
                l.append(tm)
            l = np.asarray(l)
        
        elif mc == 0:
            for yr in range(10):
                l.append(np.zeros(30))
            l = np.asarray(l)
            
    tm = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12), axis=1).flatten()
    tm = tm[tm != 0]
    
    ym=[]
    for i in range(len(tm)):
        abs_vals = np.abs(tt-tm[i])
        bool_arr = abs_vals < 1
        if bool_arr.sum() != 0:
            index = (np.where(bool_arr)[0])[0]
            ym.append(yy[index])
        elif bool_arr == 0:
            ym.append(-999)
    ym=np.asarray(ym)
    tm = tm[ym!=-999]
    ym = ym[ym!=-999]
    
    return ym,tm


# get opsim cadence file
def getOpSimCadence(opsim, name, ra = 0, dec = 0, fil = 'r'):
    
    
    import lsst.sims.maf.metrics as metrics
    import lsst.sims.maf.db as db
    import lsst.sims.maf.slicers as slicers

    import lsst.sims.maf.metricBundles as mb


    
    # We are only interested in date of observation for chosen filter and field point
    colmn = 'observationStartMJD';
    
    # Directory where tmp files are going to be stored
    outDir = 'TmpDir'
    resultsDb = db.ResultsDb(outDir=outDir)
    
    # Using MAF PassMetrics to get all desired data from the chosen opsim realisation
    metric=metrics.PassMetric(cols=[colmn,'fiveSigmaDepth', 'filter'])
    
    # Select RA and DEC
    slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
    
    # Add sql constraint to select desired filter
    sqlconstraint = 'filter = \'' + fil + '\''
    
    
    # Run simulation
    bundle = mb.MetricBundle(
        metric, slicer, sqlconstraint, runName=name)
    bgroup = mb.MetricBundleGroup(
        {0: bundle}, opsim, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll();
    filters = np.unique(bundle.metricValues[0]['filter'])
    mv = bundle.metricValues[0]
    
    
    # Get dates
    mjd = len(mv[colmn])
    
    # Out file
    filepath = name + "_" + str(ra) + "_" + str(dec) + "_" + fil +".dat";
    with open(filepath, 'w') as file_handler:
        for item in mv[colmn]:
            file_handler.write("{}\n".format(item))

    mjd=np.loadtxt(filepath)
    mjd=np.sort(mjd)
    
    return mjd;


