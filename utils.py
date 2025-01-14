##############################
###      FlatsimCorr       ###
###          ---           ###
###    L. Marconato 2024   ###
##############################

import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
from tqdm import tqdm
import datetime
from scipy.linalg import lstsq
from scipy.stats import linregress
#from scipy.signal import lombscargle

# Some figure parameters...
import matplotlib
if os.environ["TERM"].startswith("screen"): # cluster
    matplotlib.use('Agg')
else:  # laptop
    matplotlib.rc('font', **{'family':'Arial', 'weight':'medium', 'size':8, 'style':'normal'})
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['figure.dpi'] = 300

#######################################################################
################# A BUNCH OF UTILITARY FUNCTIONS ######################
#######################################################################

def remove_outliers(ramps, sigma=10.):
        median = np.nanmedian(ramps)
        std = np.nanstd(ramps)
        outliers = (np.abs(ramps-median) > sigma*std)
        Noutliers = np.count_nonzero(outliers)
        ramps[np.abs(ramps-median) > sigma*std] = np.nan
        return ramps

def sliding_median(x, y, window=0.3):
    y_filt = np.zeros_like(y)
    for i in range(len(x)):
        sel = np.logical_and(x > x[i]-window/2, x < x[i]+window/2)
        y_filt[i] = np.nanmedian(y[sel])
    return y_filt

def iono2key(model):
    '''
    Basically remove the IGS suffix if it exists.

    Args:
        * model : full name of model (e.g. 'IGS_JPL' or 'JPLD')
    
    Returns:
        * The key to access ra_ramps and az_ramps dicts (e.g. 'JPL' or 'JPLD')
    '''
    if model.startswith('IGS'):
        return model[4:]
    else:
        return model

def linear_fit(dates, data0, min_date=None, max_date=None):
    '''
    Fit a linear trend in time.

    Args:
        * dates : list or 1d array of time positions
        * data  : list or 1d array of data to fit (same size as dates)
    
    Kwargs:
        * min_date  : only fit data after this date
        * max_date  : only fit data before this date

    Returns:
        * constant, slope
    '''
    data = np.copy(data0)
    if min_date is not None:
        data[dates<min_date] = np.nan
    if max_date is not None:
        data[dates>max_date] = np.nan

    valid = ~np.isnan(data)
    vel,cst,_,_,_ = linregress(dates[valid], data[valid])
    return cst, vel

def linear_fit_sigma(dates, data0, min_date=None, max_date=None):
    '''
    Fit a linear trend in time.

    Args:
        * dates : list or 1d array of time positions
        * data  : list or 1d array of data to fit (same size as dates)
    
    Kwargs:
        * min_date  : only fit data after this date
        * max_date  : only fit data before this date

    Returns:
        * constant, slope, sigma
    '''
    data = np.copy(data0)
    if min_date is not None:
        data[dates<min_date] = np.nan
    if max_date is not None:
        data[dates>max_date] = np.nan

    valid = ~np.isnan(data)
    vel,cst,_,_,_ = linregress(dates[valid], data[valid])

    rmse = np.sqrt(np.mean(np.square(cst+dates[valid]*vel - data[valid])))
    sigma = rmse/(np.sqrt(len(dates[valid]) - 2)*np.nanstd(dates[valid]))

    return cst, vel, sigma

def linear_seasonal_fit(dates, data0, min_date=None, max_date=None):
    '''
    Fit a linear trend in time plus a sinusoidal signal of period 1 (year).

    Args:
        * dates : list or 1d array of time positions
        * data  : list or 1d array of data to fit (same size as dates)
    
    Kwargs:
        * min_date  : only fit data after this date
        * max_date  : only fit data before this date

    Returns:
        * constant, slope, sin amplitude, cos amplitude
    '''
    data = np.copy(data0)
    if min_date is not None:
        data[dates<min_date] = np.nan
    if max_date is not None:
        data[dates>max_date] = np.nan
        
    valid = ~np.isnan(data)
    A = np.c_[np.ones(np.count_nonzero(valid)), dates[valid], np.sin(2*np.pi*dates[valid]), np.cos(2*np.pi*dates[valid])]
    C,_,_,_ = lstsq(A, data[valid])

    return C[0], C[1], C[2], C[3]

def linear_seasonal_fit_sigma(dates, data0, min_date=None, max_date=None):
    '''
    Fit a linear trend in time plus a sinusoidal signal of period 1 (year).

    Args:
        * dates : list or 1d array of time positions
        * data  : list or 1d array of data to fit (same size as dates)
    
    Kwargs:
        * min_date  : only fit data after this date
        * max_date  : only fit data before this date

    Returns:
        * constant, slope, sin amplitude, cos amplitude, sigma
    '''
    data = np.copy(data0)
    if min_date is not None:
        data[dates<min_date] = np.nan
    if max_date is not None:
        data[dates>max_date] = np.nan
        
    valid = ~np.isnan(data)
    Nvalid = np.count_nonzero(valid)
    A = np.c_[np.ones(Nvalid), dates[valid], np.sin(2*np.pi*dates[valid]), np.cos(2*np.pi*dates[valid])]
    C,_,_,_ = lstsq(A, data[valid])

    model = C[0] + dates[valid]*C[1] + np.sin(2*np.pi*dates[valid])*C[2] + np.cos(2*np.pi*dates[valid])*C[3]
    rmse = np.sqrt(np.mean(np.square(model - data[valid])))
    sigma = rmse/(np.sqrt(len(dates[valid]) - 4)*np.nanstd(dates[valid]))

    return C[0], C[1], C[2], C[3], sigma

def R2(data1, data2):
    '''
    Computes the r-squared coefficient (determination coefficient) between two datasets.

    Args:
        * data1 and data2 : two lists or 1d-array with the same size

    Returns:
        * R^2
    '''
    valid = ~np.logical_or(np.isnan(data1), np.isnan(data2))
    data1_sel = data1[valid]
    data2_sel = data2[valid]
    _,_,r_val,_,_ = linregress(data1_sel, data2_sel)
    return r_val**2

def RMSE(dif):
    '''
    Computes the root-mean-square of the a series.

    Args:
        * dif : array (e.g., difference between two datasets to compute a proper RMSE)

    Returns:
        * RMSE
    '''
    return np.sqrt(np.nanmean(np.square(dif)))

def detrend(x, y):
    a = np.c_[x, y]
    a = a[~np.isnan(a).any(axis=1)]
    lin, cst = np.polyfit(a[:,0], a[:,1], 1)
    return y - (cst + np.array(x)*lin)

def correlate(x1, y1, x2, y2, window=30/365):

    # Remove NaNs
    x1 = x1[~np.isnan(y1)]
    y1 = y1[~np.isnan(y1)]
    x2 = x2[~np.isnan(y2)]
    y2 = y2[~np.isnan(y2)]

    # Keep only common values
    common_values = np.intersect1d(x1, x2)
    indices_in_x1 = np.where(np.isin(x1, common_values))[0]
    indices_in_x2 = np.where(np.isin(x2, common_values))[0]

    x1 = x1[indices_in_x1]
    y1 = y1[indices_in_x1]
    y2 = y2[indices_in_x2]

    N = len(x1)
    rho = np.zeros(N)
    slope = np.zeros(N)

    for i in range(N):
        xmin = x1[i]-window/2
        xmax = x1[i]+window/2
        sel = np.logical_and(x1 > xmin, x1 < xmax)

        if np.count_nonzero(sel) >= 5:
            rho[i] = np.corrcoef(y1[sel], y2[sel])[0,1]
            slope[i],_,_,_,_ = linregress(y2[sel], y1[sel])

    return x1, np.where(rho==0., np.nan, rho), np.where(slope==0., np.nan, slope)

def dif_ramps(dict_x, dict_y, data1, data2):

    x1 = dict_x[data1]
    y1 = dict_y[data1]
    x2 = dict_x[data2]
    y2 = dict_y[data2]

    # Keep only common values
    common_values = np.intersect1d(x1, x2)
    indices_in_x1 = np.where(np.isin(x1, common_values))[0]
    indices_in_x2 = np.where(np.isin(x2, common_values))[0]

    x1 = x1[indices_in_x1]
    y1 = y1[indices_in_x1]
    y2 = y2[indices_in_x2]

    return x1, y1-y2