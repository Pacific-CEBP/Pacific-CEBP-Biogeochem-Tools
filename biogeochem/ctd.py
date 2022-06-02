"""ctd.py"""

import warnings
import numpy as np
import scipy as sci
from scipy import signal
import xarray as xr
import gsw


"""RBR specific tools"""

def rbr_correct_zero_order_hold(ds, channel):
    """The analog-to-digital (A2D) converter on RBR instruments must
    recalibrate periodically.  In the time it takes for the calibration
    to finish, one or more samples are missed.  The onboard firmware
    fills the missed sample with the same data measured during the
    previous sample, a simple technique called a zero-order hold.

    The function identifies zero-hold points by looking for where
    consecutive differences for each channel are equal to zero, and
    replaces them with NaN."""

    ds_out = ds.copy()

    hold_idx = np.where(np.diff(ds[channel])==0.0)[0] + 1
    ds_out[channel][hold_idx] = np.NaN
    return ds_out
    

"""Despiking, filtering, binning type routines"""

def despike(ds, channel, std_limit=3, kernel_size=3):

    """Clean spikes in raw sensor data using median filter."""

    # calculate smoothed and residual signals
    raw = ds[channel].values
    smooth = signal.medfilt(raw, kernel_size=kernel_size)
    resid = raw - smooth

    # obtain indices where residual deviates by more than
    # <std_limit> standard deviations.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        ix = np.where(np.abs(resid/np.nanstd(resid))>std_limit)

    # replace spikes with NaN values
    ds[channel][ix] = np.NaN

    return ds


def lp_filter(ds, channel, tau):
    # tau is characteristic response time of sensor
    ds_out = ds

    # Nyquist frequency = 0.5 * sample frequency
    fs = 8. # Hz
    nyq = 0.5 * 8
    low = (1. / tau) / nyq

    # Create an order 5 lowpass butterworth filter
    b, a = butter(5, low, btype='lowpass')

    ds_out[channel] = filtfilt(b, a, ds[channel].values)

    return ds_out


def bin(ds, bins):
    """Bin all variables by pressure."""

    # calculate bin centers
    bin_centers = np.array([(bins[n] + bins[n+1]) / 2 for n in 
                           range(np.size(bins) - 1)])

    # bin the data
    ds = ds.groupby_bins('P', bins, labels=bin_centers, precision=4,
                         include_lowest=True, restore_coord_dims=True).mean(dim=xr.ALL_DIMS, skipna=True)

    # to do: pass thru all attributes
    
    return ds


"""Routines to calculate derived parameters"""

def sea_pressure(ds, Pair=101325.):
    """Subtracts atmospheric pressure from the pressure channel,
    converting absolute pressure to sea pressure."""
    ds['Pair'] = Pair
    ds['Pair'].attrs = {'long_name': 'atmospheric pressure',
                        'standard_name': 'surface_air_pressure',
                        'units': 'Pa'}
    ds['P'] = ds['P'] - ds['Pair'] / 1.e4
    ds['P'].attrs = {'long_name' : 'pressure',
                     'standard_name' : 'sea_water_pressure_due_to_seawater',
                     'positive' : 'down',
                     'units' : 'dbar',
                     'data_min': np.min(ds['P'].values),
                     'data_max': np.max(ds['P'].values),
                     'WHPO_Variable_Name' : 'CTDPRS'}
    return ds


def depth(ds):
    ds['z'] = -xr.apply_ufunc(gsw.z_from_p, ds['P'], ds['lat'])
    ds['z'].attrs = {'long_name': 'depth',
                     'standard_name': '',
                     'positive': 'down',
                     'units': 'm',
                     'data_min': np.min(ds['z'].values),
                     'data_max': np.min(ds['z'].values),
                     'WHPO_Variable_Name': ''}
    return ds
    
    
def practical_salinity(ds):
    ds['SP'] = xr.apply_ufunc(gsw.SP_from_C, ds['C'], ds['T'], ds['P'])
    ds['SP'].attrs = {'long_name': 'practical salinity',
                      'standard_name': 'sea_water_practical_salinity',
                      'units': '',
                      'data_min': np.nanmin(ds['SP'].values),
                      'data_max': np.nanmax(ds['SP'].values),
                      'WHPO_Variable_Name': 'CTDSAL'}
    return ds


def absolute_salinity(ds):
    ds['SA'] = xr.apply_ufunc(gsw.SA_from_SP, ds['SP'], ds['P'], ds['lon'], 
                              ds['lat'])
    ds['SA'].attrs = {'long_name': 'absolute salinity',
                      'standard_name': 'sea_water_absolute_salinity',
                      'units': 'g/kg',
                      'data_min': np.nanmin(ds['SA'].values),
                      'data_max': np.nanmax(ds['SA'].values),
                      'WHPO_Variable_Name': ''}
    return ds


def conservative_temperature(ds):
    ds['CT'] = xr.apply_ufunc(gsw.CT_from_t, ds['SA'], ds['T'], ds['P'])
    ds['CT'].attrs = {'long_name': 'conservative temperature',
                      'standard_name': 'sea_water_conservative_temperature',
                      'units': 'K',
                      'data_min': np.nanmin(ds['CT'].values),
                      'data_max': np.nanmax(ds['CT'].values),
                      'WHPO_Variable_Name': ''}
    return ds


def potential_density_anomaly_surface(ds):
    ds['sigma0'] = xr.apply_ufunc(gsw.sigma0, ds['SA'], ds['CT'])
    ds['sigma0'].attrs = {'long_name': 'potential density anomaly',
                          'standard_name': 'sea_water_sigma_theta',
                          'units': 'kg/m^3',
                          'data_min': np.nanmin(ds['sigma0'].values),
                          'data_max': np.nanmax(ds['sigma0'].values)}
    return ds





