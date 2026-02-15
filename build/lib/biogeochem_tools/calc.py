"""Routines for processing CTD data collected on small boat operations.
All routines assume a rigid naming convention and directory structure."""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import gsw

from . import ctd


def _sea_pressure(ds, Psurf=101325.):
    """Subtracts atmospheric pressure from the pressure channel,
    converting absolute pressure to sea pressure."""
    ds['Psurf'] = Psurf
    ds['Psurf'].attrs = {
        'long_name': 'sea surface atmospheric pressure',
        'standard_name': 'surface_air_pressure',
        'units': 'Pa'
    }                  
    ds['P'] = ds['P'] - ds['Psurf'] / 1.e4
    ds['P'].attrs = {
        'long_name' : 'pressure',
        'standard_name' : 'sea_water_pressure_due_to_seawater',
        'positive' : 'down',
        'units' : 'dbar',
        'data_min': np.min(ds['P'].values),
        'data_max': np.max(ds['P'].values),
        'WHPO_Variable_Name' : 'CTDPRS'
    }
    return ds


def _depth(ds):
    ds['depth'] = -xr.apply_ufunc(gsw.z_from_p, ds['P'], ds['lat'])
    ds['depth'].attrs = {
        'long_name': 'depth',
        'standard_name': 'depth',
        'positive': 'down',
        'units': 'm',
        'data_min': np.min(ds['depth'].values),
        'data_max': np.min(ds['depth'].values)
    }
    return ds
    
    
def _practical_salinity(ds):
    ds['SP'] = xr.apply_ufunc(gsw.SP_from_C, ds['C'], ds['T'], ds['P'])
    ds['SP'].attrs = {
        'long_name': 'practical salinity',
        'standard_name': 'sea_water_practical_salinity',
        'units': '1',
        'reference_scale': 'PSU',
        'data_min': np.nanmin(ds['SP'].values),
        'data_max': np.nanmax(ds['SP'].values),
        'WHPO_Variable_Name': 'CTDSAL'
    }
    return ds


def _absolute_salinity(ds):
    ds['SA'] = xr.apply_ufunc(
        gsw.SA_from_SP, 
        ds['SP'], 
        ds['P'], 
        ds['lon'], 
        ds['lat']
    )
    ds['SA'].attrs = {
        'long_name': 'absolute salinity',
        'standard_name': 'sea_water_absolute_salinity',
        'units': 'g/kg',
        'data_min': np.nanmin(ds['SA'].values),
        'data_max': np.nanmax(ds['SA'].values)
    }
    return ds


def _conservative_temperature(ds):
    ds['CT'] = xr.apply_ufunc(gsw.CT_from_t, ds['SA'], ds['T'], ds['P'])
    ds['CT'].attrs = {
        'long_name': 'conservative temperature',
        'standard_name': 'sea_water_conservative_temperature',
        'units': 'K',
        'data_min': np.nanmin(ds['CT'].values),
        'data_max': np.nanmax(ds['CT'].values)
    }
    return ds


def _potential_density_anomaly_surface(ds):
    ds['sigma0'] = xr.apply_ufunc(gsw.sigma0, ds['SA'], ds['CT'])
    ds['sigma0'].attrs = {
        'long_name': 'potential density anomaly',
        'standard_name': 'sea_water_sigma_theta',
        'units': 'kg/m^3',
        'data_min': np.nanmin(ds['sigma0'].values),
        'data_max': np.nanmax(ds['sigma0'].values)
    }
    return ds
    
    
#-------------------------------------------------------------------------------
# CTD cast calculation routines
#-------------------------------------------------------------------------------

def filter_casts(cast_flist, root_dir=None, cast_dir=None):
    """Apply a2d hold, despike and lowpass filters to raw sensor data 
    in each cast.  Finish by binning the data.  Do this for every .nc file
    in the ctd cast data directory.  If user supplies root_dir, the
    assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Filtering raw sensor data...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        # correct zero-order holds
        ds_cast = ctd.rbr_correct_zero_order_hold(ds_cast, 'P')
        ds_cast = ctd.rbr_correct_zero_order_hold(ds_cast, 'T')
        ds_cast = ctd.rbr_correct_zero_order_hold(ds_cast, 'C')
        
        # for plotting purposes, copy T, C, and P.
        ds_cast['P_raw'] = ds_cast['P']
        ds_cast['T_raw'] = ds_cast['T']
        ds_cast['C_raw'] = ds_cast['C']

        # despike data
        ds_cast = ctd.despike(ds_cast, 'T', std_limit=5, kernel_size=3)
        ds_cast = ctd.despike(ds_cast, 'C', std_limit=5, kernel_size=3)
        ds_cast = ctd.despike(ds_cast, 'P', std_limit=5, kernel_size=3)

        # for plotting purposes, copy T, C, and P.
        ds_cast['P_despike'] = ds_cast['P']
        ds_cast['T_despike'] = ds_cast['T']
        ds_cast['C_despike'] = ds_cast['C']

        # In the case of RBR instruments, the conductivity sensor has
        # a faster response than the temperature sensor and needs to
        # be "slowed down" to match the temperature sensor response
        # time.
        #ds_cast = lp_filter(ds_cast, 'T', 1.)
        #ds_cast = lp_filter(ds_cast, 'C', 1.)

        # save dataset
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))
        
    print('done.')
    
  
def bin_casts(cast_flist, root_dir=None, cast_dir=None):

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')
        
    print('Binning full resolution sensor data...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))
        
        # copy full cast ahead of binning since binning is applied
        # to all data in dataset.
        ds_cast_full = ds_cast.copy(deep=True)

        # First, bin to 0.125 meter resolution, which is the sampling
        # resolution of the RBR if the CTD is lowered perfectly vertically
        # at 1 m/s.  Second, bin to 0.5 meter resolution.  Suppress any
        # "Mean of empty slice" runtime warnings.  Binning also has the
        # effect of assigning P_bins as the coordinate.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            """
            bin_width = 0.125
            bins = np.arange(0.0625, ds_cast['P'].max(), bin_width)
            ds_cast = ctd.bin(ds_cast, bins)
            """
            bin_width = 0.250
            bins = np.arange(0.125, ds_cast['P'].max(), bin_width)
            ds_cast = ctd.bin(ds_cast, bins)

        # copy attributes to P_bins
        ds_cast['P_bins'].attrs = ds_cast_full['P_raw'].attrs
        
        # restore scalar data 
        ds_cast['time'] = ds_cast_full['time']
        ds_cast['lat'] = ds_cast_full['lat']
        ds_cast['lon'] = ds_cast_full['lon']
        ds_cast['Psurf'] = ds_cast_full['Psurf']
        ds_cast['instrument1'] = ds_cast_full['instrument1']
        
        # restore full resolution data
        ds_cast['P_raw'] = ds_cast_full['P_raw']
        ds_cast['T_raw'] = ds_cast_full['T_raw']
        ds_cast['C_raw'] = ds_cast_full['C_raw']
        ds_cast['P_despike'] = ds_cast_full['P_despike']
        ds_cast['T_despike'] = ds_cast_full['T_despike']
        ds_cast['C_despike'] = ds_cast_full['C_despike']
        
        # clean up pressure variables
        ds_cast = ds_cast.drop_vars(['P'])
        ds_cast = ds_cast.rename({'P_bins': 'P'})
        ds_cast = ds_cast.swap_dims({'P': 'z'})
        
        # save dataset
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))

    print('done.')


def derive_insitu_properties(cast_flist, root_dir=None, cast_dir=None):
    """Calculate derived in-situ water properties fore each cast.  Do
    this for every .nc file in the ctd cast data directory.  If user
    supplies root_dir, the assumption is that all directories follow
    the standard pattern. Otherwise, directories are needed for
    cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Calculating derived in situ properties...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        # calculate derived parameters
        ds_cast = _depth(ds_cast)
        ds_cast = _practical_salinity(ds_cast)
        ds_cast = _absolute_salinity(ds_cast)
        ds_cast = _conservative_temperature(ds_cast)

        # save dataset
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))

    print('done.')


def extract_niskin_salts(btl_fname, raw_fname, niskin_length, root_dir=None,
    btl_dir=None, raw_dir=None, cast_dir=None, plots_dir=None):
    """Calculate average in-situ temperature and salinity in Niskin
    bottle at the time it was closed.  If user supplies root_dir, the
    assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for btl_dir, raw_dir, cast_dir,
    and plots_dir"""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        raw_dir = os.path.join(root_dir, 'ctd', 'raw')
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')
        plots_dir = os.path.join(root_dir, 'btl', 'plots', 'ctdsal')
    if not(os.path.isdir(plots_dir)):
        os.mkdir(plots_dir)

    print('Calculating average niskin in situ properties...', end='',
        flush=True)

    # Load files
    ds_raw = xr.load_dataset(os.path.join(raw_dir, raw_fname))
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))
    print(ds_raw)
    # Cycle through the bottles
    ctdprs = []
    ctdtmp = []
    ctdsal = []
    for cast in ds_btl['cast_number'].values:
        if ds_btl.sel(cast_number=cast)['cast_f']==2:

            # Check for valid tnisk.  Sometimes the RBR didn't upload the
            # last few datapoints of the day and the niskin soak and upcast
            # of the last cast weren't saved.  In this case, there is no
            # tnisk and ctdsal etc cannot be calculated.  In this case,
            # enter NaN for those values.
            tnisk = ds_btl.sel(cast_number=cast)['time']
            missing_niskin_time = np.isnat(tnisk)

            # Load ctd cast file
            expocode = ds_btl.sel(cast_number=cast)['expocode'].values
            cast_fname = '{0:s}_{1:03d}_ct1.nc'.format(expocode, cast)
            ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

            # obtain CTD pressure of niskin
            if missing_niskin_time:
                Pnisk = xr.DataArray(np.NaN)
                Pnisk = Pnisk.assign_coords({'cast_number': cast})
            else:
                Praw = ds_raw['P'].sel(timestamp=tnisk, method='nearest')
                Psens = Praw - ds_cast['Pair'] / 1.e4
                Pnisk = Psens - ds_btl.sel(cast_number=cast)['niskin_height']
                Pnisk = Pnisk.drop('timestamp')

            # extract niskin pressure range sensor data from ctd downcast
            if missing_niskin_time:
                Tnisk = xr.DataArray(np.NaN)
                SPnisk = xr.DataArray(np.NaN)
            else:
                #ds_cast_nisk = ds_cast.sel(P=[ctdprs - 0.50 * h : ctdprs + 0.50 * h], method='nearest')
                ds_cast_nisk = ds_cast.sel(P=slice(Pnisk - 0.75 * niskin_length,
                    Pnisk + 0.75 * niskin_length))
                Tnisk = ds_cast_nisk['T'].mean()
                SPnisk = ds_cast_nisk['SP'].mean()
            Tnisk = Tnisk.assign_coords({'cast_number': cast})
            SPnisk = SPnisk.assign_coords({'cast_number': cast})

            # append extracted values to list; include proper dimensions
            ctdprs.append(Pnisk)
            ctdtmp.append(Tnisk)
            ctdsal.append(SPnisk)

            # plot extraction results
            if not missing_niskin_time:
                fig, (axT, axS) = plt.subplots(nrows=1, ncols=2, figsize=(6,8),
                                               sharey=True, constrained_layout=True)
                axS.plot(ds_cast['SP'], ds_cast['P'], 'k-')
                axS.plot(ds_cast_nisk['SP'], ds_cast_nisk['P'], 'r-', lw=3)
                axS.axhline(Pnisk, linewidth=0.5, color='k', linestyle='--')
                axS.axhline(Psens, linewidth=1, color='k', linestyle='-')
                axS.axvline(SPnisk, linewidth=0.5, color='k', linestyle='--')
                axT.plot(ds_cast['T'], ds_cast['P'], 'k-')
                axT.plot(ds_cast_nisk['T'], ds_cast_nisk['P'], 'r-', lw=3)
                axT.axhline(Pnisk, linewidth=0.5, color='k', linestyle='--')
                axT.axhline(Psens, linewidth=1, color='k', linestyle='-')
                axT.axvline(Tnisk, linewidth=0.5, color='k', linestyle='--')

                Pmax = np.ceil((ds_cast['P'].max() / 5.)) * 5.
                axT.set_ylim([0., Pmax.values])

                axT.set_xlabel(r'$T$ / $^\circ$C [ITS-90]', fontsize=9)
                axT.set_ylabel('$P$ / dbar', fontsize=9)
                axS.set_xlabel(r'$S_\mathrm{P}$ [PSU]', fontsize=9)
                axT.set_title('Cast: {0:d}\nStn: {1:s} ({2:s})'.format(ds_cast.cast_number,
                              ds_cast.station_name, ds_cast.station_id), loc='left',
                              fontsize=9)
                date_str = ds_cast['time'].dt.strftime('%Y-%m-%d').values
                time_str = ds_cast['time'].dt.strftime('%H:%M').values
                axS.set_title('UTC {0:s}\n{1:s}'.format(date_str, time_str),
                              loc='right', fontsize=9)
                axT.invert_yaxis()

                plot_fname = '{0:s}_{1:03d}.pdf'.format(ds_cast.expocode, ds_cast.cast_number)
                fig.savefig(os.path.join(plots_dir, plot_fname))
                plt.close(fig)

    # Create DataArrays from ctdprs, ctdtmp, and ctdsal result lists, assign
    # attributes and merge into the full bottle dataset.
    da_ctdprs = xr.concat(ctdprs, dim='cast_number')
    da_ctdprs.attrs = {'long_name': 'sensor pressure',
                       'standard_name' : 'sea_water_pressure_due_to_seawater',
                       'positive' : 'down',
                       'units' : 'dbar',
                       'data_min': np.nanmin(da_ctdprs.values),
                       'data_max': np.nanmax(da_ctdprs.values),
                       'WHPO_Variable_Name' : 'CTDPRS'}
    da_ctdsal = xr.concat(ctdsal, dim='cast_number')
    da_ctdsal.attrs = {'long_name': 'sensor practical salinity',
                       'standard_name': 'sea_water_practical_salinity',
                       'units': '',
                       'data_min': np.nanmin(da_ctdsal.values),
                       'data_max': np.nanmax(da_ctdsal.values),
                       'WHPO_Variable_Name': 'CTDSAL'}
    da_ctdtmp = xr.concat(ctdtmp, dim='cast_number')
    da_ctdtmp.attrs = {'long_name': 'sensor temperature',
                       'standard_name': 'sea_water_temperature',
                       'units': 'C (ITS-90)',
                       'data_min': np.nanmin(da_ctdtmp.values),
                       'data_max': np.nanmax(da_ctdtmp.values),
                       'WHPO_Variable_Name': 'CTDTMP'}
    ds_btl = xr.merge([ds_btl,
                       {'ctdprs': da_ctdprs},
                       {'ctdsal': da_ctdsal},
                       {'ctdtmp': da_ctdtmp}])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))
    print('done.')
    

#-------------------------------------------------------------------------------
# Bottle calculation routines
#-------------------------------------------------------------------------------

def recalc_alkalinity(btl_fname, dic_fname, root_dir=None, btl_dir=None,
    dic_dir=None, titration_dir=None):
    
    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        dic_dir = os.path.join(btl_dir, 'dic')
        
    print('Recalculating bottle alkalinity')
    
    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))
    
    
    return
    