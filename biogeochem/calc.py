"""Routines for processing CTD data collected on small boat operations.
All routines assume a rigid naming convention and directory structure."""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from ocean.io import rbr
from ocean.calc import ctd


#-------------------------------------------------------------------------------
# Cast extraction and calculation routines
#-------------------------------------------------------------------------------

def extract_casts(ds_raw, df_event_log, root_dir=None, cast_dir=None):
    """Extract casts based on the event log, transform the data,
    plot the results of the transformations (for QC purposes).
    If user supplies root_dir, the assumption is that all directories
    follow the standard pattern.  Otherwise, directories are needed
    for cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')
    if not(os.path.isdir(cast_dir)):
        os.mkdir(cast_dir)

    print('Extracting casts...', end='', flush=True)
    cast_flist = []
    for cast_info in df_event_log.itertuples():
        if cast_info.cast_f == 2:
            ds_cast = rbr.extract_cast(ds_raw, [cast_info.tstart,
                                                cast_info.tend])

            # add geolocation
            ds_cast['lat'] = cast_info.lat
            ds_cast['lat'].attrs = {'long_name': 'latitude',
                                    'standard_name': 'latitude',
                                    'positive' : 'north',
                                    'units': 'degree_north'}
            ds_cast['lon'] = cast_info.lon
            ds_cast['lon'].attrs = {'long_name': 'longitude',
                                    'standard_name': 'longitude',
                                    'positive' : 'east',
                                    'units': 'degree_east'}

            # correct for atmospheric pressure
            half_second = np.timedelta64(500, 'ms')
            tslice = slice(cast_info.tair - half_second,
                           cast_info.tair + half_second)
            Pair = (ds_raw['P'].sel(timestamp=tslice).mean(skipna=True)) * 1.e4
            ds_cast = ctd.sea_pressure(ds_cast, Pair)

            # create copies of T, C, and P to be used for QC plots, as
            # the default action of the ctd calculation routines is to
            # overwrite the inputs.
            ds_cast['P_raw'] = ds_cast['P']
            ds_cast['T_raw'] = ds_cast['T']
            ds_cast['C_raw'] = ds_cast['C']

            # add cast attributes
            ds_cast.attrs['station_id'] = cast_info.stn
            ds_cast.attrs['station_name'] = cast_info.name
            ds_cast.attrs['cast_number'] = np.int(cast_info.Index)

            # save netcdf cast file
            cast_fname = '{0:s}_{1:03d}_ct1.nc'.format(ds_raw.expocode,
                cast_info.Index)
            cast_flist.append(cast_fname)
            ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))
    print('done.')

    return cast_dir, cast_flist


def filter_casts(cast_flist, root_dir=None, cast_dir=None):
    """Apply despike and lowpass filters to raw sensor data in each
    cast.  Finish by binning the data.  Do this for every .nc file
    in the ctd cast data directory.  If user supplies root_dir, the
    assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Filtering raw sensor data...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

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

            bin_width = 0.125
            bins = np.arange(0.0625, ds_cast['P'].max(), bin_width)
            ds_cast = ctd.bin(ds_cast, bins)
            bin_width = 0.250
            bins = np.arange(0.125, ds_cast['P'].max(), bin_width)
            ds_cast = ctd.bin(ds_cast, bins)

        # restore scalars (ideally these would be ignored in binning
        # function.  TODO!!)
        ds_cast['lat'] = ds_cast_full['lat']
        ds_cast['lon'] = ds_cast_full['lon']
        ds_cast['Pair'] = ds_cast_full['Pair']
        ds_cast['time'] = ds_cast_full['time']

        # restore full resolution data
        ds_cast['P_raw'] = ds_cast_full['P_raw']
        ds_cast['T_raw'] = ds_cast_full['T_raw']
        ds_cast['C_raw'] = ds_cast_full['C_raw']
        ds_cast['P_despike'] = ds_cast_full['P_despike']
        ds_cast['T_despike'] = ds_cast_full['T_despike']
        ds_cast['C_despike'] = ds_cast_full['C_despike']

        # restore attributes
        ds_cast.attrs = ds_cast_full.attrs
        ds_cast['P_bins'].attrs = ds_cast_full['P'].attrs
        ds_cast['T'].attrs = ds_cast_full['T'].attrs
        ds_cast['C'].attrs = ds_cast_full['C'].attrs
        ds_cast['V0'].attrs = ds_cast_full['V0'].attrs
        ds_cast['V1'].attrs = ds_cast_full['V1'].attrs

        # for plotting purposes, duplicate T and C
        ds_cast['T_bins'] = ds_cast['T']
        ds_cast['C_bins'] = ds_cast['C']

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
        ds_cast = ctd.depth(ds_cast)
        ds_cast = ctd.practical_salinity(ds_cast)
        #ds_cast = ctd.absolute_salinity(ds_cast)
        #ds_cast = ctd.conservative_temperature(ds_cast)

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
            if np.isnat(tnisk):
                missing_niskin_time = True
            else:
                missing_niskin_time = False

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
    