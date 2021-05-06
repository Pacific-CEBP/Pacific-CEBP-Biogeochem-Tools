"""Routines for processing CTD data collected on small boat operations.
All routines assume a rigid naming convention and directory structure."""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ocean.io import rbr
from ocean.calc import ctd


def load_event_log(fname):
    """Loads the event log .csv file.  Must contain specific columns
    in order to be successful.  Will add help to list those columns
    if they are not found."""

    def _to_datetime(ts):
        return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S.%f',errors='coerce')

    print('Loading event log "{0}"...'.format(fname), end='', flush=True)
    df_event_log = pd.read_csv(
        fname, 
        comment='#',
        index_col='cast',
        date_parser=_to_datetime, 
        parse_dates={'tair': ['date', 'time_air'],
                     'tstart': ['date', 'time_start'],
                     'tend': ['date', 'time_end'],
                     'tnisk': ['date', 'time_nisk']},
        dtype={'cast': np.int16, 
               'stn': str,
               'name': str, 
               'lat': np.single, 
               'lon': np.single, 
               'sampling_platform': str,
               'cast_f': np.int8,
               'niskin': np.single,
               'nisk_f': np.int8,
               'niskin_height': np.single,
               'wire_angle': np.single,
               'sampler': str,
               'dic_btl': str,
               'dic_dup': str,
               'nut_btl': str,
               'nut_dup': str,
               'salt_btl': str,
               'salt_dup': str,
               'weather': str,
               'sea': str,
               'notes': str}
        )
    print('done.')

    return df_event_log


def import_merge_rbr(rsk_flist, expocode, root_dir=None, raw_dir=None,
                     rsk_dir=None):
    """Import multiple raw ctd files in .rsk format from RBR CTD. If
    user supplies root_dir, the assumption is that all directories
    follow the standard pattern.  Otherwise, directories are needed
    for rsk_dir and raw_dir."""

    if root_dir is not None:
        raw_dir = os.path.join(root_dir, 'ctd', 'raw')
        rsk_dir = os.path.join(raw_dir, 'rbr', 'rsk')

    print('Importing / merging raw CTD data for {0:s}...'.format(expocode),
          end='', flush=True)
    ds_raw = rbr.multi_read_rsk([os.path.join(rsk_dir, rsk_fname)
                                 for rsk_fname in rsk_flist])
    val, idx = np.unique(ds_raw.timestamp, return_index=True)
    ds_raw = ds_raw.isel(timestamp=idx) # trim the rare duplicate index values
    ds_raw.attrs['expocode'] = expocode
    print('done.', flush=True)

    print('Correcting zero-order holds in raw traces...', end='', flush=True)
    ds_raw = rbr.correct_zero_order_hold(ds_raw, 'P')
    ds_raw = rbr.correct_zero_order_hold(ds_raw, 'T')
    ds_raw = rbr.correct_zero_order_hold(ds_raw, 'C')
    ds_raw = rbr.correct_zero_order_hold(ds_raw, 'V0')
    ds_raw = rbr.correct_zero_order_hold(ds_raw, 'V1')
    print('done.', flush=True)

    print('Saving merged CTD data...', end='', flush=True)
    raw_nc_fname = '{0:s}_raw.nc'.format(expocode)
    raw_csv_fname = '{0:s}_raw.csv'.format(expocode)
    ds_raw.to_netcdf(os.path.join(raw_dir, raw_nc_fname), 'w')
    ds_raw.to_dataframe().to_csv(os.path.join(raw_dir, raw_csv_fname))
    print('done.')

    return ds_raw


def import_merge_aml(aml_flist, expocode, root_dir=None, raw_dir=None,
                     aml_dir=None):
    """Import multiple raw ctd files in .csv format AML CTD. If
    user supplies root_dir, the assumption is that all directories
    follow the standard pattern.  Otherwise, directories are needed
    for aml_dir and raw_dir."""

    if root_dir is not None:
        raw_dir = os.path.join(root_dir, 'ctd', 'raw')
        aml_dir = os.path.join(raw_dir, 'aml')


def create_bottle_file(df_event_log, expocode, root_dir=None, btl_dir=None):
    """Create bottle file for the entire cruise.  This file contains
    the CTD-derived in-situ properties of the Niskin and all analytical
    results from discrete samples.  If user supplies root_dir, the
    assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for btl_dir."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')

    # create bottle dataset from the event log
    print('Creating bottle file...', end='', flush=True)
    ds_btl = xr.Dataset.from_dataframe(df_event_log)
    ds_btl['expocode'] = xr.full_like(ds_btl['cast'], expocode, dtype='object')
    ds_btl = ds_btl.drop(['tair', 'tstart', 'tend', 'weather', 'sea', 'notes'])
    ds_btl = ds_btl.where(ds_btl['nisk_f']!=1, drop=True) # drop ctd-only casts

    # Rename and recast (change data type)
    ds_btl = ds_btl.rename({'cast': 'cast_number', 'tnisk': 'time',
        'stn': 'station_id', 'name': 'station_name', 'niskin': 'sample_number',
        'nisk_f': 'sample_number_flag_w'})
    ds_btl['cast_number'] = ds_btl['cast_number'].astype(np.int16)
    ds_btl['sample_number_flag_w'] = ds_btl['sample_number_flag_w'].astype(np.int8)
    ds_btl['cast_f'] = ds_btl['cast_f'].astype(np.int8)

    # Assign attributes
    ds_btl['cast_number'].attrs = {'long_name': 'cast number',
                                   'WHPO_Variable_Name': 'CASTNO'}
    ds_btl['station_id'].attrs = {'long_name': 'station ID',
                                  'WHPO_Variable_Name': 'STNNBR'}
    ds_btl['sample_number'].attrs = {'long_name': 'sample number',
                                     'WHPO_Variable_Name': 'SAMPNO'}
    ds_btl['lat'].attrs = {'long_name': 'latitude',
                           'standard_name': 'latitude',
                           'positive' : 'north',
                           'units': 'degree_north',
                           'WHPO_Variable_Name': 'LATITUDE'}
    ds_btl['lon'].attrs = {'long_name': 'longitude',
                           'standard_name': 'longitude',
                           'positive' : 'east',
                           'units': 'degree_east',
                           'WHPO_Variable_Name': 'LONGITUDE'}

    # save netcdf bottle file
    btl_fname = '{0:s}_hy1.nc'.format(expocode)
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))
    print('done.')

    return btl_fname


def merge_bottle_salts(btl_fname, salinity_fname, root_dir=None, btl_dir=None,
    salinity_dir=None):
    """Merge results from bottle salinity analyses.  If user supplies
    root_dir, the assumption is that all directories follow the
    standard pattern.  Otherwise, directories are needed for bottle
    and salinity files."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        salinity_dir = os.path.join(btl_dir, 'salinity')

    print('Merging bottle salinities...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Load salinity file
    df_salts = pd.read_csv(os.path.join(salinity_dir, salinity_fname),
        comment='#')
    ds_salts = xr.Dataset.from_dataframe(df_salts)
    ds_salts = ds_salts.rename({'cast': 'cast_number'})

    # Average duplicates
    ds_salts_mean = ds_salts.groupby('cast_number').mean()

    #Add quality flags 
    ds_salt_flag = ds_salts['salinity_flag_ios']

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_salts_mean['salinity'],ds_salt_flag])

    # Attach metadata
    ds_btl['salinity'].attrs = {'long_name': 'practical salinity',
                                'standard_name': 'sea_water_practical_salinity',
                                'units': '',
                                'data_min': np.nanmin(ds_btl['salinity'].values),
                                'data_max': np.nanmax(ds_btl['salinity'].values),
                                'WHPO_Variable_Name': 'SALNTY'}

    ds_btl['salinity_flag_ios'].attrs = {'long_name': 'practical salinity quality',
                                'standard_name': 'sea_water_practical_salinity status_flag',
                                'units': '',
                                'valid_range': (0,9),
                                'flag_values': '0,1,2,3,4,5,6,9',
                                'flag_meanings': 'Acceptable, Sample not analyzed, Acceptable, Questionable (probably good), Poor (probably bad), Not reported as noted bad during analysis, Mean of replicates, Not collected',
                                'WHPO_Variable_Name': 'NA'}    
                                
    # Drop bottle number columns
    ds_btl = ds_btl.drop(['salt_btl', 'salt_dup'])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))

    print('done.')


def merge_nutrients(btl_fname, nutrients_fname, root_dir=None, btl_dir=None,
    nutrients_dir=None):
    """Merge results from nutrient analyses.  If user supplies
    root_dir, the assumption is that all directories follow the
    standard pattern.  Otherwise, directories are needed for bottle
    and nutrient files."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        nutrients_dir = os.path.join(btl_dir, 'nutrients')

    print('Merging nutrient analyses...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Load nutrient file
    df_nuts = pd.read_csv(os.path.join(nutrients_dir, nutrients_fname),
        comment='#')
    ds_nuts = xr.Dataset.from_dataframe(df_nuts)
    ds_nuts = ds_nuts.rename({'cast': 'cast_number'})

    # Average duplicates
    ds_nuts_mean = ds_nuts.groupby('cast_number').mean()

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_nuts_mean['nitrate'],
                       ds_nuts_mean['silicate'],
                       ds_nuts_mean['phosphate']])

    # Attach metadata
    ds_btl['nitrate'].attrs = {'long_name': 'dissolved nitrate + nitrite concentration',
                               'standard_name': 'mole_concentration_of_nitrate_and_nitrite_in_sea_water',
                               'units': 'mol m-3',
                               'data_min': np.nanmin(ds_btl['nitrate'].values),
                               'data_max': np.nanmax(ds_btl['nitrate'].values),
                               'WHPO_Variable_Name': 'NO2+NO3'}

    ds_btl['silicate'].attrs = {'long_name': 'dissolved silicate concentration',
                               'standard_name': 'mole_concentration_of_silicate_in_sea_water',
                               'units': 'mol m-3',
                               'data_min': np.nanmin(ds_btl['silicate'].values),
                               'data_max': np.nanmax(ds_btl['silicate'].values),
                               'WHPO_Variable_Name': 'SILCAT'}

    ds_btl['phosphate'].attrs = {'long_name': 'dissolved phosphate concentration',
                               'standard_name': 'mole_concentration_of_phosphate_in_sea_water',
                               'units': 'mol m-3',
                               'data_min': np.nanmin(ds_btl['phosphate'].values),
                               'data_max': np.nanmax(ds_btl['phosphate'].values),
                               'WHPO_Variable_Name': 'PHSPHT'}

    # Drop bottle numbers
    ds_btl = ds_btl.drop(['nut_btl', 'nut_dup'])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))
    print('done.')


def merge_dic(btl_fname, dic_fname, root_dir=None, btl_dir=None,
    dic_dir=None):
    """Merge results from bottle salinity analyses.  If user supplies
    root_dir, the assumption is that all directories follow the
    standard pattern.  Otherwise, directories are needed for bottle
    and salinity files."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        dic_dir = os.path.join(btl_dir, 'dic')

    print('Merging bottle total CO2 and alkalinity...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Load salinity file
    df_dic = pd.read_csv(os.path.join(dic_dir, dic_fname),
        comment='#')
    ds_dic = xr.Dataset.from_dataframe(df_dic)
    ds_dic = ds_dic.rename({'cast': 'cast_number'})

    # Average duplicates
    ds_dic_mean = ds_dic.groupby('cast_number').mean()

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_dic_mean['dic']])
    ds_btl = xr.merge([ds_btl, ds_dic_mean['alkalinity']])

    # Attach metadata
    ds_btl['dic'].attrs = {'long_name': 'dissolved inorganic carbon',
                           'standard_name': 'mole_concentration_of_dissolved_inorganic_carbon_in_sea_water',
                           'units': 'mol m-3',
                           'data_min': np.nanmin(ds_btl['dic'].values),
                           'data_max': np.nanmax(ds_btl['dic'].values),
                           'WHPO_Variable_Name': 'TCARBN'}
    ds_btl['alkalinity'].attrs = {'long_name': 'total alkalinity',
                           'standard_name': 'sea_water_alkalinity_expressed_as_mole_equivalent',
                           'units': 'mol m-3',
                           'data_min': np.nanmin(ds_btl['alkalinity'].values),
                           'data_max': np.nanmax(ds_btl['alkalinity'].values),
                           'WHPO_Variable_Name': 'ALKALI'}


    # Drop bottle number columns
    ds_btl = ds_btl.drop(['dic_btl', 'dic_dup'])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))

    print('done.')


def write_bottle_exchange(btl_fname, root_dir=None, btl_dir=None):
    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')

    print('Exporting to WOCE Hydrographic Exchange...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    df_btl = ds_btl.to_dataframe()
    df_btl.to_csv(os.path.join(btl_dir, (btl_fname + '.csv')))
    print('done.')
    
