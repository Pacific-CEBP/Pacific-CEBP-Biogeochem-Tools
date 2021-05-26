"""Routines for processing CTD data collected on small boat operations.
All routines assume a rigid naming convention and directory structure."""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import pyrsktools as rsk


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


def read_rsk(fname):
    """Uses pyrsktools to read the RBR .rsk files.  Keeps only
    the conductivity, pressure, temperature, and voltage channels.
    Function returns an xarray Dataset that adheres to CCHDO/WOCE and
    CF data naming and metadata."""

    # Read from RBR "ruskin" file
    with rsk.open(fname) as f:
    
        df = pd.DataFrame(f.npsamples())
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        
        df = df.set_index('timestamp')
        
        ds = xr.Dataset.from_dataframe(df)
        ds = ds.drop_vars(['conductivitycelltemperature_00', 
                           'pressuretemperature_00',
                           'depth_00',
                           'salinity_00',
                           'seapressure_00',
                           'specificconductivity_00',
                           'speedofsound_00'],
                          errors='ignore')
        ds = ds.rename({'pressure_00': 'P',
                        'conductivity_00': 'C',
                        'temperature_00': 'T', 
                        'voltage_00': 'V0', 
                        'voltage_01': 'V1'})
        
        # as per ISO19115, create an instrument variable
        ds['instrument1'] = 'instrument1'
        ds['instrument1'].attrs = {'serial_number': f.instrument.serial,
                                   'calibration_date': '',
                                   'accuracy': '',
                                   'precision': '',
                                   'comment': '',
                                   'long_name': 'RBR {} CTD'.format(f.instrument.model),
                                   'ncei_name': 'CTD',
                                   'make_model': f.instrument.model}
        
        # attach sensor meta-data
        ds['P'].attrs = {'long_name': 'absolute pressure',
                         'standard_name': 'sea_water_pressure',
                         'positive' : 'down',
                         'units': 'dbar',
                         'instrument': 'instrument1',
                         'WHPO_Variable_Name': 'CTDPRS'}
        ds['T'].attrs = {'long_name': 'temperature',
                         'standard_name': 'sea_water_temperature',
                         'units': 'C (ITS-90)',
                         'instrument': 'instrument1',
                         'ncei_name': 'WATER TEMPERATURE',
                         'WHPO_Variable_Name': 'CTDTMP'}
        ds['C'].attrs = {'long_name': 'conductivity',
                         'standard_name': 'sea_water_electrical_conductivity',
                         'units': 'mS/cm',
                         'instrument': 'instrument1'}
        ds['V0'].attrs = {'long_name': 'channel 0 voltage',
                          'standard_name': 'sensor_voltage_channel_0',
                          'units': 'Volts',
                          'instrument': 'instrument1'}
        ds['V1'].attrs = {'long_name': 'channel 1 voltage',
                          'standard_name': 'sensor_voltage_channel_1',
                          'units': 'Volts',
                          'instrument': 'instrument1'}
   
    return ds
    
    
def multi_read_rsk(flist):
    return xr.concat([read_rsk(fname) for fname in flist], dim='timestamp')
    
    
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
    ds_raw = multi_read_rsk([os.path.join(rsk_dir, rsk_fname)
                                 for rsk_fname in rsk_flist])
    val, idx = np.unique(ds_raw.timestamp, return_index=True)
    ds_raw = ds_raw.isel(timestamp=idx) # trim the rare duplicate index values
    ds_raw.attrs['expocode'] = expocode
    print('done.', flush=True)

    print('Correcting zero-order holds in raw traces...', end='', flush=True)
    ds_raw = correct_zero_order_hold(ds_raw, 'P')
    ds_raw = correct_zero_order_hold(ds_raw, 'T')
    ds_raw = correct_zero_order_hold(ds_raw, 'C')
    ds_raw = correct_zero_order_hold(ds_raw, 'V0')
    ds_raw = correct_zero_order_hold(ds_raw, 'V1')
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

    # for files with blank quality flags - replace blank with 2 
    df_salts['salinity_flag_ios'] = df_salts['salinity_flag_ios'].fillna(2)
    
    # assign ranks to quality flags using dictionary    
    flag_rank_dict = {1:8, 2:1, 3:4, 4:5, 5:7, 7:2, 8:3, 9:9, 6:6}
    
    # iterate through flags in the salinity file and assign flag ranks to list
    flag_rank =[]
    for num in df_salts['salinity_flag_ios']:
        flag_rank.append(flag_rank_dict.get(num))
    
    # append flag ranks to dataframe
    df_salts['flag_rank'] = flag_rank
    
    #groupby cast, take mean of duplicates with same flag or take sample with best quality flag according to rank    
    castavg = []                   
    s_avg = []
    s_avg_f = []

    for cast, group in df_salts.groupby('cast'):
        fmin = group['flag_rank'].min()
        idx_fmin = (group['flag_rank']==fmin)
        group_fmin = group['salinity'][idx_fmin]
        s_avg.append(group['salinity'][idx_fmin].mean())
        if len(group['salinity'][idx_fmin])>1 and fmin==1:
            s_avg_f.append(6)
        else:
            s_avg_f.append(fmin)
        castavg.append(cast)
   
    # create dataframe with mean salinities, best of duplicates and quality flags (which are currently rank numbers)   
    df_salts_mean = pd.DataFrame({'cast_number': pd.Series(castavg),
                                'salinity': pd.Series(s_avg),
                                'salinity_flag_ios': pd.Series(s_avg_f)})
   
    #translate back from rank numbers to quality flags using reverse dicitionary look up
    flags = []
    for i in df_salts_mean['salinity_flag_ios']:
      flags.append(list(flag_rank_dict.keys())[list(flag_rank_dict.values()).index(i)])
    
    #replace flag ranks with flag numbers  
    df_salts_mean['salinity_flag_ios']=flags  

    #set index to cast number for merge with btl file  
    df_salts_mean = df_salts_mean.set_index('cast_number')

    #convert dataframe to xarray 
    ds_salts_mean = xr.Dataset.from_dataframe(df_salts_mean)    

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_salts_mean])

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

    #for qualify flags that are blank - replace blank with 2
    df_nuts['nitrate_flag_ios'] = df_nuts['nitrate_flag_ios'].fillna(2)
    df_nuts['silicate_flag_ios'] = df_nuts['silicate_flag_ios'].fillna(2)
    df_nuts['phosphate_flag_ios'] = df_nuts['phosphate_flag_ios'].fillna(2)

    #assign ranks to quality flag using dictionary
    flag_rank_dict = {1:8,2:1,3:4,4:5,5:7,7:2, 8:3, 9:9, 6:6}

    #iterate through nitrate flags and assign rank
    nitrate_flag_rank =[]
    for num in df_nuts['nitrate_flag_ios']:
        nitrate_flag_rank.append(flag_rank_dict.get(num))
    df_nuts['nitrate_flag_rank'] = nitrate_flag_rank

    #iterate through silicate flags and assign rank
    silicate_flag_rank =[]
    for num in df_nuts['silicate_flag_ios']:
        silicate_flag_rank.append(flag_rank_dict.get(num))
    df_nuts['silicate_flag_rank'] = silicate_flag_rank
    
    #iterate through phosphate flags and assign rank
    phosphate_flag_rank =[]
    for num in df_nuts['phosphate_flag_ios']:
        phosphate_flag_rank.append(flag_rank_dict.get(num))
    df_nuts['phosphate_flag_rank'] = phosphate_flag_rank
    
    # Nitrate - groupby cast, take mean of duplicates with same flag or take sample with best quality flag according to rank  
    castavg = []                   
    xavg = []
    xavg_f = []

    for cast, group in df_nuts.groupby('cast'):
        fmin = group['nitrate_flag_rank'].min()
        idx_fmin = (group['nitrate_flag_rank']==fmin)
        group_fmin = group['nitrate'][idx_fmin]
        xavg.append(group['nitrate'][idx_fmin].mean())
        if len(group['nitrate'][idx_fmin])>1 and fmin==1:
             xavg_f.append(6)
        else:
            xavg_f.append(fmin)
        castavg.append(cast)
        nitrate_avg = pd.DataFrame({'cast_number': pd.Series(castavg),
                            'nitrate': pd.Series(xavg),
                            'nitrate_flag_ios': pd.Series(xavg_f)})
        
    nitrate_flags = []
    for i in nitrate_avg['nitrate_flag_ios']:
        nitrate_flags.append(list(flag_rank_dict.keys())[list(flag_rank_dict.values()).index(i)])

    nitrate_avg['nitrate_flag_ios']=nitrate_flags

    # silicate - groupby cast, take mean of duplicates with same flag or take sample with best quality flag according to rank  
    castavg = []                   
    xavg = []
    xavg_f = []

    for cast, group in df_nuts.groupby('cast'):
        fmin = group['silicate_flag_rank'].min()
        idx_fmin = (group['silicate_flag_rank']==fmin)
        group_fmin = group['silicate'][idx_fmin]
        xavg.append(group['silicate'][idx_fmin].mean())
        if len(group['silicate'][idx_fmin])>1 and fmin==1:
             xavg_f.append(6)
        else:
            xavg_f.append(fmin)
        castavg.append(cast)
        silicate_avg = pd.DataFrame({'cast_number': pd.Series(castavg),
                            'silicate': pd.Series(xavg),
                            'silicate_flag_ios': pd.Series(xavg_f)})    
    silicate_flags = []
    for i in silicate_avg['silicate_flag_ios']:
        silicate_flags.append(list(flag_rank_dict.keys())[list(flag_rank_dict.values()).index(i)])

    silicate_avg['silicate_flag_ios']=silicate_flags

    #phosphate - # groupby cast, take mean of duplicates with same flag or take sample with best quality flag according to rank  
    castavg = []                   
    xavg = []
    xavg_f = []

    for cast, group in df_nuts.groupby('cast'):
        fmin = group['phosphate_flag_rank'].min()
        idx_fmin = (group['phosphate_flag_rank']==fmin)
        group_fmin = group['phosphate'][idx_fmin]
        xavg.append(group['phosphate'][idx_fmin].mean())
        if len(group['phosphate'][idx_fmin])>1 and fmin==1:
             xavg_f.append(6)
        else:
            xavg_f.append(fmin)
        castavg.append(cast)
        phosphate_avg = pd.DataFrame({'cast_number': pd.Series(castavg),
                            'phosphate': pd.Series(xavg),
                            'phosphate_flag_ios': pd.Series(xavg_f)})
        
    phosphate_flags = []
    for i in phosphate_avg['phosphate_flag_ios']:
        phosphate_flags.append(list(flag_rank_dict.keys())[list(flag_rank_dict.values()).index(i)])

    phosphate_avg['phosphate_flag_ios']=phosphate_flags

    #set index to cast number for dataframes with averages
    phosphate_avg =phosphate_avg.set_index('cast_number')
    nitrate_avg =nitrate_avg.set_index('cast_number')
    silicate_avg =silicate_avg.set_index('cast_number')

    #stitch means and qualitfy flags together for each nutrient type
    df_nuts_mean = pd.concat([phosphate_avg, nitrate_avg, silicate_avg], axis =1)
   
    #convert dataframe to xarray dataset
    ds_nuts_mean = xr.Dataset.from_dataframe(df_nuts_mean) 

    #merge bottle file and nutrient data   
    ds_btl = xr.merge([ds_btl, ds_nuts_mean])

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
    