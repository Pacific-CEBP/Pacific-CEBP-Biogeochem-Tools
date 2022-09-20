"""Routines for processing CTD data collected on small boat operations.
All routines assume a rigid naming convention and directory structure."""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import pyrsktools as rsk

from . import ctd as bgc_ctd


# define a few constants
btl_flag_valid_range = (0,9)
btl_flag_values = '0,1,2,3,4,5,6,7,8,9'
btl_flag_meanings = 'Acceptable,' + \
                    'Sample not analyzed,' + \
                    'Acceptable,' + \
                    'Questionable (probably good),' + \
                    'Poor (probably bad),' + \
                    'Not reported as noted bad during analysis,' + \
                    'Mean of replicates,' + \
                    'Manual chromatographic peak measurement,' + \
                    'Irregular digital chromatographic peak integration,' + \
                    'Not collected'


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
        # ---presently commented out until debugging can
        # be completed---
        """
        ds['instrument1'] = 'instrument1'
        ds['instrument1'].attrs = {'serial_number': f.instrument.serial,
                                   'calibration_date': '',
                                   'accuracy': '',
                                   'precision': '',
                                   'comment': '',
                                   'long_name': 'RBR {} CTD'.format(f.instrument.model),
                                   'ncei_name': 'CTD',
                                   'make_model': f.instrument.model}
        """
        
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
                     rsk_dir=None, csv_export=True):
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
    ds_raw = bgc_ctd.rbr_correct_zero_order_hold(ds_raw, 'P')
    ds_raw = bgc_ctd.rbr_correct_zero_order_hold(ds_raw, 'T')
    ds_raw = bgc_ctd.rbr_correct_zero_order_hold(ds_raw, 'C')
    ds_raw = bgc_ctd.rbr_correct_zero_order_hold(ds_raw, 'V0')
    ds_raw = bgc_ctd.rbr_correct_zero_order_hold(ds_raw, 'V1')
    print('done.', flush=True)

    print('Saving merged CTD data...', end='', flush=True)
    print(ds_raw, flush=True)
    raw_nc_fname = '{0:s}_raw.nc'.format(expocode)
    ds_raw.to_netcdf(os.path.join(raw_dir, raw_nc_fname), 'w')
    if csv_export:
        raw_csv_fname = '{0:s}_raw.csv'.format(expocode)
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

def quality_flags(df, variable, flag_column_name):
    
    """
    (Pandas DataFrame, string, string)
    Takes dataframe,varible column name (as string) and flag cloumn
    name (as string) as written in df e.g. (df_salts, 'salinity', 'salinity_flag_ios')    
    Takes mean of good (flag 2) duplicates and asssigns a flag of 6. 
    Otherwise takes sample with the best quality flag. 
    If there is just a single sample with no dupliate pair, that value is 
    returned as is, with its original quality flag
    Works for cruises where no more than one set of duplicates are taken
    from one cast, otherwise an error is generated.
    """

    #identify quality flag column in the dataframe from the string provided and set blank values to 2
    flag_column= df[flag_column_name]
    flag_column =flag_column.fillna(2)
    #assign ranks to quality flags
    flag_rank_dict = {1:8,2:1,3:4,4:5,5:7,7:2, 8:3, 9:9, 6:6}
    #set up empty lists for code iteration below
    flag_rank =[]
    castavg = []                   
    xavg = []
    xavg_f = []
    flags = []
      
    #assign rank to each quality flag in dataframe
    for num in flag_column:
        flag_rank.append(flag_rank_dict.get(num))
    #append flag ranks as a new column to dataframe
    df['flag_rank'] = flag_rank
    #groupby cast, and iterate through each cast
    for cast, group in df.groupby('cast'):
        #to find smallest and largest flag rank in a cast group
        fmin = group['flag_rank'].min()
        fmax = group['flag_rank'].max()
        idx_fmin = (group['flag_rank']==fmin)
        xavg.append(group[variable][idx_fmin].mean())
        
        #conditionals for flag ranks
        #if two flags in a group and both are 2, append a flag of 6 
        if len(group[variable])==2 and fmin==1 and fmax ==1:
            xavg_f.append(6)
        #if there are more than 2 values in one cast raise an error    
        elif len(group[variable])>2:
            print('error: more than two duplicates in one cast')
        #if there are two values but one is already a duplicate (flag 6) raise an error    
        elif len(group[variable])==2 and fmax ==6:
            print('error: more than two duplicates in one cast')
        #for all other combinations of flags append the lowest     
        else:
            xavg_f.append(fmin)
        #assign cast numbers to list    
        castavg.append(cast)
    
    # create a dataframe with the cast number, means and flags
    df_avg = pd.DataFrame({'cast_number': pd.Series(castavg),
                            variable: pd.Series(xavg),
                            flag_column_name: pd.Series(xavg_f)})
    
    #translate flag ranks back to quality flag values using reverse dictionary look up
    for i in df_avg[flag_column_name]:
        flags.append(list(flag_rank_dict.keys())[list(flag_rank_dict.values()).index(i)])
  
    #append the quality flags to the dataframe and reset index to cast_number 
    df_avg[flag_column_name]=flags
    df_avg=df_avg.set_index('cast_number')
   
    #return the dataframe
    return df_avg

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

    # Average duplicates and assign quality flags
    df_salts_mean = quality_flags(df_salts, 'salinity','salinity_flag_ios')

    # Convert dataframe to xarray 
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
                                'standard_name': 'sea_water_practical_salinity quality_flag',
                                'units': '',
                                'valid_range': btl_flag_valid_range,
                                'flag_values': btl_flag_values,
                                'flag_meanings': btl_flag_meanings,
                                'WHPO_Variable_Name': 'SALNTY_FLAG_W'}                    

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

    # Average duplicates and assign qualify flags
    nitrate_avg = quality_flags(df_nuts, 'nitrate', 'nitrate_flag_ios')
    silicate_avg = quality_flags(df_nuts, 'silicate','silicate_flag_ios')
    phosphate_avg = quality_flags(df_nuts, 'phosphate', 'phosphate_flag_ios')

    #stitch means and qualitfy flags together for each nutrient type
    df_nuts_mean = pd.concat([nitrate_avg, silicate_avg, phosphate_avg], axis =1)
   
    #convert dataframe to xarray dataset
    ds_nuts_mean = xr.Dataset.from_dataframe(df_nuts_mean) 

    #merge bottle file and nutrient data   
    ds_btl = xr.merge([ds_btl, ds_nuts_mean])

    # Attach metadata
    # metdata for nutrients
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
    # metadata for flags                           
    ds_btl['nitrate_flag_ios'].attrs = {'long_name': 'dissolved nitrate + nitrite concentration quality ',
                               'standard_name': 'mole_concentration_of_nitrate_and_nitrite_in_sea_water quality_flag',
                               'units': '',
                               'valid_range': btl_flag_valid_range,
                               'flag_values': btl_flag_values,
                               'flag_meanings': btl_flag_meanings,
                               'WHPO_Variable_Name': 'NO2+NO3_FLAG_W'}     

    ds_btl['silicate_flag_ios'].attrs = {'long_name': 'dissolved silicate concentration quality',
                               'standard_name': 'mole_concentration_of_silicate_in_sea_water quality_flag',
                               'units': '',
                               'valid_range': btl_flag_valid_range,
                               'flag_values': btl_flag_values,
                               'flag_meanings': btl_flag_meanings,
                               'WHPO_Variable_Name': 'SILCAT_FLAG_W'} 

    ds_btl['phosphate_flag_ios'].attrs = {'long_name': 'dissolved phosphate concentration quality',
                               'standard_name': 'mole_concentration_of_phosphate_in_sea_water quality_flag',
                               'units': '',
                               'valid_range': btl_flag_valid_range,
                               'flag_values': btl_flag_values,
                               'flag_meanings': btl_flag_meanings,
                               'WHPO_Variable_Name': 'PHSPHT_FLAG_W'}     

    # Drop bottle numbers
    ds_btl = ds_btl.drop(['nut_btl', 'nut_dup'])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))
    print('done.')


def merge_dic(btl_fname, dic_fname, root_dir=None, btl_dir=None,
    dic_dir=None):
    """Merge results from bottle dic and ta analyses.  If user supplies
    root_dir, the assumption is that all directories follow the
    standard pattern.  Otherwise, directories are needed for bottle
    and dic/ta files."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        dic_dir = os.path.join(btl_dir, 'dic')

    print('Merging bottle total CO2 and alkalinity...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Load dic file
    df_dic = pd.read_csv(os.path.join(dic_dir, dic_fname),
        comment='#')
    ds_dic = xr.Dataset.from_dataframe(df_dic)
    ds_dic = ds_dic.rename({'cast': 'cast_number'})

    # Average duplicates and assign qualify flags
    dic_avg = quality_flags(df_dic, 'dic', 'dic_flag_ios')
    ta_avg = quality_flags(df_dic, 'ta','ta_flag_ios')
    
    #stitch means and qualitfy flags together for dic and ta
    df_dic_mean = pd.concat([dic_avg, ta_avg], axis =1)

    #convert dataframe to xarray dataset
    ds_dic_mean = xr.Dataset.from_dataframe(df_dic_mean) 

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_dic_mean])
    
    # Attach metadata for dic and alkalinity
    ds_btl['dic'].attrs = {'long_name': 'dissolved inorganic carbon',
                           'standard_name': 'mole_concentration_of_dissolved_inorganic_carbon_in_sea_water',
                           'units': 'mol m-3',
                           'data_min': np.nanmin(ds_btl['dic'].values),
                           'data_max': np.nanmax(ds_btl['dic'].values),
                           'WHPO_Variable_Name': 'TCARBN'}
    ds_btl['ta'].attrs = {'long_name': 'total alkalinity',
                           'standard_name': 'sea_water_alkalinity_expressed_as_mole_equivalent',
                           'units': 'mol m-3',
                           'data_min': np.nanmin(ds_btl['ta'].values),
                           'data_max': np.nanmax(ds_btl['ta'].values),
                           'WHPO_Variable_Name': 'ALKALI'}

    # Metadata for quality flags
    ds_btl['dic_flag_ios'].attrs = {'long_name': 'dissolved inorganic carbon quality',
                           'standard_name': 'mole_concentration_of_dissolved_inorganic_carbon_in_sea_water quality_flag',
                           'units': '',
                           'valid_range': btl_flag_valid_range,
                           'flag_values': btl_flag_values,
                           'flag_meanings': btl_flag_meanings,
                           'WHPO_Variable_Name': 'TCARBN_FLAG_W'} 

    ds_btl['ta_flag_ios'].attrs = {'long_name': 'total alkalinity quality',
                           'standard_name': 'sea_water_alkalinity_expressed_as_mole_equivalent quality_flag',
                           'units': '',
                           'valid_range': btl_flag_valid_range,
                           'flag_values': btl_flag_values,
                           'flag_meanings': btl_flag_meanings,
                           'WHPO_Variable_Name': 'ALKALI_FLAG_W'}     

    # Drop bottle number columns
    ds_btl = ds_btl.drop(['dic_btl', 'dic_dup'])

    # Save results
    ds_btl.to_netcdf(os.path.join(btl_dir, btl_fname))

    print('done.')


def merge_del18o(btl_fname, del18o_fname, root_dir=None, btl_dir=None,
    del18o_dir=None):
    """Merge results from oxygen isotope analyses.  If user supplies
    root_dir, the assumption is that all directories follow the
    standard pattern.  Otherwise, directories are needed for bottle
    and del18o files."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        del18o_dir = os.path.join(btl_dir, 'del18o')

    print('Merging oxygen isotopes...', end='', flush=True)

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Load del18o file
    df_del18o = pd.read_csv(os.path.join(del18o_dir, del18o_fname),
        comment='#')
    ds_del18o = xr.Dataset.from_dataframe(df_del18o)
    ds_del18o = ds_del18o.rename({'cast': 'cast_number'})

    #average duplicates and assign quality flags
    df_del18o_mean = quality_flags(df_del18o, 'del18o','del18o_flag_woce')

    #convert dataframe to xarray 
    ds_del18o_mean = xr.Dataset.from_dataframe(df_del18o_mean)    

    # Merge into bottle file
    ds_btl = xr.merge([ds_btl, ds_del18o_mean])

    # Attach metadata
    ds_btl['del18o'].attrs = {'long_name': 'isotope_ratio_of_18O_to_16O',
                                'standard_name': 'isotope_ratio_of_18O_to_16O_in_sea_water_excluding_solutes_and_solids',
                                'units': '/MILLE',
                                'data_min': np.nanmin(ds_btl['del18o'].values),
                                'data_max': np.nanmax(ds_btl['del18o'].values),
                                'WHPO_Variable_Name': 'DELO18'}

    ds_btl['del18o_flag_woce'].attrs = {'long_name': 'isotope_ratio_of_18O_to_16O quality',
                                'standard_name': 'isotope_ratio_of_18O_to_16O_in_sea_water_excluding_solutes_and_solids quality_flag',
                                'units': '',
                                'valid_range': btl_flag_valid_range,
                                'flag_values': btl_flag_values,
                                'flag_meanings': btl_flag_meanings,
                                'WHPO_Variable_Name': 'DELO18_FLAG_W'} 
    # Drop bottle number columns
    ds_btl = ds_btl.drop(['del18o_btl', 'del18o_dup'])

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

   
def export_ctd_to_matlab(cast_flist, root_dir=None, ctd_dir=None):
    """Generate Matlab .mat files.  If user supplies root_dir,
    the assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Creating Matlab (.mat) files...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        # create scipy vectors
        
        # save
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))

    print('done.')
