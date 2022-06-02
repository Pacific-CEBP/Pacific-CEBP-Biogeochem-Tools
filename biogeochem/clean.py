"""Routines to clean data files of intermediate processing
variables and for generation of metadata standard compliant
files.  All routines assume a rigid naming convention and
directory structure."""

import os
import xarray as xr


def convert_ios_quality_flags(ios_flags, type='bottle'):
    """Convert quality flags assigned by water properties group at IOS to
    equivalent quality flags using the WOCE convention."""
    
    ios_to_woce_btl = { 
                        1:1, # sample drawn from bottle but not analyzed
                        2:2, # acceptable measurement
                       '':2, # acceptable measurement
                        3:3, # questionable measurement
                        4:4, # bad measurement
                        5:5, # not reported
                        6:6, # mean of replicate measurements
                        7:7, # manual chromatographic 
                        8:8, # irregular digital chromatog. peak integration
                        9:9, # sample not drawn for this measurement
                       36:3  # questionable measurement
                      }
    
    woce_flags = ios_flags
    for i, flag in enumerate(ios_flags):
        woce_flags[i] = ios_to_woce[flag]
    
    return woce_flags


def clean_cast_files(cast_flist, root_dir=None, cast_dir=None):
    """Remove intermediate data processing results from cast files.
    If user supplies root_dir, the assumption is that all directories
    follow the standard pattern. Otherwise, directories are needed for
    cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Finalizing cast files...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        # remove full resolution, time domain data
        ds_cast = ds_cast.drop_vars(['timestamp', 'P',
                                     'P_raw', 'T_raw', 'C_raw',
                                     'P_despike', 'T_despike', 'C_despike',
                                     'T_bins', 'C_bins'])

        # remove raw data that have been used to calculate derived props
        ds_cast = ds_cast.drop_vars(['C', 'V0', 'V1'])

        # rename binned pressure to pressure
        ds_cast = ds_cast.rename({'P_bins': 'P'})

        # swap dimensions from P to z

        # expand coordinates
        #ds_cast = ds_cast.expand_dims(['lat', 'lon', 'time'])
        #ds_cast = ds_cast.assign_coords({'lat': ds_cast['lat'],
        #                                 'lon': ds_cast['lon'],
        #                                 'time': ds_cast['time']})

        # save dataset
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))

    print('done.')

    
def iso19115(cast_flist, root_dir=None, cast_dir=None):
    """Generate ISO19115 compliant files.  If user supplies root_dir,
    the assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for cast_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')

    print('Creating ISO19115 compliant files...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        # set all kinds of attributes
        ds_cast = ds_cast.assign_attrs({
            'title': 'Physical water properties of the Port of Prince Rupert and Skeena River Estuary',
            'naming_authority': 'ca.gc.dfo-mpo',
            'program': 'DFO OPP-CEBP',

            'creator_name': 'Paul Covert',
            'creator_institution': 'Fisheries and Oceans Canada; Institute of Ocean Sciences',
            'creator_email': 'Paul.Covert@dfo-mpo.gc.ca',

            'sea_name': 'Coastal Waters of British Columbia',
            'feature_type': 'profile',
            'cdm_data_type': 'Station',
            'instrument': 'In Situ/Laboratory Instruments > Profilers/Sounders > > > CTD',
            'platform': 'In Situ Ocean-base Platforms > SHIPS'})

        # rename a few params

        # save dataset
        ds_cast.to_netcdf(os.path.join(cast_dir, cast_fname))

    print('done.')
