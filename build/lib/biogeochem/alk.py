"""Utility routines for recalculating alkalinity from raw titration
data."""

import os


def _read_header_line(f):
    return f.readline().split('\t')[0]
    
    
def read_titration_header(filepath_or_buffer):
    with open(filepath_or_buffer) as f:
        station = _read_header_line(f)
        castno = int(_read_header_line(f))
        lat = float(_read_header_line(f))
        lon = float(_read_header_line(f))
        nisk = _read_header_line(f)
        depth = float(_read_header_line(f))
        salinity = float(_read_header_line(f))
        dic_btl = _read_header_line(f)
        analyte_units = _read_header_line(f)
        analyte_amount = float(_read_header_line(f))
    
    return (station, castno, lat, lon, nisk, depth, salinity, dic_btl,
            analyte_units, analyte_amount)
                

def add_crm_info():


def create_calkulate_meta(btl_fname, dic_fname, root_dir=None, btl_dir=None, 
    dic_dir=None, titr_dir=None, crm_id='crm'):

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        dic_dir = os.path.join(btl_dir, 'dic')
        titr_dir = os.path.join(dic_dir, 'titrations')
    
    file_name = []
    file_path = []
    analysis_date = []
    analysis_time = []
    dic_btl = []
    analyte_mass = []
    analyte_volume = []
    
    for fname in flist:
    
        # collect info from titration file names
        file_name.append(os.path.basename(fname))
        file_path.append(os.path.normpath(fname))
        analysis_date.append(
            '20' + 
            file_name[-1].split('-')[3].split('_')[1][0:2] + 
            '-' + 
            file_name[-1].split('-')[3].split('_')[1][2:4] + 
            '-' + 
            file_name[-1].split('-')[3].split('_')[1][4:6]
        )
        analysis_time.append(
            file_name[-1].split('-')[3].split('_')[2][0:2] + 
            ':' + 
            file_name[-1].split('-')[3].split('_')[2][2:4]
        )
        
        # collect necessary info from titration file headers
        meta = read_titration_header(fname)
        if meta['dic_btl']==crm_identifier:
            dic_btl.append('crm')
        else:
            dic_btl.append(meta['dic_btl'])
        if meta['analyte_units']=='kg':
            analyte_mass.append(meta['analyte_amount'])
            analyte_volume.append(np.nan)
        elif meta['analyte_units']=='g':
            analyte_mass.append(meta['analyte_amount'] / 1000.)
            analyte_volume.append(np.nan)
        elif analyte_units=='ml':
            analyte_mass.append(np.nan)
            analyte_volume.append(meta['analyte_amount'])
        else:
            analyte_mass.append(np.nan)
            analyte_volume.append(np.nan)
    
    # create pandas dataframe 
    df_alk = pd.DataFrame({
        'dic_btl': dic_btl,
        'file_name': file_name,
        'file_path': file_path,
        'analysis_date': analysis_date,
        'analysis_time': analysis_time,
        'analyte_mass': analyte_mass,
        'analyte_volume': analyte_volume
    })
    df_alk = df_alk.set_index('dic_btl')
    
    # load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))
    df_btl = ds_btl.to_dataframe()
    
    # merge bottle salts and sensor salinity, with bottle salts taking
    # precedence.
    df_btl['salinity'] = df_btl['salinity'].combine_first(df_btl['ctdsal'])

    # trim unnecessary columns ** note: change to collect necessary!
    df_btl = df_btl.drop(columns=['time', 'station_id', 'station_name', 'lat', 
                                  'lon', 'sampling_platform', 'cast_f', 
                                  'sample_number', 'sample_number_flag_w', 
                                  'niskin_height', 'wire_angle', 'sampler', 
                                  'expocode', 'ctdprs', 'ctdsal', 'ctdtmp',
                                  'nitrate'])
    df_btl = df_btl.rename(columns={'phosphate': 'total_phosphate',
                                'silicate': 'total_silicate'})             
                                
    # reshape to dic_btl or dic_dup as index
    df_btl_dic = df_btl.drop(columns=['dic_dup'])
    df_btl_dic = df_btl_dic.set_index('dic_btl')
    df_btl_dup = df_btl.drop(columns=['dic_btl'])
    df_btl_dup['dic_dup'] = df_btl_dup['dic_dup'].replace('', np.nan)
    df_btl_dup = df_btl_dup.dropna(subset=['dic_dup'])
    df_btl_dup = df_btl_dup.rename(columns={'dic_dup': 'dic_btl'})
    df_btl_dup = df_btl_dup.set_index('dic_btl')
    df_btl = pd.concat([df_btl_dic, df_btl_dup])

    # now merge
    df = pd.merge(left=df, right=df_btl, how='left', left_on='dic_btl', 
                  right_on='dic_btl')
                  
    return df_alk
    