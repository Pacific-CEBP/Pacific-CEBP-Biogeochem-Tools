from pandas import read_csv

def _load_dickson_crm():
    crm_path = './dickson_crm_db.csv'
    crm_df = read_csv(crm_path, comment='#', parse_dates=['date'],
                         index_col='batch', na_values=-99)
    return crm_df
    

def crm(batch):
    crm_df = _load_dickson_crm()
    print(crm_df)
    return {'batch': batch,
            'date': crm_df['date'][batch],
            'salinity': crm_df['salinity'][batch],
            'TCO2': crm_df['TCO2'][batch],
            'TCO2CL': crm_df['TCO2_CL95'][batch],
            'TAlk': crm_df['TAlk'][batch],
            'TAlkCL': crm_df['TAlk_CL95'][batch],
            'phosphate': crm_df['phosphate'][batch],
            'silicate': crm_df['silicate'][batch],
            'nitrite': crm_df['nitrite'][batch],
            'nitrate': crm_df['nitrate'][batch]}
    