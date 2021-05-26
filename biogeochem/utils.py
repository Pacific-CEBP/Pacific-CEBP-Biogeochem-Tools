"""Utility routines for executing the full data processing
sequence.  All routines assume a rigid naming convention and
directory structure."""

import os

from . import io as bgc_io
from . import calc as bgc_calc
from . import clean as bgc_clean
from . import plot as bgc_plot


#-------------------------------------------------------------------------------
# Full processing routines
#-------------------------------------------------------------------------------

def process_ctd_data(expocode, eventlog_fname, root_dir, rsk_flist=None,
    aml_flist=None):
    """Follow the standard CTD processing procedure.  This assumes
    standard file and directory structure and no cruise-specific
    data treatment."""

    # load the event log
    df_event_log = bgc_io.load_event_log(os.path.join(root_dir, eventlog_fname))

    # process ctd data
    ds_raw = bgc_io.import_merge_rbr(rsk_flist, expocode, root_dir=root_dir)
    _, cast_flist = bgc_calc.extract_casts(ds_raw, df_event_log,
        root_dir=root_dir)
    bgc_calc.filter_casts(cast_flist, root_dir=root_dir)
    bgc_calc.derive_insitu_properties(cast_flist, root_dir=root_dir)
    bgc_plot.plot_casts(cast_flist, root_dir=root_dir)
    bgc_clean.clean_cast_files(cast_flist, root_dir=root_dir)
    bgc_clean.iso19115(cast_flist, root_dir=root_dir)


def process_niskin_data(expocode, eventlog_fname, root_dir, niskin_length,
    salinity_fname=None, nutrients_fname=None, dic_fname=None, alk_fname=None,
    del18o_fname=None, doc_fname=None):
    """Follow the standard Niskin processing procedure.  This assumes
    standard file and directory structure and no cruise-specific
    data treatment."""

    # load the event log
    df_event_log = bgc_io.load_event_log(os.path.join(root_dir, eventlog_fname))

    # create bottle file
    btl_fname = bgc_io.create_bottle_file(df_event_log, expocode,
        root_dir=root_dir)

    # process niskin data
    raw_fname = '{0:s}_raw.nc'.format(expocode)
    bgc_calc.extract_niskin_salts(btl_fname, raw_fname, niskin_length,
        root_dir=root_dir)
    if salinity_fname is not None:
        bgc_io.merge_bottle_salts(btl_fname, salinity_fname, root_dir=root_dir)
        bgc_plot.plot_ctdsal_qc(btl_fname, root_dir=root_dir)
    if nutrients_fname is not None:
        bgc_io.merge_nutrients(btl_fname, nutrients_fname, root_dir=root_dir)
    if dic_fname is not None:
        bgc_io.merge_dic(btl_fname, dic_fname, root_dir=root_dir)
    if alk_fname is not None:
        pass
    if del18o_fname is not None:
        pass
    if doc_fname is not None:
        pass

    # write to csv
    bgc_io.write_bottle_exchange(btl_fname, root_dir=root_dir)


def refit_alkalinity_titrations():
    pass
    return None