"""Plotting routines used for visual quality control of
processed CTD and bottle data. All routines assume a rigid
naming convention and directory structure."""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_casts(cast_flist, root_dir=None, cast_dir=None, plot_dir=None):
    """Plot temperature, conductivity, and salinity profiles of the
    individual casts. For each parameter, the raw, processed, and
    intermediate processing steps are plot.  This aids in visual
    inspection during quality control.  If user supplies root_dir, the
    assumption is that all directories follow the standard pattern.
    Otherwise, directories are needed for cast_dir and plot_dir."""

    if root_dir is not None:
        cast_dir = os.path.join(root_dir, 'ctd', 'cast')
        plot_dir = os.path.join(root_dir, 'ctd', 'plot')
    if not(os.path.isdir(plot_dir)):
        os.mkdir(plot_dir)

    print('Plotting cast profiles...', end='', flush=True)
    for cast_fname in cast_flist:

        # load dataset
        ds_cast = xr.load_dataset(os.path.join(cast_dir, cast_fname))

        fig, (axT, axC, axS) = plt.subplots(nrows=1, ncols=3, figsize=(6,8),
            sharey=True, constrained_layout=True)

        try:
            axT.plot(ds_cast['T_raw'], ds_cast['P_raw'],
                     '-', color=(0.5,0.5,0.5), linewidth=3, label='raw')
            axC.plot(ds_cast['C_raw'], ds_cast['P_raw'],
                     '-', color=(0.5,0.5,0.5), linewidth=3, label='raw')
        except:
            pass

        try:
            axT.plot(ds_cast['T_despike'], ds_cast['P_despike'],
                     'b-', linewidth=2, label='despike')
            axC.plot(ds_cast['C_despike'], ds_cast['P_despike'],
                     'b-', linewidth=2, label='despike')
        except:
            pass

        try:
            axT.plot(ds_cast['T'], ds_cast['P'], 'k-', label='binned')
            axC.plot(ds_cast['C'], ds_cast['P'], 'k-', label='binned')
            axS.plot(ds_cast['SP'], ds_cast['P'], 'k-', label='binned')
        except:
            pass

        Pmax = np.ceil((ds_cast['P'].max() / 5.)) * 5.
        axT.set_ylim([0., Pmax.values])
        axT.invert_yaxis()
        axC.invert_yaxis()
        axS.invert_yaxis()

        axT.set_xlabel(r'$T$ / $^\circ$C [ITS-90]', fontsize=9)
        axT.set_ylabel('$P$ / dbar', fontsize=9)
        axC.set_xlabel(r'$C$ / $\mu$S$\,$cm$^{-1}$', fontsize=9)
        axS.set_xlabel(r'$S_\mathrm{A}$ / g$\,$kg$^{-1}$', fontsize=9)

        axT.set_title('Cast: {0:d}\nStn: {1:s} ({2:s})'.format(
            ds_cast.cast_number, ds_cast.station_name, ds_cast.station_id),
            loc='left', fontsize=9)
        date_str = ds_cast['time'].dt.strftime('%Y-%m-%d').values
        time_str = ds_cast['time'].dt.strftime('%H:%M').values
        axS.set_title('UTC {0:s}\n{1:s}'.format(date_str, time_str),
            loc='right', fontsize=9)

        axT.legend()
        axC.legend()
        axS.legend()

        plot_fname = '{0:s}_{1:03d}.pdf'.format(ds_cast.expocode,
            ds_cast.cast_number)
        fig.savefig(os.path.join(plot_dir, plot_fname))
        plt.close(fig)

    print('done.')


def plot_ctdsal_qc(btl_fname, root_dir=None, btl_dir=None, plot_dir=None):
    """Generate a plot to aid in quality control of the ctdsal
    calculations."""

    if root_dir is not None:
        btl_dir = os.path.join(root_dir, 'btl')
        plot_dir = os.path.join(root_dir, 'btl', 'plots', 'ctdsal')

    # Load bottle file
    ds_btl = xr.load_dataset(os.path.join(btl_dir, btl_fname))

    # Create WOCE quality flag for ctdsal, ctdtmp, and ctdprs based
    # on difference between ctdsal and bottle salinity.  If bottle
    # salinity does not exist, the assume data quality is good.
    ds_btl['ctdsal_flag'] = xr.full_like(ds_btl['ctdsal'], 2)
    ds_btl['ctdsal_flag'] = ds_btl['ctdsal_flag'].where(
        np.abs(ds_btl['salinity'] - ds_btl['ctdsal'])<0.5, 3)

    # create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0., 35.], [0., 35.], 'k-', linewidth=0.5)
    ds_btl_good = ds_btl.where(ds_btl['ctdsal_flag']==2, drop=True)
    ds_btl_ok = ds_btl.where(ds_btl['ctdsal_flag']==3, drop=True)
    ax.plot(ds_btl_good['salinity'], ds_btl_good['ctdsal'], 'o', markersize=4)
    ax.plot(ds_btl_ok['salinity'], ds_btl_ok['ctdsal'], 'ro', markersize=4)
    for i, lbl in enumerate(ds_btl_ok['cast_number'].values):
        ax.annotate(lbl, (ds_btl_ok['salinity'][i], ds_btl_ok['ctdsal'][i]),
            size=10, weight='bold', color='r')
    ax.set_xlim([0., 35.])
    ax.set_ylim([0., 35.])
    ax.set_xlabel('Analytical salinity (PSS-78)')
    ax.set_ylabel('Calculated salinity (PSS-78)')

    # save figure
    plot_fname = '{0:s}_ctdsal_qc.pdf'.format(ds_btl['expocode'].values[0])
    fig.savefig(os.path.join(plot_dir, plot_fname))
    plt.close(fig)
