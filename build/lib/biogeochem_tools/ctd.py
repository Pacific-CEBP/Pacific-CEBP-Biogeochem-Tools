"""ctd.py"""

import warnings
import numpy as np
import scipy as sci
from scipy import signal
import xarray as xr
import gsw


"""RBR specific tools"""


def rbr_correct_zero_order_hold(ds, channel):
    """The analog-to-digital (A2D) converter on RBR instruments must
    recalibrate periodically.  In the time it takes for the calibration
    to finish, one or more samples are missed.  The onboard firmware
    fills the missed sample with the same data measured during the
    previous sample, a simple technique called a zero-order hold.

    The function identifies zero-hold points by looking for where
    consecutive differences for each channel are equal to zero, and
    replaces them with NaN."""

    ds_out = ds.copy()

    hold_idx = np.where(np.diff(ds[channel]) == 0.0)[0] + 1
    ds_out[channel][hold_idx] = np.NaN
    return ds_out


"""Despiking, filtering, binning type routines"""


def despike(ds, channel, std_limit=3, kernel_size=3):
    """Clean spikes in raw sensor data using median filter."""

    # calculate smoothed and residual signals
    raw = ds[channel].values
    smooth = signal.medfilt(raw, kernel_size=kernel_size)
    resid = raw - smooth

    # obtain indices where residual deviates by more than
    # <std_limit> standard deviations.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        ix = np.where(np.abs(resid / np.nanstd(resid)) > std_limit)

    # replace spikes with NaN values
    ds[channel][ix] = np.NaN

    return ds


def lp_filter(ds, channel, tau):
    # tau is characteristic response time of sensor
    ds_out = ds

    # Nyquist frequency = 0.5 * sample frequency
    fs = 8.0  # Hz
    nyq = 0.5 * 8
    low = (1.0 / tau) / nyq

    # Create an order 5 lowpass butterworth filter
    b, a = butter(5, low, btype="lowpass")

    ds_out[channel] = filtfilt(b, a, ds[channel].values)

    return ds_out


def bin(ds, bins):
    """Bin all variables by pressure."""

    # calculate bin centers
    bin_centers = np.array(
        [(bins[n] + bins[n + 1]) / 2 for n in range(np.size(bins) - 1)]
    )

    # bin the data
    ds = ds.groupby_bins(
        "P",
        bins,
        labels=bin_centers,
        precision=4,
        include_lowest=True,
        restore_coord_dims=True,
    ).mean(dim=xr.ALL_DIMS, skipna=True, keep_attrs=True)

    return ds
