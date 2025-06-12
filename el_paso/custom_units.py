from astropy import units as u
from astropy.constants import R_earth
import cdflib
from datetime import datetime, timezone

import numpy as np


# Time units
tt2000 = u.def_unit("tt2000",)
posixtime = u.def_unit("posixtime")
datenum = u.def_unit("datenum")

# custom conversions
epoch_tt2000_posixtime = [(
    tt2000,
    posixtime,
    lambda x: cdflib.cdfepoch.unixtime(x),
    lambda x: cdflib.cdfepoch.posixtime_to_tt2000(x)
)]

def posixtime_to_datenum(posixtime_array):
    # MATLAB's datenum is the number of days since 0000-01-01, plus 1
    # Python's datetime's toordinal() gives the number of days since 0001-01-01

    dt_array = [datetime.fromtimestamp(posixtime, tz=timezone.utc) for posixtime in posixtime_array]

    matlab_datenum_offset = 366  # Difference between MATLAB and Python's reference dates
    return np.array([dt.toordinal() + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400 + matlab_datenum_offset
                        for dt in dt_array])

def datenum_to_posixtime(datenum_array):
    """
    Converts a MATLAB datenum array to POSIX timestamps.
    """
    posix_timestamps = []
    for dn in np.atleast_1d(datenum_array): # Ensure it works with single values or arrays

        dt_parts = cdflib.cdfepoch.breaktime(dn, epoch=cdflib.cdfepoch.EPOCH_TO_DATENUM)

        # Reconstruct datetime object from parts
        dt_obj = datetime(dt_parts[0], dt_parts[1], dt_parts[2],
                          dt_parts[3], dt_parts[4], dt_parts[5],
                          dt_parts[6] * 1000 + dt_parts[7], # milliseconds to microseconds
                          tzinfo=timezone.utc)

        posix_timestamps.append(dt_obj.timestamp())
    return np.array(posix_timestamps)

posixtime_datenum = [(
    posixtime,
    datenum,
    lambda x: posixtime_to_datenum(x),
    lambda x: datenum_to_posixtime(x),
)]


def tt2000_to_datenum(tt2000_val):
    """
    Converts tt2000 (ns) to MATLAB datenum (days) via posixtime.
    This function will be used directly in the new equivalency.
    """
    # 1. Convert tt2000 to posixtime
    posix_val = cdflib.cdfepoch.unixtime(tt2000_val) # Returns in seconds
    # 2. Convert posixtime to datenum
    datenum_val = posixtime_to_datenum(posix_val) # Returns in days
    return datenum_val

def datenum_to_tt2000(datenum_val):
    """
    Converts MATLAB datenum (days) to tt2000 (ns) via posixtime.
    This function will be used directly in the new equivalency.
    """
    # 1. Convert datenum to posixtime
    posix_val = datenum_to_posixtime(datenum_val) # Returns in seconds
    # 2. Convert posixtime to tt2000
    tt2000_val = cdflib.cdfepoch.posixtime_to_tt2000(posix_val) # Returns in nanoseconds
    return tt2000_val

tt2000_datenum = [(
    tt2000,
    datenum,
    lambda x: tt2000_to_datenum(x),
    lambda x: datenum_to_tt2000(x)
)]

# Position units
RE = u.def_unit("RE", R_earth)

# we are adding all custom units to the module, so we can access them like built-in units
# e.g., u.RE
setattr(u, "RE", RE)
setattr(u, "tt2000", tt2000)
setattr(u, "posixtime", posixtime)
setattr(u, "datenum", datenum)

# Adding conversion from degree to radians
u.add_enabled_equivalencies(u.dimensionless_angles())
u.add_enabled_equivalencies(epoch_tt2000_posixtime)
u.add_enabled_equivalencies(posixtime_datenum)
u.add_enabled_equivalencies(tt2000_datenum)
u.add_enabled_units(RE)
u.add_enabled_units(posixtime)
u.add_enabled_units(datenum)
u.add_enabled_units(tt2000)
