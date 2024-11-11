from astropy import units as u
from astropy.constants import R_earth
import cdflib
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# Time units
epoch_tt2000 = u.def_unit('epoch_tt2000')
epoch_timestamp = u.def_unit('epoch_timestamp')
epoch_datenum = u.def_unit('epoch_datenum')

# custom conversions
epoch_tt2000_timestamp = [(
    epoch_tt2000,
    epoch_timestamp,
    lambda x: cdflib.cdfepoch.unixtime(x.astype(np.int64)),
    lambda x: cdflib.cdfepoch.timestamp_to_tt2000(x)
)]

def timestamp_to_datenum(timestamp_array):
    # MATLAB's datenum is the number of days since 0000-01-01, plus 1
    # Python's datetime's toordinal() gives the number of days since 0001-01-01

    dt_array = [datetime.fromtimestamp(timestamp, tz=timezone.utc) for timestamp in timestamp_array]

    matlab_datenum_offset = 366  # Difference between MATLAB and Python's reference dates
    return np.array([dt.toordinal() + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400 + matlab_datenum_offset
                        for dt in dt_array])

epoch_timestamp_datenum = [(
    epoch_timestamp,
    epoch_datenum,
    lambda x: timestamp_to_datenum(x),
    lambda x: cdflib.cdfepoch.timestamp_to_tt2000(x)
)]

# Position units
RE = u.def_unit('RE', R_earth)

# we are adding all custom units to the module, so we can access them like built-in units
# e.g., u.RE
setattr(u, 'RE', RE)
setattr(u, 'epoch_tt2000', epoch_tt2000)
setattr(u, 'epoch_timestamp', epoch_timestamp)
setattr(u, 'epoch_datenum', epoch_datenum)

# Adding conversion from degree to radians
u.add_enabled_equivalencies(u.dimensionless_angles())
u.add_enabled_equivalencies(epoch_tt2000_timestamp)
u.add_enabled_equivalencies(epoch_timestamp_datenum)
u.add_enabled_units(RE)

