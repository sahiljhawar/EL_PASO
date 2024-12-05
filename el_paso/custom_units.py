from astropy import units as u
from astropy.constants import R_earth
import cdflib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from IRBEM import Coords

# Time units
tt2000 = u.def_unit('tt2000')
posixtime = u.def_unit('posixtime')
datenum = u.def_unit('datenum')

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

posixtime_datenum = [(
    posixtime,
    datenum,
    lambda x: posixtime_to_datenum(x),
    lambda x: cdflib.cdfepoch.posixtime_to_tt2000(x)
)]

# Position units
RE = u.def_unit('RE', R_earth)

# we are adding all custom units to the module, so we can access them like built-in units
# e.g., u.RE
setattr(u, 'RE', RE)
setattr(u, 'tt2000', tt2000)
setattr(u, 'posixtime', posixtime)
setattr(u, 'datenum', datenum)

# Adding conversion from degree to radians
u.add_enabled_equivalencies(u.dimensionless_angles())
u.add_enabled_equivalencies(epoch_tt2000_posixtime)
u.add_enabled_equivalencies(posixtime_datenum)
u.add_enabled_units(RE)
u.add_enabled_units(posixtime)
u.add_enabled_units(datenum)
u.add_enabled_units(tt2000)
