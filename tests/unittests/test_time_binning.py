from datetime import datetime, timedelta, timezone
from copy import deepcopy

from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from el_paso.classes import Variable, TimeVariable, TimeBinMethod
from el_paso.standardization import time_bin_all_variables

def test_time_binning():

    start_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2000, 1, 1, 1, tzinfo=timezone.utc)

    time_var = TimeVariable(original_unit=u.posixtime)

    variables = {
        'Time': time_var,
        'Data': Variable(original_unit='', time_bin_method=TimeBinMethod.NanMean, time_variable=time_var)
    }

    variables['Time'].data = np.arange(start_time.timestamp(), end_time.timestamp(), timedelta(minutes=1).total_seconds())
    variables['Data'].data = np.empty_like(variables['Time'].data )

    for i in range(len(variables['Time'].data)):
        variables['Data'].data[i] = len(variables['Time'].data) - i // 5

    default_time = deepcopy(variables['Time'])
    default_data = deepcopy(variables['Data'])

    plt.figure(figsize=(10,10))
    plt.plot(variables['Time'].data, variables['Data'].data, 'k*')

    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time)

    plt.plot(variables['Time'].data, variables['Data'].data, 'rx')

    variables['Time'] = default_time
    variables['Data'] = default_data
    variables['Data'].time_variable = variables['Time']

    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement='left')

    plt.plot(variables['Time'].data, variables['Data'].data, 'gx')

    plt.legend(['Before', 'After (center)', 'After (left)'])
    plt.savefig('time_binning_test.png')

if __name__ == '__main__':
    test_time_binning()