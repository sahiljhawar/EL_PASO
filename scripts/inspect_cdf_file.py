import cdflib
from tabulate import tabulate
import numpy as np
from scipy.integrate import simpson
from icecream import ic

#file_path = '../erg_xep_l2_omniflux_20180803_v00_00.cdf'
file_path = '../erg_hep_l2_omniflux_20180101_v03_01.cdf'
cdf_file = cdflib.CDF(file_path)

variable_names = cdf_file.cdf_info().zVariables

var_attrs_to_print = []

for var in variable_names:
    var_attrs_full = cdf_file.varattsget(var)
    vdr_info       = cdf_file.varinq(var)
    var_data       = cdf_file.varget(var)

    var_shape = var_data.shape

    if 'UNITS' in var_attrs_full.keys():
        units = var_attrs_full['UNITS']
    else:
        units = ''

    if 'CATDESC' in var_attrs_full.keys():
        desc = var_attrs_full['CATDESC']
    else:
        desc = ''

    if 'FILLVAL' in var_attrs_full.keys():
        fillvall = var_attrs_full['FILLVAL']
    else:
        fillvall = ''

    data_type = vdr_info.Data_Type_Description

    var_attrs_to_print.append([var, data_type, units, var_shape, fillvall, desc])

print(tabulate(var_attrs_to_print, headers=['Variable name', 'Data Type', 'Units', 'Data Shape', 'Fill value', 'Description']))

file_path = '../erg_hep_l2_omniflux_20180101_v03_01.cdf'
cdf_file = cdflib.CDF(file_path)

flux_omni = cdf_file.varget('FEDO_H')[2,0]

file_path = '../erg_hep_l3_pa_20180101_v01_01.cdf'
cdf_file = cdflib.CDF(file_path)

flux_diff = cdf_file.varget('FEDU_H')[2,0,:]
pa = np.deg2rad(cdf_file.varget('FEDU_H_Alpha'))

ic(flux_omni)
#ic(flux_diff)
ic(np.sum(flux_diff) / (2*np.pi))
ic(simpson(x=pa, y=flux_diff*np.sin(pa)))