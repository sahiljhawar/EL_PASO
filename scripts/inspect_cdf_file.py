import cdflib
from tabulate import tabulate

def inspect_cdf_file(file_path:str):

    cdf_file = cdflib.CDF(file_path)

    variable_names = cdf_file.cdf_info().zVariables

    var_attrs_to_print = []

    for var in variable_names:
        var_attrs_full = cdf_file.varattsget(var)
        vdr_info       = cdf_file.varinq(var)
        var_data       = cdf_file.varget(var)

        var_shape = var_data.shape

        units = var_attrs_full.get("UNITS", "")

        desc = var_attrs_full.get("CATDESC", "")

        fillvall = var_attrs_full.get("FILLVAL", "")

        data_type = vdr_info.Data_Type_Description

        var_attrs_to_print.append([var, data_type, units, var_shape, fillvall, desc])

    print(tabulate(var_attrs_to_print, headers=["Variable name", "Data Type", "Units", "Data Shape", "Fill value", "Description"]))

if __name__ == "__main__":
    file_name = "X"
    inspect_cdf_file(file_name)
