from datetime import timedelta, timezone
from os import name

import dateutil
import numpy as np
from astropy import units as u

from el_paso import IRBEM_SYSAXIS_GDZ, IRBEM_SYSAXIS_GEO
from el_paso.classes import DerivedVariable, SourceFile, TimeBinMethod, TimeVariable, Variable
from el_paso.processing import (
    compute_magnetic_field_variables,
    compute_PSD,
    convert_all_data_to_standard_units,
    fold_pitch_angles_and_flux,
    load_variables_from_source_files,
    time_bin_all_variables,
)
from el_paso.save_standards.data_org import DataorgPMF
from IRBEM import Coords


def process_DSX_orbit(
    save_data_dir, download_data_dir, irbem_lib_path, start_time, end_time, num_cores=32,
):
    varnames = {}
    varnames["time"] = "Epoch"
    varnames["energy_channels"] = "Energy_FEDU"
    varnames["Flux"] = "FEDU"
    varnames["alpha_local"] = "PA_local_FEDU"
    varnames["xGEO"] = "xGEO"
    varnames["PSD"] = "PSD_FEDU"
    varnames["alpha_eq_model"] = "PA_eq_T89"
    varnames["MLT"] = "MLT_T89"
    varnames["Lstar"] = "Lstar_T89"
    varnames["Lm"] = "Lm_T89"
    varnames["R_eq"] = "R_eq_T89"
    varnames["PSD"] = "PSD_FEDU"
    varnames["B_eq"] = "B_eq_T89"
    varnames["B_local"] = "B_local_T89"
    varnames["InvMu"] = "invMu_T89"
    varnames["InvK"] = "invK_T89"
    save_standard = DataorgPMF(
        mission="DSX",
        source="",
        instrument="orbit",
        save_text_segments=[save_data_dir, "orbit", "n4", "4", "T89", "ver4"],
        product_variable_names=varnames,
    )

    # Part 1: specify source files to extract variables

    time_var = TimeVariable(name_or_column_in_file="DATETIME", original_unit="", standard_name="Epoch_posixtime")

    variables_to_extract = {
        "Epoch": time_var,
        "alt": Variable(name_or_column_in_file="alt(km)",
                        time_variable=time_var,
                        original_unit=u.km,
                        time_bin_method=TimeBinMethod.Mean),
        "lat": Variable(name_or_column_in_file="lat(deg)",
                        time_variable=time_var,
                        original_unit=u.deg,
                        time_bin_method=TimeBinMethod.Mean),
        "lon": Variable(name_or_column_in_file="lon(deg)",
                        time_variable=time_var,
                        original_unit=u.deg,
                        time_bin_method=TimeBinMethod.Mean),
    }

    source_file = SourceFile(
        download_path="/home/bhaas/data/dsx_ephem_for_pager.csv",
        variables_to_extract=variables_to_extract,
    )

    variables: dict[str, Variable] = load_variables_from_source_files(source_file, start_time, end_time)

    # Part 2: standardize variables
    time_var.data = np.asarray([dateutil.parser.parse(t).replace(tzinfo=timezone.utc) for t in time_var.data])
    time_var_datetime = time_var.data

    time_var.metadata.unit = u.posixtime
    time_var.data = [t.timestamp() for t in time_var.data]

    xGDZ_arr = np.stack((variables["alt"].data, variables["lat"].data, variables["lon"].data))
    xGEO_var = Variable(time_variable=time_var, standard_name="xGEO", original_unit=u.RE, time_bin_method=TimeBinMethod.Mean)

    model_coord = Coords(path=irbem_lib_path)

    # convert time_array to datenums for transform function
    xGEO_var.data = model_coord.transform(time_var_datetime, xGDZ_arr, IRBEM_SYSAXIS_GDZ, IRBEM_SYSAXIS_GEO)

    variables["xGEO"] = xGEO_var

    PA_local_var = Variable(time_variable=None, original_unit=u.deg, standard_name="PA_local", time_bin_method=TimeBinMethod.Repeat)
    PA_local_var.data = np.asarray([0.3, 89.7])
    variables["PA_local"] = PA_local_var

    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement="center")

    # Part 3: compute derived variables

    magnetic_field_variables = {
        "B_local_T89": DerivedVariable(standard_name="B_local_T89"),
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        "R_eq_T89": DerivedVariable(standard_name="R_eq_T89"),
        "B_eq_T89": DerivedVariable(standard_name="B_eq_T89"),
        "Lstar_T89": DerivedVariable(standard_name="Lstar_T89"),
        "PA_eq_T89": DerivedVariable(standard_name="PA_eq_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    # Part 4: save variables according to save standard

    save_standard.save(start_time, end_time, variables | magnetic_field_variables)
