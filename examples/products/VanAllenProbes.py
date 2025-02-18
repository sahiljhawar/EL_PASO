from datetime import timedelta

import numpy as np
from astropy import units as u

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


def process_rbsp_hope_electron(
    satellite_str, save_data_dir, download_data_dir, irbem_lib_path, start_time, end_time, num_cores=32
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
        mission="RBSP",
        source=satellite_str,
        instrument="HOPE-electron-l3",
        save_text_segments=[save_data_dir, satellite_str, "n4", "4", "T89", "ver4"],
        product_variable_names=varnames,
    )

    # Part 1: specify source files to extract variables

    time_var = TimeVariable(name_or_column_in_file="Epoch_Ele", original_unit=u.tt2000, standard_name="Epoch_posixtime")

    variables_to_extract = {
        "Epoch": time_var,
        "Energy_FEDU": Variable(
            name_or_column_in_file="HOPE_ENERGY_Ele",
            standard_name="Energy_FEDU",
            original_unit=u.eV,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDU": Variable(
            name_or_column_in_file="FEDU",
            standard_name="FEDU",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "PA_local_FEDU": Variable(
            name_or_column_in_file="PITCH_ANGLE", original_unit=u.deg, standard_name="PA_local", time_variable=None
        ),
        "xGEO": Variable(
            name_or_column_in_file="Position_Ele",
            standard_name="xGEO",
            original_unit=u.km,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
    }

    data_path_stem = f"{download_data_dir}rbsp/{satellite_str}/hope/l3/pitchangle/YYYY/"
    file_name_stem = f"rbsp{satellite_str[-1]}_rel04_ect-hope-pa-l3_YYYYMMDD*.cdf"

    source_file = SourceFile(
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/{satellite_str}/l3/ect/hope/pitchangle/rel04/YYYY/",
        download_arguments_prefixes=f"-r -np -N -nH -e robots=off --cut-dirs=10 --accept '{file_name_stem}'",
        download_arguments_suffixes=f"-P {data_path_stem}",
        download_path=f"{data_path_stem}{file_name_stem}",
        variables_to_extract=variables_to_extract,
    )

    # source_file.download(start_time, end_time)

    variables: dict[str, Variable] = load_variables_from_source_files(source_file, start_time, end_time)

    # Part 2: standardize variables

    variables["FEDU"].transpose_data((0, 2, 1))
    variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    convert_all_data_to_standard_units(variables)
    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement="center")

    # Part 3: compute derived variables

    fold_pitch_angles_and_flux(variables, produce_statistic_plot=False)

    magnetic_field_variables = {
        "B_local_T89": DerivedVariable(standard_name="B_local_T89"),
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        #"R_eq_T89": DerivedVariable(standard_name="R_eq_T89"),
        "B_eq_T89": DerivedVariable(standard_name="B_eq_T89"),
        #"Lstar_T89": DerivedVariable(standard_name="Lstar_T89"),
        #"PA_eq_T89": DerivedVariable(standard_name="PA_eq_T89"),
        "invMu_T89": DerivedVariable(standard_name="invMu_T89"),
        "invK_T89": DerivedVariable(standard_name="invK_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    PSD_var = DerivedVariable(standard_name="PSD_FEDU")

    compute_PSD(variables, PSD_var)
    variables["PSD_FEDU"] = PSD_var

    # Part 4: save variables according to save standard

    save_standard.save(start_time, end_time, variables | magnetic_field_variables)
