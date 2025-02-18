import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.convolution import interpolate_replace_nans

from el_paso import IRBEM_SYSAXIS_GDZ, IRBEM_SYSAXIS_GEO
from el_paso.classes import DerivedVariable, SourceFile, TimeBinMethod, TimeVariable, Variable
from el_paso.processing import (
    compute_magnetic_field_variables,
    compute_PSD,
    convert_all_data_to_standard_units,
    extrapolate_leo_to_equatorial,
    fold_pitch_angles_and_flux,
    load_variables_from_source_files,
    time_bin_all_variables,
)
from el_paso.save_standards.data_org import DataorgPMF
from IRBEM import Coords

poes_satellite_literal = Literal["metop1", "metop2", "metop3", "noaa05", "noaa06", "noaa07", "noaa08", "noaa10",
                                 "noaa12", "noaa14", "noaa15", "noaa16", "noaa17", "noaa18", "noaa19"]

def process_poes_ted_electron(
    satellite_str:poes_satellite_literal,
    save_data_dir:str,
    download_data_dir:str,
    irbem_lib_path:str,
    start_time:datetime,
    end_time:datetime,
    num_cores:int=32
):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
    varnames["R0"] = "R_eq_T89"
    varnames["PSD"] = "PSD_FEDU"
    varnames["B_eq"] = "B_eq_T89"
    varnames["B_local"] = "B_local_T89"
    varnames["InvMu"] = "invMu_T89"
    varnames["InvK"] = "invK_T89"
    save_standard = DataorgPMF(
        mission="POES",
        source=satellite_str,
        instrument="TED-electron",
        save_text_segments=[save_data_dir, satellite_str, "n4", "4", "T89", "ver4"],
        product_variable_names=varnames,
    )

    # Part 1: specify source files to extract variables

    time_var = TimeVariable(name_or_column_in_file="Epoch", original_unit=u.tt2000, standard_name="Epoch_posixtime")

    variables_to_extract = {
        "Epoch": time_var,
        "Energy_FEDU": Variable(
            name_or_column_in_file="ted_ele_diff_energies",
            standard_name="Energy_FEDU",
            original_unit=u.eV,
            time_bin_method=TimeBinMethod.Repeat,
            time_variable=None,
        ),
        "FEDU": Variable(
            name_or_column_in_file="ted_ele_flux",
            standard_name="FEDU",
            original_unit=(u.cm**2 * u.s * u.sr * u.eV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "PA_local_FEDU_tel0": Variable(
            name_or_column_in_file="ted_alpha_0_sat",
            original_unit=u.deg,
            standard_name="PA_local",
            time_variable=time_var,
            time_bin_method=TimeBinMethod.NanMean,
        ),
        "PA_local_FEDU_tel30": Variable(
            name_or_column_in_file="ted_alpha_30_sat",
            original_unit=u.deg,
            standard_name="PA_local",
            time_variable=time_var,
            time_bin_method=TimeBinMethod.NanMean,
        ),
        "alt": Variable(
            name_or_column_in_file="alt",
            original_unit=u.km,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "lat": Variable(
            name_or_column_in_file="lat",
            original_unit=u.deg,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "lon": Variable(
            name_or_column_in_file="lon",
            original_unit=u.deg,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
    }

    data_path_stem = f"{download_data_dir}/{satellite_str}/YYYY/MM/"
    file_name_stem = f"{satellite_str}_poes-sem2_fluxes-2sec_YYYYMMDD*.cdf"

    source_file = SourceFile(
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite_str}/sem2_fluxes-2sec/YYYY/",
        download_arguments_prefixes=f"-r -np -N -nH -e robots=off --cut-dirs=10 --accept '{file_name_stem}'",
        download_arguments_suffixes=f"-P {data_path_stem}",
        download_path=f"{data_path_stem}{file_name_stem}",
        variables_to_extract=variables_to_extract,
    )

    #source_file.download(start_time, end_time)

    variables: dict[str, Variable] = load_variables_from_source_files(source_file, start_time, end_time)

    # Part 2: standardize variables

    variables["FEDU"].transpose_data((0, 2, 1))
    variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    variables["PA_local_FEDU"] = Variable(time_variable=time_var, original_unit=u.deg, time_bin_method=TimeBinMethod.Mean, standard_name="PA_local")
    variables["PA_local_FEDU"].data = np.stack((variables["PA_local_FEDU_tel0"].data, variables["PA_local_FEDU_tel30"].data)).T

    del variables["PA_local_FEDU_tel0"]
    del variables["PA_local_FEDU_tel30"]

    convert_all_data_to_standard_units(variables)
    time_bin_all_variables(variables, timedelta(minutes=0.5), start_time, end_time, window_alignement="center")

    for ie in range(4):
        variables["FEDU"].data[:,ie,0] = interpolate_replace_nans(variables["FEDU"].data[:,ie,0], [1/2, 0, 1/2])
        variables["FEDU"].data[:,ie,1] = interpolate_replace_nans(variables["FEDU"].data[:,ie,1], [1/2, 0, 1/2])

    for it in range(len(time_var.data)):
        mask = np.isfinite(variables["FEDU"].data[it,:,0])
        if np.sum(mask) == 0:
            continue

        variables["FEDU"].data[it,:,0] = np.interp(np.log10(variables["Energy_FEDU"].data[it,:]),
                                                   np.log10(variables["Energy_FEDU"].data[it,mask]),
                                                   variables["FEDU"].data[it,mask,0],
                                                   left=np.nan, right=np.nan)

    for it in range(len(time_var.data)):
        mask = np.isfinite(variables["FEDU"].data[it,:,1])
        if np.sum(mask) == 0:
            continue

        variables["FEDU"].data[it,:,1] = np.interp(np.log10(variables["Energy_FEDU"].data[it,:]),
                                                   np.log10(variables["Energy_FEDU"].data[it,mask]),
                                                   variables["FEDU"].data[it,mask,1],
                                                   left=np.nan, right=np.nan)

    variables["FEDU"].data[:,:,1] = np.max(variables["FEDU"].data, axis=2)

    value_0 = np.sum(np.sum(np.isnan(variables["FEDU"].data[:,:,0]), axis=1) == 0)
    value_1 = np.sum(np.sum(np.isnan(variables["FEDU"].data[:,:,0]), axis=1) == 1)
    value_2 = np.sum(np.sum(np.isnan(variables["FEDU"].data[:,:,0]), axis=1) == 2)
    value_3 = np.sum(np.sum(np.isnan(variables["FEDU"].data[:,:,0]), axis=1) == 3)
    value_4 = np.sum(np.sum(np.isnan(variables["FEDU"].data[:,:,0]), axis=1) == 4)

    print(value_1)
    print(value_2)
    print(value_3)
    print(value_4)

    print((value_1 + value_2 + value_3 + value_4) / variables["FEDU"].data.shape[0] * 100)

    xGDZ_arr = np.stack((variables["alt"].data, variables["lat"].data, variables["lon"].data)).T
    xGEO_var = Variable(time_variable=time_var, standard_name="xGEO", original_unit=u.RE, time_bin_method=TimeBinMethod.Mean)

    model_coord = Coords(path=irbem_lib_path)

    # convert time_array to datetimes for transform function
    time_var_datetime = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_var.data]
    xGEO_var.data = model_coord.transform(time_var_datetime, xGDZ_arr, IRBEM_SYSAXIS_GDZ, IRBEM_SYSAXIS_GEO)
    variables["xGEO"] = xGEO_var

    # Part 3: compute derived variables

    magnetic_field_variables = {
        "B_local_T89": DerivedVariable(standard_name="B_local_T89"),
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        "xGEO_T89": DerivedVariable(standard_name="xGEO_T89"),
        "R_eq_T89": DerivedVariable(standard_name="R_eq_T89"),
        "B_eq_T89": DerivedVariable(standard_name="B_eq_T89"),
        #"Lstar_T89": DerivedVariable(standard_name="Lstar_T89"),
        "PA_eq_T89": DerivedVariable(standard_name="PA_eq_T89"),
        # "invMu_T89": DerivedVariable(standard_name="invMu_T89"),
        # "invK_T89": DerivedVariable(standard_name="invK_T89"),
        "B_fofl_T89": DerivedVariable(standard_name="B_fofl_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    variables |= magnetic_field_variables

    # extrapolate to higher pitch angles
    pa_eq_extrap_var = Variable(time_var, original_unit=u.deg, standard_name="PA_eq_T89")
    pa_eq_extrap_var.data = np.tile(np.arange(5, 91, 10), (len(time_var.data), 1))

    flux_extrap_var = Variable(time_var, original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1), standard_name="FEDU")
    flux_extrap_var.data = np.full((len(time_var.data), 4, pa_eq_extrap_var.data.shape[1]), np.nan)

    extrapolate_leo_to_equatorial(magnetic_field_variables["PA_eq_T89"],
                                  variables["FEDU"],
                                  pa_eq_extrap_var,
                                  flux_extrap_var,
                                  magnetic_field_variables["B_fofl_T89"],
                                  magnetic_field_variables["B_local_T89"],
                                  magnetic_field_variables["B_eq_T89"])

    variables["PA_eq_T89"] = pa_eq_extrap_var
    variables["FEDU"] = flux_extrap_var

    print(np.sum(np.isnan(flux_extrap_var.data)) / flux_extrap_var.data.size * 100)

    magnetic_field_variables = {
        "invMu_T89": DerivedVariable(standard_name="invMu_T89"),
        "invK_T89": DerivedVariable(standard_name="invK_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores,
                                     pa_local_key="PA_eq_T89", xgeo_key="xGEO_T89")

    PSD_var = DerivedVariable(standard_name="PSD_FEDU")

    compute_PSD(variables, PSD_var)
    variables["PSD_FEDU"] = PSD_var

    # Part 4: save variables according to save standard

    save_standard.save(start_time, end_time, variables | magnetic_field_variables)
