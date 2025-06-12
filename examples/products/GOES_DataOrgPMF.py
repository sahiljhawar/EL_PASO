import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from dateutil import parser
from datetime import datetime, timezone

from el_paso.classes import DerivedVariable, SourceFile, TimeBinMethod, TimeVariable, Variable
from el_paso.processing import (
    compute_magnetic_field_variables,
    compute_PSD,
    construct_pitch_angle_distribution,
    convert_all_data_to_standard_units,
    get_real_time_tipsod,
    time_bin_all_variables,
)
from el_paso.save_standards.real_time_mat import RealtimeMat
from el_paso.save_standards.data_org import DataorgPMF
from el_paso.utils import fill_str_template_with_time
import logging
from datetime import timedelta
from datetime import datetime, timezone
from examples.products.GOES_realtime import process_goes_real_time
from examples.products.Arase_realtime import process_arase_xep_real_time
from pathlib import Path
import time


logging.captureWarnings(True)


def _weight_energy_channels_exponentially(energy_ranges: list[str]):
    """Calculate the exponential weighing of the two numbers in the energy range string."""
    b = 7.068e-3  # constant as in the MATLAB code

    weighted_energy_channels = []

    for energy_range in energy_ranges:
        if isinstance(energy_range, str):
            match = re.match(r"(\d+)-(\d+)", energy_range)
            if match:
                E_min = float(match.group(1))  # noqa: N806
                E_max = float(match.group(2))  # noqa: N806
                # Calculate the exponential weighing, translated from MATLAB
                E_mean = np.log(  # noqa: N806
                    (((-1.0 / b) * np.exp(-b * E_max)) + (1.0 / b) * np.exp(-b * E_min)) / (E_max - E_min)
                ) / (-b)

                weighted_energy_channels.append(E_mean)

    return np.asarray(weighted_energy_channels)


def process_goes_real_time(
    satellite_str: Literal["primary", "secondary"],
    save_data_dir: str,
    download_data_dir: str,
    irbem_lib_path: str,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
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
    varnames["density"] = "density"


    save_standard = DataorgPMF(
        mission="GOES",
        source=satellite_str,
        instrument="MAGED",
        save_text_segments=[save_data_dir, f"tmp_{satellite_str}", "n4", "4", "T89", "ver4"],
        product_variable_names=varnames,
        file_format=".pickle",
    )

    # Part 1: specify source files to extract variables

    time_var = TimeVariable(name_or_column_in_file="time_tag", standard_name="Epoch_posixtime", original_unit="")

    energy_var = Variable(
        name_or_column_in_file="energy",
        standard_name="Energy_FEDO",
        original_unit=u.keV,
        time_bin_method=TimeBinMethod.Repeat,
        time_variable=None,
    )

    variables_to_extract = {
        "Epoch": time_var,
        "Energy_FEDO": energy_var,
        "FEDO": Variable(
            name_or_column_in_file="flux",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
            dependent_variables=[energy_var],
        ),
        "sat_id": Variable(
            name_or_column_in_file="satellite",
            original_unit="",
            time_variable=None,
            time_bin_method=TimeBinMethod.NoBinning,
        ),
    }

    data_path_stem = f"{download_data_dir}goes/YYYY/MM/{satellite_str}/"
    file_name_stem = f"{satellite_str}_YYYYMMDD.json"

    Path(fill_str_template_with_time(data_path_stem, start_time)).mkdir(exist_ok=True, parents=True)

    #donwload 3 day file
    source_file = SourceFile(
        download_url=f"https://services.swpc.noaa.gov/json/goes/{satellite_str}/differential-electrons-3-day.json",
        download_arguments_prefixes="",
        download_arguments_suffixes=f"-O {data_path_stem}{file_name_stem}",
        download_path=f"{data_path_stem}{file_name_stem}",
        variables_to_extract=variables_to_extract,
        file_cadence="custom_one_file",
    )

    source_file.download(start_time, end_time)
    variables: dict[str, Variable] = source_file.extract_variables(start_time, end_time)

    # parse time strings
    variables["Epoch"].data = np.asarray([parser.parse(t).timestamp() for t in variables["Epoch"].data])
    variables["Epoch"].metadata.unit = u.posixtime

    # generated weighted energy channels
    variables["Energy_FEDO"].data = _weight_energy_channels_exponentially(variables["Energy_FEDO"].data)

    # Get the sorting order based on the row
    sorting_order = np.argsort(variables["Energy_FEDO"].data)

    # Apply the sorting order to all rows
    variables["Energy_FEDO"].data = variables["Energy_FEDO"].data[sorting_order]
    variables["FEDO"].data = variables["FEDO"].data[:, sorting_order]
    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)

    convert_all_data_to_standard_units(variables)
    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement="center")

    # Add additional variables
    variables |= {
        "PA_local_FEDU": Variable(original_unit=u.deg, standard_name="PA_local", time_variable=time_var),
        "xGEO": Variable(
            name_or_column_in_file="Position_Ele",
            standard_name="xGEO",
            original_unit=u.km,
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
    }

    sat_name = "goes" + str(variables["sat_id"].data[0])

    variables["xGEO"].data = get_real_time_tipsod(variables["xGEO"].time_variable.data, sat_name)
    variables["xGEO"].metadata.unit = u.km
    variables["xGEO"].convert_to_standard_unit()

    # Local pitch angles from 5 to 90 deg
    variables["PA_local_FEDU"].data = np.tile(np.arange(5, 91, 5), (len(time_var.data), 1))

    # Calculate magnetic field variables
    magnetic_field_variables = {
        "B_local_T89": DerivedVariable(standard_name="B_local_T89"),
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        "B_eq_T89": DerivedVariable(standard_name="B_eq_T89"),
        "R_eq_T89": DerivedVariable(standard_name="R_eq_T89"),
        "Lstar_T89": DerivedVariable(standard_name="Lstar_T89"),
        "PA_eq_T89": DerivedVariable(standard_name="PA_eq_T89"),
        "invMu_T89": DerivedVariable(standard_name="invMu_T89"),
        "invK_T89": DerivedVariable(standard_name="invK_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    variables |= magnetic_field_variables

    # generate differential flux
    FEDU_var = Variable(  # noqa: N806
        time_variable=time_var,
        standard_name="FEDU",
        original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
    )

    FEDU_var.data = construct_pitch_angle_distribution(
        variables,
        pa_eq_key="PA_eq_T89",
        omni_flux_key="FEDO",
    )

    variables["FEDU"] = FEDU_var
    variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    PSD_var = DerivedVariable(standard_name="PSD_FEDU")  # noqa: N806

    compute_PSD(variables, PSD_var, flux_key="FEDU")
    variables["PSD_FEDU"] = PSD_var

    save_standard.save(start_time, end_time, variables, append=True)
