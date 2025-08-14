import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from astropy import units as u
from dateutil import parser
from numpy.typing import NDArray

import el_paso as ep

logging.captureWarnings(True)


def _weight_energy_channels_exponentially(energy_ranges: list[str]) -> NDArray[np.float64]:
    """Calculate the exponential weighing of the two numbers in the energy range string."""
    b = 7.068e-3  # constant as in the MATLAB code

    weighted_energy_channels:list[float] = []

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
    # Part 1: specify source files to extract variables

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="time_tag",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDO",
            name_or_column="flux",
            unit=(u.cm**2 * u.s * u.keV) ** (-1),
            dependent_variables=["time_tag", "energy"],
        ),
        ep.ExtractionInfo(
            result_key="sat_id",
            name_or_column="satellite",
            unit=u.dimensionless_unscaled,
            is_time_dependent=False,
        ),
    ]

    data_path_stem = f"{download_data_dir}goes/YYYY/MM/{satellite_str}/"
    rename_file_name_stem = f"{satellite_str}_YYYYMMDD.json"
    url = f"https://services.swpc.noaa.gov/json/goes/{satellite_str}/"

    ep.download(start_time, end_time,
                save_path=data_path_stem, file_cadence="single_file",
                download_url=url, file_name_stem="differential-electrons-3-day.json",
                rename_file_name_stem=rename_file_name_stem)

    variables = ep.extract_variables_from_files(start_time, end_time,
                                                file_cadence="single_file", data_path=data_path_stem,
                                                file_name_stem=rename_file_name_stem,
                                                extraction_infos=extraction_infos)

    # parse time strings
    datetimes = ep.processing.convert_string_to_datetime(variables["Epoch"])
    variables["Epoch"].set_data(np.asarray([t.timestamp() for t in datetimes]), u.posixtime)

    # generated weighted energy channels
    variables["Energy"].set_data(_weight_energy_channels_exponentially(variables["Energy"].get_data()), "same")

    # Get the sorting order based on the row
    sorting_order = np.argsort(variables["Energy"].get_data())

    # Apply the sorting order to all rows
    variables["Energy"].set_data(variables["Energy"].get_data()[sorting_order], "same")
    variables["FEDO"].set_data(variables["FEDO"].get_data()[:, sorting_order], "same")
    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)

    time_bin_methods = {
        "FEDO": ep.TimeBinMethod.NanMedian,
        "Energy": ep.TimeBinMethod.Repeat,
    }

    binned_time_var = ep.processing.bin_by_time(time_variable=variables["Epoch"],
                                                variables=variables,
                                                time_bin_method_dict=time_bin_methods,
                                                time_binning_cadence=timedelta(minutes=5))

    sat_name = "goes" + str(variables["sat_id"].get_data()[0])

    variables["xGEO"] = ep.processing.get_real_time_tipsod(binned_time_var.get_data(), sat_name)

    # Local pitch angles from 5 to 90 deg
    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    # Calculate magnetic field variables
    # var_names_to_compute = ["B_local_T89", "MLT_T89", "B_eq_T89", "R_eq_T89", "PA_eq_T89", "invMu_T89", "invK_T89", "Lstar_T89"]
    var_names_to_compute = ["B_local_T89", "MLT_T89", "B_eq_T89", "R_eq_T89", "PA_eq_T89", "invMu_T89"]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(time_var = binned_time_var,
                                                                              xgeo_var = variables["xGEO"],
                                                                              energy_var = variables["Energy"],
                                                                              pa_local_var = variables["PA_local_FEDU"],
                                                                              particle_species = "electron",
                                                                              var_names_to_compute = var_names_to_compute,
                                                                              irbem_lib_path = irbem_lib_path,
                                                                              irbem_options = [1, 1, 4, 4, 0],
                                                                              num_cores = num_cores)

    FEDU_var = ep.processing.construct_pitch_angle_distribution(variables["FEDO"],
                                                                variables["PA_local_FEDU"],
                                                                magnetic_field_variables["PA_eq_T89"])
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    PSD_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy"], particle_species="electron")

    variables_to_save = {
        "time": variables["Epoch"],
        "Flux": FEDU_var,
        "xGEO": variables["xGEO"],
        "energy_channels": variables["Energy"],
        "alpha_local": variables["PA_local_FEDU"],
        "PSD": PSD_var,
        "alpha_eq_model": magnetic_field_variables["PA_eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        # "Lstar": magnetic_field_variables["Lstart_T89"],
        "R0": magnetic_field_variables["R_eq_T89"],
        "B_eq": magnetic_field_variables["B_eq_T89"],
        "B_local": magnetic_field_variables["B_local_T89"],
        "InvMu": magnetic_field_variables["invMu_T89"],
        # "InvK": magnetic_field_variables["invK_T89"]
    }

    saving_strategy = ep.saving_strategies.DataOrgStrategy(save_data_dir,
                                                           mission="GOES",
                                                           satellite=satellite_str,
                                                           instrument="MAGED",
                                                           kext="T89",
                                                           file_format=".pickle")
    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=variables["Epoch"], append=True)

if __name__ == "__main__":
    start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=0.1)

    for i in ["primary", "secondary"]:
        process_goes_real_time(
            satellite_str=i,
            download_data_dir="goes/raw/",
            save_data_dir="goes/processed/",
            irbem_lib_path="../IRBEM/libirbem.so",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )