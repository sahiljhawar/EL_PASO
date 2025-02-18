from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u
from dateutil import parser
from numpy.typing import NDArray

from el_paso import IRBEM_SYSAXIS_GEO, IRBEM_SYSAXIS_SM
from el_paso.classes import DerivedVariable, SourceFile, TimeBinMethod, TimeVariable, Variable
from el_paso.classes.sourcefile import FileCadence
from el_paso.processing import (
    compute_equatorial_plasmaspheric_density,
    compute_magnetic_field_variables,
    compute_PSD,
    construct_pitch_angle_distribution,
    convert_all_data_to_standard_units,
    convert_string_to_datetime,
    time_bin_all_variables,
)
from el_paso.save_standards.data_org import DataorgPMF
from el_paso.save_standards.real_time_mat import RealtimeMat
from IRBEM import Coords


def process_arase_xep_real_time(
    save_data_dir:str,
    download_data_dir:str,
    irbem_lib_path:str,
    start_time:datetime,
    end_time:datetime,
    num_cores:int=32,
) -> None:

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    varnames = {}
    varnames["newtime"] = "Epoch"
    varnames["Energy"] = "Energy_FEDO"
    varnames["Flux"] = "FEDU"
    varnames["Pitch_Angles"] = "PA_local_FEDU"
    varnames["alpha_loc"] = "PA_local_FEDU"
    varnames["alpha_eq"] = "PA_eq_T89"
    varnames["Position_GEO"] = "xGEO"
    varnames["PSD"] = "PSD_FEDU"
    varnames["MLT"] = "MLT_T89"
    varnames["Lstar"] = "Lstar_T89"
    varnames["B_eq"] = "B_eq_T89"
    varnames["B_loc"] = "B_local_T89"
    varnames["InvMu"] = "invMu_T89"
    varnames["InvK"] = "invK_T89"

    save_standard = RealtimeMat(
        mission="Arase",
        source="Arase",
        instrument="XEP",
        save_text_segments=["arase", "xep", "T89", "oneraextrap"],
        product_variable_names = varnames,
    )

    orb_variables = _get_orb_variables(download_data_dir, start_time, end_time, irbem_lib_path)
    xep_variables = _get_xep_variables(download_data_dir, start_time, end_time)

    variables = orb_variables | xep_variables

    convert_all_data_to_standard_units(variables)
    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement="center")

    time_var = variables["FEDO"].time_variable

    # add local pitch angle variable
    variables["PA_local_FEDU"] = Variable(original_unit=u.deg, standard_name="PA_local", time_variable=time_var)

    # Local pitch angles from 5 to 90 deg
    variables["PA_local_FEDU"].data = np.tile(np.arange(5, 91, 5), (len(time_var.data), 1))

    # Calculate magnetic field variables
    magnetic_field_variables = {
        "B_local_T89": DerivedVariable(standard_name="B_local_T89"),
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        "B_eq_T89": DerivedVariable(standard_name="B_eq_T89"),
        "Lstar_T89": DerivedVariable(standard_name="Lstar_T89"),
        "PA_eq_T89": DerivedVariable(standard_name="PA_eq_T89"),
        "invMu_T89": DerivedVariable(standard_name="invMu_T89"),
        "invK_T89": DerivedVariable(standard_name="invK_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    variables |= magnetic_field_variables

    # generate differential flux
    FEDU_var = Variable(  # noqa: N806
        time_variable=time_var, standard_name="FEDU", original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
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

    save_standard.save(start_time, end_time, variables)

def process_arase_pew_real_time(
    save_data_dir:str,
    download_data_dir:str,
    irbem_lib_path:str,
    start_time:datetime,
    end_time:datetime,
    num_cores:int=32,
) -> None:

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    varnames = {}
    varnames["time"] = "Epoch"
    varnames["xGEO"] = "xGEO"
    varnames["MLT"] = "MLT_T89"
    varnames["R0"] = "R_eq_T89"
    varnames["density"] = "Density"
    varnames["energy_channels"] = "Energy_FEDU"
    varnames["Flux"] = "FEDU"
    varnames["alpha_local"] = "PA_local_FEDU"
    varnames["PSD"] = "PSD_FEDU"
    varnames["alpha_eq_model"] = "PA_eq_T89"
    varnames["Lstar"] = "Lstar_T89"
    varnames["Lm"] = "Lm_T89"
    varnames["PSD"] = "PSD_FEDU"
    varnames["B_eq"] = "B_eq_T89"
    varnames["B_local"] = "B_local_T89"
    varnames["InvMu"] = "invMu_T89"
    varnames["InvK"] = "invK_T89"

    save_standard = DataorgPMF(
        mission="Arase",
        source="arase",
        instrument="PWE-density",
        save_text_segments=[save_data_dir, "arase", "n4", "4", "T89", "ver4"],
        product_variable_names=varnames,
    )

    pew_variables = _get_pew_variables(start_time, end_time)
    orb_variables = _get_orb_variables(download_data_dir, start_time, end_time, irbem_lib_path)

    variables = orb_variables | pew_variables

    convert_all_data_to_standard_units(variables)
    time_bin_all_variables(variables, timedelta(minutes=5), start_time, end_time, window_alignement="center")

    # Calculate magnetic field variables
    magnetic_field_variables = {
        "MLT_T89": DerivedVariable(standard_name="MLT_T89"),
        "R_eq_T89": DerivedVariable(standard_name="R_eq_T89"),
    }

    compute_magnetic_field_variables(variables, magnetic_field_variables, irbem_lib_path, [1, 1, 4, 4, 0], num_cores)

    variables |= magnetic_field_variables

    variables["Density"].apply_thresholds_on_data(lower_threshold=0)

    variables["R_local"] = DerivedVariable()
    variables["R_local"].data = np.linalg.norm(variables["xGEO"].data, 2, axis=1)

    compute_equatorial_plasmaspheric_density(variables["Density"], variables["R_eq_T89"], variables["R_local"])

    save_standard.save(start_time, end_time, variables)

def _get_pew_variables(start_time:datetime, end_time:datetime) -> dict[str,Variable]:

    time_var = TimeVariable(name_or_column_in_file="Epoch", standard_name="Epoch_posixtime", original_unit=u.tt2000)

    pew_variables_to_extract = {
        "Epoch": time_var,
        "Density": Variable(
            name_or_column_in_file="ne_mgf",
            original_unit=u.cm**(-3),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
    }

    file_name_stem = "erg_pwe_hfa_l3_1min_YYYYMMDD_v05_08.cdf"

    pew_source_file = SourceFile(
        download_path=f"/home/bhaas/data/da_data/may_plasmasphere/{file_name_stem}",
        variables_to_extract=pew_variables_to_extract,
        file_cadence=FileCadence.DAILY,
    )

    # Bernhard: the header is also in the file, but there is a comment after it, so it cannot be read by pd.read_csv
    pew_variables = pew_source_file.extract_variables(start_time, end_time)

    return pew_variables

def _get_xep_variables(download_data_dir:str, start_time:datetime, end_time:datetime) -> dict[str,Variable]:

    # Part 1: specify source files to extract variables

    # Energies from the User's guide
    energy_min = np.asarray((400., 600., 1000., 1500., 2200., 3500., 4300., 5400.))
    energy_max = np.asarray((600., 1000., 1500., 2200., 3500., 4300., 5400., 9800.))
    energy_mean = _get_mean_energy(energy_min, energy_max)

    time_var = TimeVariable(name_or_column_in_file="time", standard_name="Epoch_posixtime", original_unit="")

    xep_variables_to_extract = {
        "Epoch": time_var,
        "FEDO_ch1": Variable(
            name_or_column_in_file="ch1",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch2": Variable(
            name_or_column_in_file="ch2",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch3": Variable(
            name_or_column_in_file="ch3",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch4": Variable(
            name_or_column_in_file="ch4",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch5": Variable(
            name_or_column_in_file="ch5",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch6": Variable(
            name_or_column_in_file="ch6",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch7": Variable(
            name_or_column_in_file="ch7",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
        "FEDO_ch8": Variable(
            name_or_column_in_file="ch8",
            standard_name="FEDO",
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            time_bin_method=TimeBinMethod.NanMedian,
            time_variable=time_var,
        ),
    }


    data_path_stem = f"{download_data_dir}arase/YYYY/MM/"
    file_name_stem = "erg_real_xep_YYYYMMDD_v002.txt"

    #Path(fill_str_template_with_time(data_path_stem, start_time)).mkdir(exist_ok=True, parents=True)

    xep_source_file = SourceFile(
        download_url=f"https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/swx/xep/l2/{file_name_stem}",
        download_arguments_prefixes="--user erg_sw_users --password arase_swx",
        download_arguments_suffixes=f"-O {data_path_stem}{file_name_stem}",
        download_path=f"{data_path_stem}{file_name_stem}",
        variables_to_extract=xep_variables_to_extract,
        file_cadence=FileCadence.DAILY,
    )

    xep_source_file.download(start_time, end_time)

    # Bernhard: the header is also in the file, but there is a comment after it, so it cannot be read by pd.read_csv
    xep_header = ("time", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8") 
    xep_variables = xep_source_file.extract_variables(start_time, end_time, pd_read_csv_kwargs={"skiprows":6, "names":xep_header})

    # convert time variable
    # parse time strings
    convert_string_to_datetime(xep_variables["Epoch"])
    xep_variables["Epoch"].data = np.asarray([t.timestamp() for t in xep_variables["Epoch"].data])
    xep_variables["Epoch"].metadata.unit = u.posixtime

    # add energy variable
    energy_var = Variable(
        standard_name="Energy_FEDO",
        original_unit=u.keV,
        time_bin_method=TimeBinMethod.Repeat,
        time_variable=None,
    )
    energy_var.data = energy_mean
    xep_variables["Energy_FEDO"] = energy_var

    # build flux variable from channels
    flux_var = Variable(time_variable=time_var,
                        original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
                        time_bin_method=TimeBinMethod.NanMedian,
                        dependent_variables=energy_var)

    flux_var.data = np.vstack((xep_variables["FEDO_ch1"].data, xep_variables["FEDO_ch2"].data, xep_variables["FEDO_ch3"].data, 
                               xep_variables["FEDO_ch4"].data, xep_variables["FEDO_ch5"].data, xep_variables["FEDO_ch6"].data,
                               xep_variables["FEDO_ch7"].data, xep_variables["FEDO_ch8"].data)).T

    xep_variables["FEDO"] = flux_var

    # delete unused variables
    del xep_variables["FEDO_ch1"]
    del xep_variables["FEDO_ch2"]
    del xep_variables["FEDO_ch3"]
    del xep_variables["FEDO_ch4"]
    del xep_variables["FEDO_ch5"]
    del xep_variables["FEDO_ch6"]
    del xep_variables["FEDO_ch7"]
    del xep_variables["FEDO_ch8"]

    return xep_variables

def _get_orb_variables(download_data_dir:str, start_time:datetime, end_time:datetime, irbem_lib_path:str) -> dict[str,Variable]:

    time_var = TimeVariable(name_or_column_in_file="time", standard_name="Epoch_posixtime", original_unit="")

    orb_variables_to_extract = {
        "Epoch": time_var,
        "X_sm": Variable(
            name_or_column_in_file="sm_x",
            original_unit=u.RE,
            time_bin_method=TimeBinMethod.NanMean,
            time_variable=time_var,
        ),
        "Y_sm": Variable(
            name_or_column_in_file="sm_y",
            original_unit=u.RE,
            time_bin_method=TimeBinMethod.NanMean,
            time_variable=time_var,
        ),
        "Z_sm": Variable(
            name_or_column_in_file="sm_z",
            original_unit=u.RE,
            time_bin_method=TimeBinMethod.NanMean,
            time_variable=time_var,
        )
    }

    data_path_stem = f"{download_data_dir}arase/YYYY/MM/"
    file_name_stem = "erg_orb_pre_l2_YYYYMMDD_v01.txt"

    orb_source_file = SourceFile(
        download_url=f"https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/swx/orb/{file_name_stem}",
        download_arguments_prefixes="--user erg_sw_users --password arase_swx",
        download_arguments_suffixes=f"-O {data_path_stem}{file_name_stem}",
        download_path=f"{data_path_stem}{file_name_stem}",
        variables_to_extract=orb_variables_to_extract,
        file_cadence=FileCadence.DAILY,
    )

    orb_source_file.download(start_time, end_time)

    # Bernhard: the header is also in the file, but there is a comment after it, so it cannot be read by pd.read_csv
    orb_variables = orb_source_file.extract_variables(start_time, end_time, pd_read_csv_kwargs={})

    convert_string_to_datetime(orb_variables["Epoch"])
    time_var_datetime = time_var.data

    time_var.metadata.unit = u.posixtime
    time_var.data = [t.timestamp() for t in time_var.data]

    # convert SM to GEO
    xSM_arr = np.stack((orb_variables_to_extract["X_sm"].data, orb_variables_to_extract["Y_sm"].data, orb_variables_to_extract["Z_sm"].data)).T
    xGEO_var = Variable(time_variable=time_var, standard_name="xGEO", original_unit=u.RE, time_bin_method=TimeBinMethod.Mean)

    model_coord = Coords(path=irbem_lib_path)

    xGEO_var.data = model_coord.transform(time_var_datetime, xSM_arr, IRBEM_SYSAXIS_SM, IRBEM_SYSAXIS_GEO)
    orb_variables["xGEO"] = xGEO_var

    # delete unused variables
    del orb_variables["X_sm"]
    del orb_variables["Y_sm"]
    del orb_variables["Z_sm"]

    return orb_variables

def _get_mean_energy(E_min:NDArray[np.float64], E_max:NDArray[np.float64]) -> NDArray[np.float64]:

    b = 7.068e-3 # Bernhard: copied from Ingo's scripts. Don't know where this comes from

    weighted_max = (1/b) * np.exp(-b*E_max)
    weighted_min = (1/b) * np.exp(-b*E_min)

    tmp = (weighted_min - weighted_max) / (E_max-E_min)

    return -np.log(tmp) / b