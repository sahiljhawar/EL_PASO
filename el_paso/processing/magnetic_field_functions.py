import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from io import StringIO
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from tqdm import tqdm

import el_paso as ep
from data_management.io.kp import read_kp_from_multiple_models
from el_paso.utils import show_process_bar_for_map_async, timed_function
from IRBEM import Coords, MagFields

logger = logging.getLogger(__name__)

FORTRAN_BAD_VALUE = np.float64(-1.0e31)


def get_W_parameters(timestamps:NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    years = np.unique([dt.year for dt in datetimes])

    if years[-1] > 2023:
        msg = "W parameters are only available until 2023!"
        raise ValueError(msg)

    w_params:dict[str, list[float]] = {"W1":[], "W2":[], "W3":[], "W4":[], "W5":[], "W6":[]}

    for year in years:

        url = f"https://geo.phys.spbu.ru/~tsyganenko/models/ts05/{year:d}_OMNI_5m_with_TS05_variables.dat"

        response = requests.get(url, stream=True, verify=False)

        if response.status_code == 404:
            msg = f"File not found on server: {url}"
            raise FileNotFoundError(msg)

        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), names=["Year", "Day", "Hour", "Min", "W1", "W2", "W3", "W4", "W5", "W6"], usecols=[0,1,2,3,17,18,19,20,21,22], sep="\s+")

        timestamps_data:list[datetime] = []

        for _, row in df.loc[:,["Year","Day","Hour","Min"]].iterrows():
            year = int(row["Year"])
            day  = int(row["Day"])
            hour = int(row["Hour"])
            minute = int(row["Min"])

            timestamps_data.append(datetime.strptime(f"{year:04d}-{day:03d}-{hour:02d}-{minute:02d}", "%Y-%j-%H-%M").replace(tzinfo=timezone.utc).timestamp())

        # find the timestamps for the current year
        timestamp_year_begin = datetime(year,1,1,tzinfo=timezone.utc).timestamp()
        timestamp_year_end = datetime(year,12,31,23,59,59,tzinfo=timezone.utc).timestamp()

        curr_year_idx = (timestamps >= timestamp_year_begin) & (timestamps <= timestamp_year_end)

        for w_str in ["W1", "W2", "W3", "W4", "W5", "W6"]:
            w_data = np.interp(timestamps[curr_year_idx], timestamps_data, df[w_str].values)

            w_params[w_str] += list(w_data)

    return {key: np.asarray(data).astype(np.float64) for key, data in w_params.items()}


def calculate_G1(timestamps:NDArray[np.floating], sw_speed:NDArray[np.float64], IMF_Bz:NDArray[np.float64], IMF_By:NDArray[np.float64]) -> NDArray[np.float64]:

    B_perp = np.sqrt(IMF_Bz**2 + IMF_By**2)
    theta = np.atan2(IMF_By, IMF_Bz)

    G1:list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[:it+1] - curr_time) <= timedelta(hours=1).total_seconds())

        G1.append(float(np.nanmean(sw_speed[idx] * (B_perp[idx]/40)**2/(1+B_perp[idx]/40) * np.sin(theta[idx]/2)**3)))

    return np.asarray(G1)

def calculate_G2(timestamps:NDArray[np.float64], sw_speed:NDArray[np.float64], IMF_Bz:NDArray[np.float64]) -> NDArray[np.float64]:

    Bs = np.where(IMF_Bz < 0, -IMF_Bz, 0)

    G2:list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[:it+1] - curr_time) <= timedelta(hours=1).total_seconds())

        G2.append(float(np.nanmean(sw_speed[idx] * Bs[idx]/200)))

    return np.asarray(G2)

def calculate_G3(timestamps:NDArray[np.float64], sw_speed:NDArray[np.float64], sw_density:NDArray[np.float64], IMF_Bz:NDArray[np.float64]) -> NDArray[np.float64]:

    Bs = np.where(IMF_Bz < 0, -IMF_Bz, 0)

    G3:list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[:it+1] - curr_time) <= timedelta(hours=1).total_seconds())

        G3.append(float(np.nanmean(sw_density[idx] * sw_speed[idx] * Bs[idx]/2000)))

    return np.asarray(G3)

@dataclass
class IrbemInput:
    irbem_lib_path: str
    magnetic_field_str: str
    maginput: dict[str, NDArray[np.float64]]
    irbem_options: list[int]
    num_cores: int = 4

class IrbemOutput(NamedTuple):
    arr: NDArray[np.float64]
    unit: u.UnitBase

def construct_maginput(time_var: ep.Variable,
                       indices_solar_wind: dict[str, ep.Variable]|None=None,
                       magnetic_field_str:str|None=None):
    """Construct the basic magnetospheric input parameters array.

    This function retrieves all solar wind data from the ACE dataset on CDAWeb, as well as the Kp and Dst indices,
    interpolates them to the cadence of `newtime`, and returns an array with the columns as follows:
    1: Kp, value of Kp as in OMNI2 files but as double instead of integer type.
    (NOTE: consistent with OMNI2, this is Kp*10, and it is in the range 0 to 90)
    2: Dst, Dst index (nT)
    3: Dsw, solar wind density (cm-3)
    4: Vsw, solar wind velocity (km/s)
    5: Pdyn, solar wind dynamic pressure (nPa)
    6: By, GSM y component of interplanetary magnetic field (nT)
    7: Bz, GSM z component of interplanetary magnetic field (nT), from ACE
    8-16: Qin-Denton parameters, implement this!
    17: AL auroral index (if not in ACE, fill with NaN)
    18-25: fill with NaN

    Args:
        newtime (array-like): Array of new time points for interpolation.
        sw_path (str, optional): Path to the solar wind data directory.
                                Defaults to environment Variable 'FC_ACE_REALTIME_PROCESSED_DATA_DIR'.
        kp_path (str, optional): Path to the Kp data directory.
                                Defaults to environment Variable 'RT_KP_PROC_DIR'.
        kp_type (str, optional): Type of Kp to read using data_management.
                                Defaults to 'niemegk'.

    Returns:
        np.ndarray: Array of interpolated magnetospheric input parameters.
    """
    time = time_var.get_data(ep.units.posixtime).astype(np.float64)

    if indices_solar_wind is None:

        start_time = datetime.fromtimestamp(time[0], tz=timezone.utc)
        end_time = datetime.fromtimestamp(time[-1], tz=timezone.utc)

        os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso" / "KpOmni")
        os.environ["RT_KP_NIEMEGK_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso" / "KpNiemegk")
        os.environ["KP_ENSEMBLE_OUTPUT_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso")
        os.environ["RT_KP_SWPC_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso")

        kp_df = read_kp_from_multiple_models(start_time, end_time, download=True, synthetic_now_time=datetime.now(tz=timezone.utc))

        kp_time = [dt.timestamp() for dt in kp_df.index.to_pydatetime()]
        kp_value = kp_df["kp"].values

        interpolation_function = interp1d(kp_time, kp_value, kind="previous", bounds_error=False, fill_value="extrapolate")

        # Interpolate the data
        interpolated_data = interpolation_function(time)
        kp_data = interpolated_data

        # Interpolate data to the newtime cadence
        # Dst = interpolate_data(dst_data['Dst'], newtime)
        # Dsw = interpolate_data(ace_data['n_p'], newtime)
        # Vsw = interpolate_data(ace_data['v_sw'], newtime)
        # Pdyn = 1.6726e-6 * Dsw * Vsw ** 2  # Calculate dynamic pressure
        # By = interpolate_data(ace_data['by_gsm'], newtime)
        # Bz = interpolate_data(ace_data['bz_gsm'], newtime)
        # AL = np.full_like(Kp, np.nan)  # Fill AL with NaNs

        # Construct the output array
        maginput = np.full((len(time), 25), np.nan)
        maginput[:, 0] = np.asarray(kp_data * 10, dtype=np.float64)
        # IRBEM takes Kp10 as an input
        """
        1 Kp value of Kp as in OMNI2 files but has to be double instead of integer type. (NOTE, consistent with OMNI2, this is Kp*10, and it is in the range 0 to 90)
        2 Dst Dst index (nT)
        3 Dsw solar wind density (cm-3)
        4 Vsw solar wind velocity (km/s)
        5 Pdyn solar wind dynamic pressure (nPa)
        6 By GSM y component of interplanetary magnetic field (nT)
        7 Bz GSM z component of interplanetary magnetic field (nT)
        8 G1 <Vsw (Bperp/40)2/(1+Bperp/40) sin3(θ/2)> where the <> mean an average over the previous 1 hour, Bperp is the transverse IMF component (GSM) and θ its clock angle
        9 G2 <a Vsw Bs> where Bs=|IMF Bz| when IMF Bz < 0 and Bs=0 when IMF Bz > 0, a=0.005
        10 G3 <Vsw Dsw Bs/2000>
        11-16 W1 W2 W3 W4 W5 W6 see definitions in (Tsyganenko et al., 2005)
        17 AL auroral index
        18-25 reserved for future use (leave as NaN)
        """

    else:

        maginput = np.full((len(time), 25), np.nan)
        if "Kp" in indices_solar_wind:
            kp_data = indices_solar_wind["Kp"].get_data().astype(np.float64)

            if len(kp_data) != len(time):
                msg = f"Encountered size missmatch for Kp: len of Kp data: {len(kp_data)}, requested len: {len(time)}"
                raise ValueError(msg)

            maginput[:, 0] = np.asarray(np.round(kp_data*10), dtype=np.float64)

        if "Dst" in indices_solar_wind:
            dst_data = indices_solar_wind["Dst"].get_data()

            match magnetic_field_str:
                case "T89"|"T01s"|"TS04"|"T04s"|"TS05":
                    pass
                case "T01":
                    dst_data = dst_data.clip(-50, 20)
                case "T96":
                    dst_data = dst_data.clip(-100, 20)
                case _:
                    msg = "Please provide a valid magnetic field string!"
                    raise ValueError(msg)

            if len(dst_data) != len(time):
                msg = f"Encountered size missmatch for Kp: len of Kp data: {len(dst_data)}, requested len: {len(time)}"
                raise ValueError(msg)

            maginput[:, 1] = np.asarray(dst_data, dtype=np.float64)

        if "Pdyn" in indices_solar_wind:
            pdyn_data = indices_solar_wind["Pdyn"].get_data()

            match magnetic_field_str:
                case "T89"|"T01s"|"TS04"|"T04s"|"TS05":
                    pass
                case "T01":
                    pdyn_data = pdyn_data.clip(0.5, 5)
                case "T96":
                    pdyn_data = pdyn_data.clip(0.5, 10)
                case _:
                    msg = "Please provide a valid magnetic field string!"
                    raise ValueError(msg)

            if len(pdyn_data) != len(time):
                msg = f"Encountered size missmatch for Kp: len of Kp data: {len(pdyn_data)}, requested len: {len(time)}"
                raise ValueError(msg)

            maginput[:, 4] = np.asarray(pdyn_data, dtype=np.float64)

        if "IMF_By" in indices_solar_wind:
            imf_by_data = indices_solar_wind["IMF_By"].get_data()

            match magnetic_field_str:
                case "T89"|"T01s"|"TS04"|"T04s"|"TS05":
                    pass
                case "T01":
                    imf_by_data = imf_by_data.clip(-5, 5)
                case "T96":
                    imf_by_data = imf_by_data.clip(-10, 10)
                case _:
                    msg = "Please provide a valid magnetic field string!"
                    raise ValueError(msg)

            if len(imf_by_data) != len(time):
                msg = f"Encountered size missmatch for Kp: len of Kp data: {len(imf_by_data)}, requested len: {len(time)}"
                raise ValueError(msg)

            maginput[:, 5] = np.asarray(np.abs(imf_by_data), dtype=np.float64)

        if "IMF_Bz" in indices_solar_wind:
            imf_bz_data = indices_solar_wind["IMF_Bz"].get_data()

            match magnetic_field_str:
                case "T89"|"T01s"|"TS04"|"T04s"|"TS05":
                    pass
                case "T01":
                    imf_bz_data = imf_bz_data.clip(-5, 5)
                case "T96":
                    imf_bz_data = imf_bz_data.clip(-10, 10)
                case _:
                    msg = "Please provide a valid magnetic field string!"
                    raise ValueError(msg)

            if len(imf_bz_data) != len(time):
                msg = f"Encountered size missmatch for Kp: len of Kp data: {len(imf_bz_data)}, requested len: {len(time)}"
                raise ValueError(msg)

            maginput[:, 6] = np.asarray(np.abs(imf_bz_data), dtype=np.float64)


        if magnetic_field_str in ["T01", "T01s"]:

            sw_speed = indices_solar_wind["SW_speed"].get_data().astype(np.float64)
            sw_density = indices_solar_wind["SW_density"].get_data().astype(np.float64)
            imf_bz = indices_solar_wind["IMF_Bz"].get_data().astype(np.float64)
            imf_by = indices_solar_wind["IMF_By"].get_data().astype(np.float64)

            g2 = calculate_G2(time, sw_speed, imf_bz)

            if magnetic_field_str == "T01":
                g1 = calculate_G1(time, sw_speed, imf_bz, imf_by)
                g1 = g1.clip(0, 10)

                g2 = g2.clip(0, 10)

                maginput[:, 7] = g1

            maginput[:, 8] = g2

            if magnetic_field_str == "T01s":
                g3 = calculate_G3(time, sw_speed, sw_density, imf_bz)
                maginput[:, 9] = g3

    if magnetic_field_str in ["TS04", "T04s", "TS05"]:

        w_params = get_W_parameters(time)

        maginput[:,10] = w_params["W1"]
        maginput[:,11] = w_params["W2"]
        maginput[:,12] = w_params["W3"]
        maginput[:,13] = w_params["W4"]
        maginput[:,14] = w_params["W5"]
        maginput[:,15] = w_params["W6"]

    maginput_dict = {
        "Kp": maginput[:, 0],
        "Dst": maginput[:, 1],
        "dens": maginput[:, 2],
        "velo": maginput[:, 3],
        "Pdyn": maginput[:, 4],
        "ByIMF": maginput[:, 5],
        "BzIMF": maginput[:, 6],
        "G1": maginput[:, 7],
        "G2": maginput[:, 8],
        "G3": maginput[:, 9],
        "W1": maginput[:, 10],
        "W2": maginput[:, 11],
        "W3": maginput[:, 12],
        "W4": maginput[:, 13],
        "W5": maginput[:, 14],
        "W6": maginput[:, 15],
        "AL": maginput[:, 16],
    }

    return maginput_dict


def _magnetic_field_str_to_kext(magnetic_field_str:str) -> int:
    match magnetic_field_str:
        case "T89":
            kext = 4
        case "T01":
            kext = 9
        case "T01s":
            kext = 10
        case "TS04"|"TS05"|"T04s":
            kext = 11
        case "T96":
            kext = 7
        case "OP77Q":
            kext = 5
        case _:
            msg = "Invalid magnetic field model!"
            raise ValueError(msg)

    return kext


def _get_magequator_parallel(irbem_args: tuple[str,list[int],int,int],
                             x_geo: NDArray[np.float64],
                             datetimes: list[datetime],
                             maginput: dict[str,NDArray[np.float64]],
                             it: int) -> tuple[float, NDArray[np.float64]]:
    model = MagFields(
        path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3], verbose=False
    )

    x_dict_single:dict[str,datetime|float] = {"dateTime": datetimes[it],
                                              "x1": x_geo[it, 0],
                                              "x2": x_geo[it, 1],
                                              "x3": x_geo[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput}

    magequator_output = model.find_magequator(x_dict_single, maginput_single)

    return magequator_output["bmin"], magequator_output["XGEO"]


@timed_function()
def get_magequator(xgeo_var: ep.Variable,
                   time_var: ep.Variable,
                   irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating magnetic field and radial distance at the equator ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_magequator_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = len(datetimes)//irbem_input.num_cores // 4 # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    B_eq = np.empty_like(datetimes)
    x_geo_min = np.empty_like(x_geo)

    results = rs.get()

    for i in range(len(datetimes)):
        B_eq[i] = results[i][0]
        x_geo_min[i] = results[i][1]

    # for i in range(len(datetimes)):
    #     if isinstance(rs._value, Exception):
    #         print(rs._value)
    #         continue

    #     B_eq[i] = rs._value[i][0]
    #     xGEO_min[i, :] = rs._value[i][1]

    B_eq[B_eq == FORTRAN_BAD_VALUE] = np.nan
    x_geo_min[x_geo_min == FORTRAN_BAD_VALUE] = np.nan

    B_eq_var = ep.Variable(data=B_eq.astype(np.float64),original_unit=u.nT)
    B_eq_var.metadata.add_processing_note(
        f"Calculated magnetic field at the equator using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    x_geo_var = ep.Variable(data=x_geo_min.astype(np.float64), original_unit=ep.units.RE)
    x_geo_var.metadata.add_processing_note(
        f"Calculated radial distance at the equator using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    # add radial distance field in SM coordinates
    x_sm = Coords(path=irbem_input.irbem_lib_path).transform(datetimes,  # type: ignore[reportUnknownMemberType]
                                                             x_geo_min,
                                                             ep.IRBEM_SYSAXIS_GEO,
                                                             ep.IRBEM_SYSAXIS_SM)

    R_eq_var = ep.Variable(data=np.linalg.norm(x_sm, ord=2, axis=1).astype(np.float64),
                           original_unit=ep.units.RE)
    R_eq_var.metadata.add_processing_note(
        f"Calculated radial distance at the equator in SM coordinates using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    return {"B_eq_" + irbem_input.magnetic_field_str: B_eq_var,
            "R_eq_" + irbem_input.magnetic_field_str: R_eq_var,
            "xGEO_eq_" + irbem_input.magnetic_field_str: x_geo_var}

def _get_footpoint_atmosphere_parallel(irbem_args: tuple[str,list[int],int,int],
                                       x_geo: NDArray[np.float64],
                                       datetimes: list[datetime],
                                       maginput: dict[str,NDArray[np.float64]],
                                       it: int) -> list[float]:
    model = MagFields(
        path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3], verbose=False,
    )

    x_dict_single:dict[str,datetime|float] = {"dateTime": datetimes[it],
                                              "x1": x_geo[it, 0],
                                              "x2": x_geo[it, 1],
                                              "x3": x_geo[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput}

    footpoint_output = model.find_foot_point(x_dict_single, maginput_single, stopAlt=100, hemiFlag=0)

    return footpoint_output["BFOOTMAG"]

@timed_function()
def get_footpoint_atmosphere(xgeo_var: ep.Variable,
                             time_var: ep.Variable,
                             irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating magnetic foot point at the atmosphere ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_footpoint_atmosphere_parallel,
                            irbem_args,
                            x_geo,
                            datetimes,
                            irbem_input.maginput)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = len(datetimes)//irbem_input.num_cores // 4 # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    B_foot = np.empty_like(datetimes)  # noqa: N806

    results = rs.get()

    for i in range(len(datetimes)):
        B_foot[i] = results[i][0]


    # for i in range(len(datetimes)):
    #     if isinstance(rs._value, Exception):
    #         print(rs._value)
    #         continue

    #     B_foot[i] = rs._value[i][0]

    B_foot[B_foot == FORTRAN_BAD_VALUE] = np.nan

    var = ep.Variable(data=B_foot.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Calculated foot point at the atmosphere using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    return {"B_fofl_" + irbem_input.magnetic_field_str: var}

@timed_function()
def get_MLT(xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating magnetic local time ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    # Ensure xGEO and maginput are floating-point arrays
    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    model = MagFields(path=irbem_input.irbem_lib_path,
                      options=irbem_input.irbem_options,
                      kext=kext,
                      sysaxes=sysaxes,
                      verbose=False)

    mlt_output = np.empty_like(datetimes)

    for i in range(len(datetimes)):
        x_dict:dict[str,datetime|float] = {"dateTime": datetimes[i],
                                           "x1": x_geo[i, 0],
                                           "x2": x_geo[i, 1],
                                           "x3": x_geo[i, 2]}
        mlt_output[i] = model.get_mlt(x_dict)

    mlt_output = mlt_output.astype(np.float64)

    var = ep.Variable(data=mlt_output, original_unit=u.hour)
    var.metadata.add_processing_note(
        f"Calculated MLT using IRBEM model {irbem_input.magnetic_field_str} with options {irbem_input.irbem_options}."
    )

    return {"MLT_" + irbem_input.magnetic_field_str: var}


@timed_function()
def get_local_B_field(xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating local magnetic field values ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure x_geo and maginput are floating-point arrays
    x_geo = x_geo.astype(np.float64)
    for key in irbem_input.maginput:
        irbem_input.maginput[key] = np.array(irbem_input.maginput[key], dtype=np.float64)

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    x_dict:dict[str,NDArray[np.float64]|list[datetime]] = {"dateTime": datetimes,
                                                           "x1": x_geo[:, 0],
                                                           "x2": x_geo[:, 1],
                                                           "x3": x_geo[:, 2]}
    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    model = MagFields(
        path=irbem_input.irbem_lib_path, options=irbem_input.irbem_options, kext=kext, sysaxes=sysaxes, verbose=False
    )

    field_multi_output = model.get_field_multi(x_dict, irbem_input.maginput)

    # replace bad values with nan
    for key in field_multi_output:
        field_multi_output[key][field_multi_output[key] == fortran_bad_value] = np.nan

    return {"B_local_" + irbem_input.magnetic_field_str: ep.Variable(data=field_multi_output["Bl"],
                                                                     original_unit=u.nT)}



def _get_mirror_point_parallel(irbem_args:tuple[str,list[int],int,int],
                               x_geo:NDArray[np.float64],
                               datetimes:list[datetime],
                               maginput:dict[str,NDArray[np.float64]],
                               pa_local:NDArray[np.float64],
                               it:int):
    model = MagFields(path=irbem_args[0],
                      options=irbem_args[1],
                      kext=irbem_args[2],
                      sysaxes=irbem_args[3],
                      verbose=False)

    x_dict_single:dict[str,datetime|float] = {"dateTime": datetimes[it],
                                              "x1": x_geo[it, 0],
                                              "x2": x_geo[it, 1],
                                              "x3": x_geo[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput}

    mirror_point_output = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        mirror_point_output[i] = model.find_mirror_point(x_dict_single, maginput_single, pa)["bmin"]

    return mirror_point_output.astype(np.float64)


@timed_function()
def get_mirror_point(xgeo_var: ep.Variable,
                     time_var: ep.Variable,
                     pa_local_var: ep.Variable,
                     irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating mirror points ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)
    pa_local = pa_local_var.get_data(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)
    pa_local = pa_local.astype(np.float64)
    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = (f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)
    if len(datetimes) != len(pa_local):
        msg = (f"Encountered size mismatch for pa_local: len of pa_local data: {len(pa_local)}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_mirror_point_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput, pa_local)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = len(datetimes)//irbem_input.num_cores // 4 # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    mirror_point_output = np.empty_like(pa_local)

    results = rs.get()

    for i in range(len(datetimes)):
        mirror_point_output[i, :] = results[i]

    # for i in range(len(datetimes)):
    #     if isinstance(rs._value, Exception):
    #         print(rs._value)
    #         continue

    #     mirror_point_output[i, :] = rs._value[i]

    # replace bad values with nan
    mirror_point_output[mirror_point_output < 0] = np.nan

    var = ep.Variable(data=mirror_point_output.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Calculated mirror points using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    return {"B_mirr_" + irbem_input.magnetic_field_str: var}


def _make_lstar_shell_splitting_parallel(irbem_args: tuple[str,list[int],int,int],
                                         x_geo: NDArray[np.float64],
                                         datetimes: list[datetime],
                                         maginput: dict[str,NDArray[np.float64]],
                                         pa_local: NDArray[np.float64],
                                         it: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

    model = MagFields(path=irbem_args[0],
                      options=irbem_args[1],
                      kext=irbem_args[2],
                      sysaxes=irbem_args[3],
                      verbose=False)

    x_dict_single:dict[str,datetime|float] = {"dateTime": datetimes[it],
                                              "x1": x_geo[it, 0],
                                              "x2": x_geo[it, 1],
                                              "x3": x_geo[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput}

    Lm = np.empty_like(pa_local[it, :])  # noqa: N806
    Lstar = np.empty_like(pa_local[it, :])  # noqa: N806
    xj = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        Lstar_output_single = model.make_lstar_shell_splitting(x_dict_single, maginput_single, pa)

        Lm[i] = Lstar_output_single["Lm"][0]
        Lstar[i] = Lstar_output_single["Lstar"][0]
        xj[i] = Lstar_output_single["xj"][0]

    return (Lm.astype(np.float64), Lstar.astype(np.float64), xj.astype(np.float64))


@timed_function()
def get_Lstar(xgeo_var: ep.Variable,
              time_var: ep.Variable,
              pa_local_var: ep.Variable,
              irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    logger.info("\tCalculating Lstar and J ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)
    pa_local = pa_local_var.get_data(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)
    pa_local = pa_local.astype(np.float64)
    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = (f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)
    if len(datetimes) != len(pa_local):
        msg = (f"Encountered size mismatch for pa_local: len of pa_local data: {len(pa_local)}, "
               f"requested len: {len(datetimes)}")
        raise ValueError(msg)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(
        _make_lstar_shell_splitting_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput, pa_local
    )

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = len(datetimes)//irbem_input.num_cores // 4 # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    Lm = np.empty_like(pa_local)  # noqa: N806
    Lstar = np.empty_like(pa_local)  # noqa: N806
    xj = np.empty_like(pa_local)

    results = rs.get()

    for i in range(len(datetimes)):
        Lm[i, :] = results[i][0]
        Lstar[i, :] = results[i][1]
        xj[i, :] = results[i][2]
    # for i in range(len(datetimes)):
    #     if isinstance(rs._value, Exception):
    #         print(rs._value)
    #         continue

    #     Lm[i, :] = rs._value[i][0]
    #     Lstar[i, :] = rs._value[i][1]
    #     xj[i, :] = rs._value[i][2]

    # replace bad values with nan
    for arr in [Lm, Lstar, xj]:
        arr[arr < 0] = np.nan
        if not np.any(np.isfinite(arr)):
            msg = "Lstar calculation failed! All NaNs!"
            raise ValueError(msg)

    Lm_var = ep.Variable(data=Lm.astype(np.float64), original_unit=u.dimensionless_unscaled)  # noqa: N806
    Lm_var.metadata.add_processing_note(
        f"Calculated Lm using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    Lstar_var = ep.Variable(data=Lstar.astype(np.float64), original_unit=u.dimensionless_unscaled)  # noqa: N806
    Lstar_var.metadata.add_processing_note(
        f"Calculated Lstar using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    XJ_var = ep.Variable(data=xj.astype(np.float64), original_unit=ep.units.RE)  # noqa: N806
    XJ_var.metadata.add_processing_note(
        f"Calculated XJ using IRBEM model {irbem_input.magnetic_field_str} "
        f"with options {irbem_input.irbem_options}.")

    return {"Lm_" + irbem_input.magnetic_field_str: Lm_var,
            "Lstar_" + irbem_input.magnetic_field_str: Lstar_var,
            "XJ_" + irbem_input.magnetic_field_str: XJ_var}

