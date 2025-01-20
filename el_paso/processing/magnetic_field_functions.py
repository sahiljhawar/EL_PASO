import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
from tqdm import tqdm

from data_management.io.kp import read_kp_from_multiple_models
from el_paso import IRBEM_SYSAXIS_GEO, IRBEM_SYSAXIS_SM
from el_paso.classes import TimeVariable, Variable
from el_paso.utils import timed_function
from IRBEM import Coords, MagFields


@dataclass
class IrbemInput:
    irbem_lib_path: str
    magnetic_field_str: str
    maginput: dict[str, np.ndarray]
    irbem_options: list
    num_cores: int = 4


def construct_maginput(time: np.ndarray):
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
                                Defaults to environment variable 'FC_ACE_REALTIME_PROCESSED_DATA_DIR'.
        kp_path (str, optional): Path to the Kp data directory.
                                Defaults to environment variable 'RT_KP_PROC_DIR'.
        kp_type (str, optional): Type of Kp to read using data_management.
                                Defaults to 'niemegk'.

    Returns:
        np.ndarray: Array of interpolated magnetospheric input parameters.
    """
    start_time = datetime.fromtimestamp(time[0], tz=timezone.utc)
    end_time = datetime.fromtimestamp(time[-1], tz=timezone.utc)

    os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso" / "KpOmni")
    os.environ["RT_KP_NIEMEGK_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso" / "KpNiemegk")
    os.environ["KP_ENSEMBLE_OUTPUT_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso")
    os.environ["RT_KP_SWPC_STREAM_DIR"] = str(Path(os.getenv("HOME")) / ".el_paso")

    kp_df = read_kp_from_multiple_models(start_time, end_time, download=True)

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


def _magnetic_field_str_to_kext(magnetic_field_str):
    match magnetic_field_str:
        case "T89":
            kext = 4
        case "T04s":
            kext = 11
        case "OP77Q":
            kext = 5

    return kext


def _get_magequator_parallel(irbem_args: list, xGEO: np.ndarray, datetimes: np.ndarray, maginput: dict, it: int):
    model = MagFields(
        path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3], verbose=False
    )

    x_dict_single = {"dateTime": datetimes[it], "x1": xGEO[it, 0], "x2": xGEO[it, 1], "x3": xGEO[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput.keys()}

    magequator_output = model.find_magequator(x_dict_single, maginput_single)

    return magequator_output["bmin"], magequator_output["XGEO"]


@timed_function()
def get_magequator(xgeo_var: Variable, time_var: TimeVariable, irbem_input: IrbemInput):
    logging.info("\tCalculating magnetic field and radial distance at the equator ...")

    timestamps = (time_var.data * time_var.metadata.unit).to_value(u.posixtime)
    xGEO = (xgeo_var.data * xgeo_var.metadata.unit).to_value(u.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = IRBEM_SYSAXIS_GEO

    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)

    assert len(datetimes) == len(xGEO)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = [irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes]

    parallel_func = partial(_get_magequator_parallel, irbem_args, xGEO, datetimes, irbem_input.maginput)

    with Pool(processes=irbem_input.num_cores) as pool:
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=1)
        init = rs._number_left
        with tqdm(total=init) as t:
            while True:
                if rs.ready():
                    break
                t.n = init - rs._number_left
                t.refresh()
                time.sleep(1)

    # write async results into one array
    B_eq = np.empty_like(datetimes)
    xGEO_min = np.empty_like(xGEO)
    for i in range(len(datetimes)):
        if isinstance(rs._value, Exception):
            print(rs._value)
            continue

        B_eq[i] = rs._value[i][0]
        xGEO_min[i, :] = rs._value[i][1]

    B_eq[B_eq == fortran_bad_value] = np.nan
    xGEO_min[xGEO_min == fortran_bad_value] = np.nan

    magequator_output = {}
    magequator_output["B_eq_" + irbem_input.magnetic_field_str] = (B_eq.astype(np.float64), u.nT)

    # add total radial distance field in SM coordinates
    xSM = Coords(path=irbem_input.irbem_lib_path).transform(
        datetimes, xGEO_min, IRBEM_SYSAXIS_GEO, IRBEM_SYSAXIS_SM
    )
    magequator_output["R_eq_" + irbem_input.magnetic_field_str] = (
        np.linalg.norm(xSM, ord=2, axis=1).astype(np.float64),
        u.RE,
    )

    return magequator_output


@timed_function()
def get_MLT(
    xgeo_var: Variable, time_var: TimeVariable, irbem_input: IrbemInput
) -> dict[str, tuple[np.ndarray, u.UnitBase]]:
    logging.info("\tCalculating magnetic local time ...")

    timestamps = (time_var.data * time_var.metadata.unit).to_value(u.posixtime)
    xGEO = (xgeo_var.data * xgeo_var.metadata.unit).to_value(u.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = IRBEM_SYSAXIS_GEO

    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)

    assert len(datetimes) == len(xGEO)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    model = MagFields(
        path=irbem_input.irbem_lib_path, options=irbem_input.irbem_options, kext=kext, sysaxes=sysaxes, verbose=False
    )

    mlt_output = np.empty_like(datetimes)

    for i in range(len(datetimes)):
        x_dict = {"dateTime": datetimes[i], "x1": xGEO[i, 0], "x2": xGEO[i, 1], "x3": xGEO[i, 2]}
        mlt_output[i] = model.get_mlt(x_dict)

    mlt_output = mlt_output.astype(np.float64)

    unit = u.hour
    # convert to dict to match other functions
    mlt_output = {"MLT_" + irbem_input.magnetic_field_str: (mlt_output, unit)}

    return mlt_output


@timed_function()
def get_local_B_field(xgeo_var: Variable, time_var: Variable, irbem_input: IrbemInput):
    logging.info("\tCalculating local magnetic field values ...")

    timestamps = (time_var.data * time_var.metadata.unit).to_value(u.posixtime)
    xGEO = (xgeo_var.data * xgeo_var.metadata.unit).to_value(u.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = IRBEM_SYSAXIS_GEO

    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in irbem_input.maginput.keys():
        irbem_input.maginput[key] = np.array(irbem_input.maginput[key], dtype=np.float64)

    assert len(datetimes) == len(irbem_input.maginput["Kp"])
    assert len(datetimes) == len(xGEO)

    x_dict = {"dateTime": datetimes, "x1": xGEO[:, 0], "x2": xGEO[:, 1], "x3": xGEO[:, 2]}
    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    model = MagFields(
        path=irbem_input.irbem_lib_path, options=irbem_input.irbem_options, kext=kext, sysaxes=sysaxes, verbose=False
    )

    field_multi_output = model.get_field_multi(x_dict, irbem_input.maginput)

    # replace bad values with nan
    for key in field_multi_output.keys():
        field_multi_output[key][field_multi_output[key] == fortran_bad_value] = np.nan

    # map irbem output names to standard names and add unit information
    irbem_name_map = {"Bl": "B_local_" + irbem_input.magnetic_field_str}
    field_multi_output_mapped = {}
    for key in irbem_name_map.keys():
        field_multi_output_mapped[irbem_name_map[key]] = (field_multi_output[key], u.nT)

    return field_multi_output_mapped


def _get_mirror_point_parallel(irbem_args, xGEO, datetimes, maginput, pa_local, it):
    model = MagFields(
        path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3], verbose=False
    )

    x_dict_single = {"dateTime": datetimes[it], "x1": xGEO[it, 0], "x2": xGEO[it, 1], "x3": xGEO[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput.keys()}

    mirror_point_output = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        mirror_point_output[i] = model.find_mirror_point(x_dict_single, maginput_single, pa)["bmin"]

    return mirror_point_output.astype(np.float64)


@timed_function()
def get_mirror_point(xgeo_var: Variable, time_var: Variable, pa_local_var: Variable, irbem_input: IrbemInput):
    logging.info("\tCalculating mirror points ...")

    timestamps = (time_var.data * time_var.metadata.unit).to_value(u.posixtime)
    xGEO = (xgeo_var.data * xgeo_var.metadata.unit).to_value(u.RE)
    pa_local = (pa_local_var.data * pa_local_var.metadata.unit).to_value(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = IRBEM_SYSAXIS_GEO

    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in irbem_input.maginput.keys():
        irbem_input.maginput[key] = np.array(irbem_input.maginput[key], dtype=np.float64)

    assert len(datetimes) == len(irbem_input.maginput["Kp"])
    assert len(datetimes) == len(xGEO)
    assert len(datetimes) == len(pa_local)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = [irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes]

    parallel_func = partial(_get_mirror_point_parallel, irbem_args, xGEO, datetimes, irbem_input.maginput, pa_local)

    with Pool(processes=irbem_input.num_cores) as pool:
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=1)
        init = rs._number_left
        with tqdm(total=init) as t:
            while True:
                if rs.ready():
                    break
                t.n = init - rs._number_left
                t.refresh()
                time.sleep(1)

    # write async results into one array
    mirror_point_output = np.empty_like(pa_local)
    for i in range(len(datetimes)):
        if isinstance(rs._value, Exception):
            print(rs._value)
            continue

        mirror_point_output[i, :] = rs._value[i]

    # replace bad values with nan
    mirror_point_output[mirror_point_output < 0] = np.nan

    return {"B_mirr_" + irbem_input.magnetic_field_str: (mirror_point_output, u.nT)}


def _make_lstar_shell_splitting_parallel(
    irbem_args: list, xGEO: np.ndarray, datetimes: list[datetime], maginput: dict, pa_local: np.ndarray, it: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = MagFields(
        path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3], verbose=False
    )

    x_dict_single = {"dateTime": datetimes[it], "x1": xGEO[it, 0], "x2": xGEO[it, 1], "x3": xGEO[it, 2]}
    maginput_single = {key: maginput[key][it] for key in maginput}

    Lm = np.empty_like(pa_local[it, :])
    Lstar = np.empty_like(pa_local[it, :])
    xj = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        Lstar_output_single = model.make_lstar_shell_splitting(x_dict_single, maginput_single, pa)

        Lm[i] = Lstar_output_single["Lm"][0]
        Lstar[i] = Lstar_output_single["Lstar"][0]
        xj[i] = Lstar_output_single["xj"][0]

    return (Lm.astype(np.float64), Lstar.astype(np.float64), xj.astype(np.float64))


@timed_function()
def get_Lstar(xgeo_var: Variable, time_var: Variable, pa_local_var: Variable, irbem_input: IrbemInput):
    logging.info("\tCalculating Lstar and J ...")

    timestamps = (time_var.data * time_var.metadata.unit).to_value(u.posixtime)
    xGEO = (xgeo_var.data * xgeo_var.metadata.unit).to_value(u.RE)
    pa_local = (pa_local_var.data * pa_local_var.metadata.unit).to_value(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = IRBEM_SYSAXIS_GEO

    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in irbem_input.maginput:
        irbem_input.maginput[key] = np.array(irbem_input.maginput[key], dtype=np.float64)

    assert len(datetimes) == len(irbem_input.maginput["Kp"])
    assert len(datetimes) == len(xGEO)
    assert len(datetimes) == len(pa_local)

    kext = _magnetic_field_str_to_kext(irbem_input.magnetic_field_str)

    irbem_args = [irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes]

    parallel_func = partial(
        _make_lstar_shell_splitting_parallel, irbem_args, xGEO, datetimes, irbem_input.maginput, pa_local
    )

    with Pool(processes=irbem_input.num_cores) as pool:
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=1)
        init = rs._number_left
        with tqdm(total=init) as t:
            while True:
                if rs.ready():
                    break
                t.n = init - rs._number_left
                t.refresh()
                time.sleep(1)

    # write async results into one array
    Lm = np.empty_like(pa_local)
    Lstar = np.empty_like(pa_local)
    xj = np.empty_like(pa_local)
    for i in range(len(datetimes)):
        if isinstance(rs._value, Exception):
            print(rs._value)
            continue

        Lm[i, :] = rs._value[i][0]
        Lstar[i, :] = rs._value[i][1]
        xj[i, :] = rs._value[i][2]

    # replace bad values with nan
    for arr in [Lm, Lstar, xj]:
        arr[arr < 0] = np.nan
        if not np.any(np.isfinite(arr)):
            raise ValueError("Lstar calculation failed! All NaNs!")

    Lstar_output_mapped = {}
    Lstar_output_mapped["Lm_" + irbem_input.magnetic_field_str] = (Lm, "")
    Lstar_output_mapped["Lstar_" + irbem_input.magnetic_field_str] = (Lstar, "")
    Lstar_output_mapped["XJ_" + irbem_input.magnetic_field_str] = (xj, "")

    return Lstar_output_mapped
