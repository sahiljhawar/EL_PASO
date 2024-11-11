import os
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import interp1d
from data_management.io.kp import KpOMNI
from IRBEM import IRBEM
from astropy import units as u

#from el_paso.classes import Product, DerivedVariable

def construct_maginput(time: np.ndarray):
    """
    Constructs the basic magnetospheric input parameters array.

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
    end_time   = datetime.fromtimestamp(time[-1], tz=timezone.utc)

    kp_df = KpOMNI(data_dir=Path(os.getenv('HOME')) / '.el_paso' / 'Kp').read(start_time, end_time, download=True)

    kp_time = [dt.timestamp() for dt in kp_df.index.to_pydatetime()]
    kp_value = kp_df['kp'].values

    interpolation_function = interp1d(kp_time, kp_value, kind='previous',
                                      bounds_error=False, fill_value="extrapolate")

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
    maginput[:, 0] = kp_data*10 # IRBEM takes Kp10 as an input
    '''
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
    '''

    maginput_dict = {"Kp": maginput[:, 0],
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
                     "AL": maginput[:, 16]
    }

    return maginput_dict

def _magnetic_field_str_to_kext(magnetic_field_str):
    match magnetic_field_str:
        case 'T89':
            kext = 4
        case 'T04s':
            kext = 11
        case 'OP77Q':
            kext = 5
    
    return kext

def get_magequator(xGEO, timestamps, irbem_lib_path, irbem_options, magnetic_field_str, maginput):
    
    datetimes   = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = 1
    
    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    
    assert len(datetimes) == len(xGEO)

    kext = _magnetic_field_str_to_kext(magnetic_field_str)

    model = IRBEM.MagFields(path=irbem_lib_path, options=irbem_options, kext=kext, sysaxes=sysaxes, verbose=False)

    magequator_output = {'bmin': np.empty_like(datetimes), 'XGEO': np.empty((len(datetimes), 3))}

    for i in range(len(datetimes)):
        x_dict_single = {"dateTime": datetimes[i], "x1": xGEO[i,0], "x2": xGEO[i,1], "x3": xGEO[i,2]}
        maginput_single = {key: maginput[key][i] for key in maginput.keys()}
        magequator_output_single = model.find_magequator(x_dict_single, maginput_single)
        
        for key in magequator_output_single.keys():
            magequator_output[key][i,...] = magequator_output_single[key]

    # replace bad values with nan
    for key in magequator_output.keys():
        magequator_output[key][magequator_output[key] == fortran_bad_value] = np.nan

    # map irbem output names to standard names and add unit information
    irbem_name_map = {
        'bmin': 'B_eq_' + magnetic_field_str
    }
    magequator_output_mapped = {}
    for key in irbem_name_map.keys():
        magequator_output_mapped[irbem_name_map[key]] = (magequator_output[key].astype(np.float64), u.nT)

    # add total radial distance field 
    magequator_output_mapped['R_eq_' + magnetic_field_str] = (np.linalg.norm(magequator_output['XGEO'], ord=2, axis=1), u.RE)

    return magequator_output_mapped


def get_MLT(xGEO, timestamps, irbem_lib_path, irbem_options, magnetic_field_str):
    
    datetimes   = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = 1
    
    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    
    assert len(datetimes) == len(xGEO)

    kext = _magnetic_field_str_to_kext(magnetic_field_str)

    model = IRBEM.MagFields(path=irbem_lib_path, options=irbem_options, kext=kext, sysaxes=sysaxes, verbose=False)

    mlt_output = np.empty_like(datetimes)

    for i in range(len(datetimes)):
        x_dict = {"dateTime": datetimes[i], "x1": xGEO[i,0], "x2": xGEO[i,1], "x3": xGEO[i,2]}
        mlt_output[i] = model.get_mlt(x_dict)

    unit = u.hour    
    # convert to dict to match other functions
    mlt_output = {'MLT_' + magnetic_field_str: (mlt_output, unit)}

    return mlt_output

def get_local_B_field(xGEO, timestamps, irbem_lib_path, irbem_options, magnetic_field_str, maginput):
    
    datetimes   = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = 1
    
    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in maginput.keys(): maginput[key] = np.array(maginput[key], dtype=np.float64)

    assert len(datetimes) == len(maginput['Kp'])
    assert len(datetimes) == len(xGEO)

    x_dict = {"dateTime": datetimes, "x1": xGEO[:,0], "x2": xGEO[:,1], "x3": xGEO[:,2]}
    kext = _magnetic_field_str_to_kext(magnetic_field_str)

    model = IRBEM.MagFields(path=irbem_lib_path, options=irbem_options, kext=kext, sysaxes=sysaxes, verbose=False)

    field_multi_output = model.get_field_multi(x_dict, maginput)
    
    # replace bad values with nan
    for key in field_multi_output.keys():
        field_multi_output[key][field_multi_output[key] == fortran_bad_value] = np.nan

    # map irbem output names to standard names and add unit information
    irbem_name_map = {
        'Bl': 'B_local_' + magnetic_field_str
    }
    field_multi_output_mapped = {}
    for key in irbem_name_map.keys():
        field_multi_output_mapped[irbem_name_map[key]] = (field_multi_output[key], u.nT)

    return field_multi_output_mapped    

def get_mirror_point(xGEO, timestamps, pa_local, irbem_lib_path, irbem_options, magnetic_field_str, maginput):
    
    datetimes   = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = 1
    
    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in maginput.keys(): maginput[key] = np.array(maginput[key], dtype=np.float64)

    assert len(datetimes) == len(maginput['Kp'])
    assert len(datetimes) == len(xGEO)

    # make sure pa_local does not change in time
    assert np.all(np.repeat(pa_local[0,:][np.newaxis,:], len(datetimes), axis=0) == pa_local)

    kext = _magnetic_field_str_to_kext(magnetic_field_str)

    model = IRBEM.MagFields(path=irbem_lib_path, options=irbem_options, kext=kext, sysaxes=sysaxes, verbose=False)

    for i, pa in enumerate(pa_local[0,:]):

        mirror_point_output = np.empty_like(pa_local)

        for it in range(len(datetimes)):
            x_dict = {"dateTime": datetimes[it], "x1": xGEO[it,0], "x2": xGEO[it,1], "x3": xGEO[it,2]}
            mirror_point_output[it,i] = model.find_mirror_point(x_dict, maginput, pa)['bmin']

    # replace bad values with nan
    mirror_point_output[mirror_point_output < 0] = np.nan

    return {'B_mirr_' + magnetic_field_str: (mirror_point_output, 'nT')}   

def get_Lstar(xGEO, timestamps, pa_local, irbem_lib_path, irbem_options, magnetic_field_str, maginput):
    
    datetimes   = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = 1
    
    # Ensure xGEO and maginput are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    for key in maginput.keys(): maginput[key] = np.array(maginput[key], dtype=np.float64)

    assert len(datetimes) == len(maginput['Kp'])
    assert len(datetimes) == len(xGEO)

    # make sure pa_local does not change in time
    assert np.all(np.repeat(pa_local[0,:][np.newaxis,:], len(datetimes), axis=0) == pa_local)

    x_dict = {"dateTime": datetimes, "x1": xGEO[:,0], "x2": xGEO[:,1], "x3": xGEO[:,2]}
    kext = _magnetic_field_str_to_kext(magnetic_field_str)

    model = IRBEM.MagFields(path=irbem_lib_path, options=irbem_options, kext=kext, sysaxes=sysaxes, verbose=False)

    Lstar_output = {
        'Lm': np.empty_like(pa_local),
        'Lstar': np.empty_like(pa_local),
        'xj': np.empty_like(pa_local)
    }

    for i, pa in enumerate(pa_local[0,:]):
        Lstar_output_single = model.make_lstar_shell_splitting(x_dict, maginput, pa)

        for key in Lstar_output.keys():
            Lstar_output[key][:,i] = Lstar_output_single[key]

    # replace bad values with nan
    for key in Lstar_output.keys():
        Lstar_output[key][Lstar_output[key] < 0] = np.nan

    # map irbem output names to standard names and add unit information
    irbem_name_map = {
        'Lm': 'Lm_' + magnetic_field_str,
        'Lstar': 'Lstar_' + magnetic_field_str,
        'xj': 'XJ_' + magnetic_field_str
    }
    Lstar_output_mapped = {}
    Lstar_output_mapped[irbem_name_map['Lm']] = (Lstar_output['Lm'], '')
    Lstar_output_mapped[irbem_name_map['Lstar']] = (Lstar_output['Lstar'], '')
    Lstar_output_mapped[irbem_name_map['xj']] = (Lstar_output['xj'], '')

    return Lstar_output_mapped    


def lstar_shell_splitting_parallel(args):
    onera_lib_file, options, kext, sysaxes, single_x_dict, single_maginput_dict, alpha_value, r_zero, debug = args
    model = IRBEM.MagFields(path=onera_lib_file, options=options, kext=kext, sysaxes=sysaxes, verbose=True)

    if debug:
        lstar_shell_splitting_output = model.make_lstar_shell_splitting(single_x_dict, single_maginput_dict, alpha_value)
    else:
        # Suppress stdout and stderr
        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                lstar_shell_splitting_output = model.make_lstar_shell_splitting(single_x_dict, single_maginput_dict, alpha_value)

    return lstar_shell_splitting_output


def compute_adiabatic_invariants(
    xGEO,
    datetimes,
    maginput,
    alpha_local,
    onera_lib_file,
    options=[1, 0, 0, 0, 0],
    kext=4,
    sysaxes=1,
    r_zero=1,
    num_cores=4,
    verbose=True,
    debug=False,
):
    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e30)

    # Ensure xGEO and kp are floating-point arrays
    xGEO = np.array(xGEO, dtype=np.float64)
    maginput = np.array(maginput, dtype=np.float64)

    # Replace any nan values with Fortran bad value
    xGEO[np.isnan(xGEO)] = fortran_bad_value
    maginput[np.isnan(maginput)] = fortran_bad_value

    # Initialize arrays to collect results
    n_time = len(datetimes)
    n_alpha = len(alpha_local[0])

    results = {
        "Lm": np.full((n_time, n_alpha), np.nan),
        "lstar": np.full((n_time, n_alpha), np.nan),
        "bmin": np.full((n_time, n_alpha), np.nan),
        "bmirr": np.full((n_time, n_alpha), np.nan),
        "XJ": np.full((n_time, n_alpha), np.nan),
        "blocal": np.full((n_time, 1), np.nan),
        "MLT": np.full((n_time, 1), np.nan),
    }

    # Prepare inputs for parallel processing
    inputs = []
    bfield_inputs = []
    mlt_inputs = []
    for i in range(n_time):
        single_x_dict = {"dateTime": datetimes[i], "x1": xGEO[i, 0], "x2": xGEO[i, 1], "x3": xGEO[i, 2]}
        single_maginput_dict = {"Kp": maginput[0, i],
                                "Dst": maginput[1, i],
                                "dens": maginput[2, i],
                                "velo": maginput[3, i],
                                "Pdyn": maginput[4, i],
                                "ByIMF": maginput[5, i],
                                "BzIMF": maginput[6, i],
                                "G1": maginput[7, i],
                                "G2": maginput[8, i],
                                "G3": maginput[9, i],
                                "W1": maginput[10, i],
                                "W2": maginput[11, i],
                                "W3": maginput[12, i],
                                "W4": maginput[13, i],
                                "W5": maginput[14, i],
                                "W6": maginput[15, i],
                                "AL": maginput[16, i]
                                }
        bfield_inputs.append((onera_lib_file, options, kext, sysaxes, single_x_dict, single_maginput_dict, debug))
        mlt_inputs.append((onera_lib_file, options, kext, sysaxes, single_x_dict, debug))
        for j in range(n_alpha):
            alpha_value = float(alpha_local[i, j])
            inputs.append(
                (onera_lib_file, options, kext, sysaxes, single_x_dict, single_maginput_dict, alpha_value, r_zero, debug)
            )

    # Execute in parallel
    # If not in verbose mode, do a straightforward parallel computing call
    # In verbose mode, do a more complicated call for continuous progress indication
    if not verbose:
        with Pool(processes=num_cores) as pool:
            lstar_results_parallel = pool.map(lstar_shell_splitting_parallel, inputs)
        with Pool(processes=num_cores) as pool:
            bfield_results_parallel = pool.map(get_field_multi_parallel, bfield_inputs)
        with Pool(processes=num_cores) as pool:
            mlt_results_parallel = pool.map(get_mlt_parallel, mlt_inputs)
        with Pool(processes=num_cores) as pool:
            bmirr_results_parallel = pool.map(get_bmirr_parallel, inputs)
    else:

        def update_progress(index, total, interval):
            if index % interval == 0:
                print(f"Progress is at {index}/{total}")

        # Use Manager to keep track of progress
        manager = Manager()
        progress = manager.Value("i", 0)
        lock = manager.Lock()

        def update_progress_callback(total, interval):
            with lock:
                progress.value += 1
            update_progress(progress.value, total, interval)

        with Pool(processes=num_cores) as pool:
            lstar_results_parallel = pool.map_async(
                lstar_shell_splitting_parallel, inputs, callback=update_progress_callback(n_time * n_alpha, 100)
            ).get()
        with Pool(processes=num_cores) as pool:
            bfield_results_parallel = pool.map_async(
                get_field_multi_parallel, bfield_inputs, callback=update_progress_callback(n_time, 10)
            ).get()
        with Pool(processes=num_cores) as pool:
            mlt_results_parallel = pool.map_async(
                get_mlt_parallel, mlt_inputs, callback=update_progress_callback(n_time, 10)
            ).get()
        with Pool(processes=num_cores) as pool:
            bmirr_results_parallel = pool.map_async(
                get_bmirr_parallel, inputs, callback=update_progress_callback(n_time * n_alpha, 100)
            ).get()

    # Collect results from parallel execution
    index = 0
    for i in range(n_time):
        field_multi_output = bfield_results_parallel[i]
        results["blocal"][i] = field_multi_output["Bl"]
        mlt_output = mlt_results_parallel[i]
        results["MLT"][i] = mlt_output
        for j in range(n_alpha):
            lstar_output = lstar_results_parallel[index]
            bmirr_output = bmirr_results_parallel[index]
            results["Lm"][i, j] = lstar_output["Lm"][0]
            results["lstar"][i, j] = lstar_output["Lstar"][0]
            results["bmin"][i, j] = lstar_output["bmin"][0]
            results["bmirr"][i, j] = bmirr_output["bmin"]
            results["XJ"][i, j] = lstar_output["xj"][0]
            index += 1

    return results

