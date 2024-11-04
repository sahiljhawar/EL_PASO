from datetime import datetime, timezone

import numpy as np
from data_management.io.kp import KpOmni

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

    kp_df = KpOmni().read(start_time, end_time, download=True)

    kp_time = kp_df.index
    kp_value = kp_df['kp']

    print(kp_time)
    asdf

    interpolation_function = interp1d(kp_time, kp_value, kind='previous',
                                      bounds_error=False, fill_value="extrapolate")

    # Interpolate the data
    interpolated_data = interpolation_function(target_times)

    kp_data = interpolated_data

    Kp = kp_data

    # Interpolate data to the newtime cadence
    # Dst = interpolate_data(dst_data['Dst'], newtime)
    # Dsw = interpolate_data(ace_data['n_p'], newtime)
    # Vsw = interpolate_data(ace_data['v_sw'], newtime)
    # Pdyn = 1.6726e-6 * Dsw * Vsw ** 2  # Calculate dynamic pressure
    # By = interpolate_data(ace_data['by_gsm'], newtime)
    # Bz = interpolate_data(ace_data['bz_gsm'], newtime)
    # AL = np.full_like(Kp, np.nan)  # Fill AL with NaNs

    # Construct the output array
    maginput_array = np.full((len(time), 25), np.nan)
    maginput_array[:, 0] = Kp
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

    return maginput_array


def get_local_B_field(product, magnetic_field_model):
    pass