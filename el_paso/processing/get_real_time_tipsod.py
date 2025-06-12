from datetime import datetime
from datetime import timezone as tz

import numpy as np
from astropy import units as u
from sscws.sscws import CoordinateSystem, SscWs

import el_paso as ep


def get_real_time_tipsod(timestamps:np.ndarray, sat_name:str, coord_system:str="GEO") -> ep.Variable:
    """
    Return xGEO to be used in the adiabatic invariant calculation.

    Args:
        datetimes (numpy.ndarray): The array of datetimes for which the xGEO values are to be calculated.
                                    Can be a list of datetime objects or strings.
        sat_name (str): The name of the satellite.
        coord_system (str): The coordinate system in which the xGEO values are to be calculated. Default is "GEO".

    Returns:
        numpy.ndarray: The xGEO (x, y, z) values for the given datetimes.
    """

    minimum_datetimes = 2

    if len(timestamps) < minimum_datetimes:
        msg = "At least two datetimes are required."
        raise ValueError(msg)

    datetimes = [datetime.fromtimestamp(t, tz=tz.utc) for t in timestamps]

    def _convert_to_sscws_compatible_time(dt: datetime) -> datetime:
        """Add milliseconds to the datetime objects.

        Args:
            dt (datetime): The datetime to be converted.

        Returns:
            str: The datetime in SSCWS compatible format with milliseconds.

        """
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    original_datetimes = datetimes
    extra_interval = original_datetimes[-1] - original_datetimes[-2]
    datetimes_str = [_convert_to_sscws_compatible_time(dt) for dt in datetimes]

    ssc = SscWs()

    # Retrieve data for the entire time range
    start_time = datetimes_str[0]
    end_time = _convert_to_sscws_compatible_time(original_datetimes[-1] + extra_interval)
    time_range = [start_time, end_time]

    coord_systems_dict = {"GEO": "Geo", "GM": "Gm", "GSE": "Gse", "GSM": "Gsm", "SM": "Sm", "J2000": "GeiJ2000"}
    result = ssc.get_locations([sat_name], time_range, coords=[CoordinateSystem(coord_systems_dict[coord_system])])

    try:
        # Extract X, Y, Z coordinates and corresponding times
        times = np.array(result["Data"][0]["Time"])
        x_coords = np.array([entry["X"] for entry in result["Data"][0]["Coordinates"]]).flatten()
        y_coords = np.array([entry["Y"] for entry in result["Data"][0]["Coordinates"]]).flatten()
        z_coords = np.array([entry["Z"] for entry in result["Data"][0]["Coordinates"]]).flatten()
    except(Exception) as e:
        raise ValueError(str(result)) from e

    # Bin the data according to the given datetimes grid and compute the median of the points in each bin
    all_xyz = []
    for i in range(len(datetimes) - 1):
        bin_mask = (times >= datetimes[i]) & (times < datetimes[i + 1])
        if bin_mask.any():
            x_median = np.median(x_coords[bin_mask])
            y_median = np.median(y_coords[bin_mask])
            z_median = np.median(z_coords[bin_mask])
            all_xyz.append([x_median, y_median, z_median])

    # Handle the last entry with a small extra interval
    last_bin_mask = (times >= datetimes[-1]) & (times <= original_datetimes[-1] + extra_interval)
    if last_bin_mask.any():
        x_median = np.median(x_coords[last_bin_mask])
        y_median = np.median(y_coords[last_bin_mask])
        z_median = np.median(z_coords[last_bin_mask])
        all_xyz.append([x_median, y_median, z_median])

    var = ep.Variable(data=np.asarray(all_xyz), original_unit=u.km)
    var.convert_to_unit(u.RE)

    return var
