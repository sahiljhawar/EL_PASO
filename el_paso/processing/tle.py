# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skyfield.api import EarthSatellite, load

logger = logging.getLogger(__name__)


def calculate_geo_coords(
    tle_filename: str | Path,
) -> tuple[str, list[datetime], NDArray[np.float64]]:
    """Calculate GEO coordinates (x, y, z) in kilometers from a TLE file.

    Args:
        tle_filename (str | Path): The file path containing the TLE data.

    Returns:
        tuple[str, list[datetime], NDArray[np.float64]]: A tuple of:
            - satellite_name: The name of the satellite.
            - tle_times: UTC datetime for each TLE epoch.
            - geo_coordinates: Shape ``(n, 3)`` array of ``(x, y, z)`` in kilometers.
    """
    lines = Path(tle_filename).read_text().splitlines()
    tle_data = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines), 2)]
    satellite_name = tle_data[0][0].split()[1]

    timescale = load.timescale()
    tle_times = []
    geo_coordinates = []

    for line1, line2 in tle_data:
        year = int(line1.split()[3][:2])
        doy = float(line1.split()[3][2:])
        tle_time = datetime(2000 + year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy)

        satellite = EarthSatellite(line1, line2, satellite_name)
        geocentric = satellite.at(timescale.from_datetime(tle_time))

        tle_times.append(tle_time)
        geo_coordinates.append(geocentric.xyz.km)

    result = np.asarray(geo_coordinates, dtype=np.float64)

    if np.isnan(result).any():
        nan_indices = np.where(np.isnan(result).any(axis=1))[0]
        logger.warning(
            f"NaN values found in GEO coordinates at indices: {', '.join(str(i) for i in nan_indices)}. "
            "Check the TLE file at these indices."
        )

    return satellite_name, tle_times, result
