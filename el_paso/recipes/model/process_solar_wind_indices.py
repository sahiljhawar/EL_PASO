# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

import el_paso as ep


def solar_wind_indices_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    *,
    file_format: ep.typing.MFSFormats = "nc",
) -> ep.SavingStrategy:
    """Monthly solar wind indices saving strategy following the GFZ data standard."""
    return ep.saving_strategies.SolarWindIndicesStrategy(
        base_data_path,
        mag_field,
        ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


def process_solar_wind_indices(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["none"] = "none",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz"] = "gfz",
    *,
    skip_existing: bool = True,
) -> None:
    """Save the solar wind and geomagnetic indices required by a magnetic field model over a time range.

    Builds a uniform time grid from `start_time` to `end_time` at `bin_cadence`, loads the
    indices that `mag_field` needs (e.g. Kp for T89, plus Dst, solar wind and IMF parameters for
    TS04), interpolates them onto the grid, and writes them to `processed_data_path` as monthly
    files using the GFZ data standard. The indices are independent of any spacecraft, so this
    recipe takes no satellite. Use it to keep the model inputs on disk without processing particle data.

    Args:
        start_time (datetime): Start of the time range (inclusive).
        end_time (datetime): End of the time range (exclusive).
        satellite (Literal["none"]): Unused. The indices do not belong to any spacecraft; the
                                     parameter only keeps the recipe interface uniform.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model whose required indices
                                                    are saved (e.g. "T89", "TS04").
        raw_data_path (str | Path): Unused. The indices are fetched by the processing step.
        processed_data_path (str | Path): Base directory in which the results are saved.
        bin_cadence (timedelta): Time step between consecutive samples of the time grid.
        num_cores (int): Unused. Loading the indices is not parallelized.
        save_strategy (Literal["gfz"]): The saving strategy used to write the results. The indices
                                        are only written in the GFZ monthly format.
        skip_existing (bool): Unused. The recipe downloads no files.
    """
    del satellite, raw_data_path, num_cores, save_strategy, skip_existing

    timestamps = []
    curr_time = start_time
    while curr_time < end_time:
        timestamps.append(curr_time.timestamp())
        curr_time += bin_cadence

    time_var = ep.Variable(data=np.asarray(timestamps), original_unit=ep.units.posixtime)

    # the indices are merged into the saved variables by `save_sw`, so only the time axis is passed here
    ep.save(
        {"Epoch": time_var},
        solar_wind_indices_strategy(processed_data_path, mag_field),
        start_time,
        end_time,
        time_var,
        save_sw=True,
    )


if __name__ == "__main__":
    ep.run_recipe_cli(process_solar_wind_indices)
