import logging
import typing
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import el_paso as ep
from el_paso.utils import datenum_to_datetime, timed_function


@timed_function()
def bin_by_time(
    time_variable: ep.Variable,
    variables: dict[str, ep.Variable],
    time_bin_method_dict: dict[str, ep.TimeBinMethod],
    time_binning_cadence: timedelta,
    window_alignement: Literal["center", "left", "right"] = "center",
    start_time: datetime|None=None,
    end_time: datetime|None=None,
) -> ep.Variable:

    logger = logging.getLogger(__name__)
    logger.info("Binning by time...")

    start_time = start_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[0])
    end_time   = end_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[-1])

    binned_time, time_bins = _create_binned_time_and_bins(start_time, end_time, time_binning_cadence, window_alignement)

    # Cache digitized indices for every time variable
    index_iterables = None

    for key, var in variables.items():

        if key not in time_bin_method_dict:
            continue

        # Just repeat in case of no time dependency
        if time_bin_method_dict[key] == ep.TimeBinMethod.Repeat:
            var.set_data(np.repeat(var.get_data()[np.newaxis, ...], len(binned_time), axis=0), "same")
            continue

        # check if time variable and data content sizes match

        if var.get_data().shape[0] != len(time_variable.get_data()):
            msg = f"Variable {key}: size of dimension 0 does not match length of time variable!"
            raise ValueError(msg)

        # calculate bin indices for given time array if it has not been calculated before
        if not index_iterables:
            timestamps = typing.cast("NDArray[np.floating]", time_variable.get_data(ep.units.posixtime))
            index_iterables = _calculate_index_iterables(timestamps, time_bins)

        unique_indices, indices_separation = index_iterables

        # Initialize binned_data as an array of np.nans with the same shape as self._data,
        # but with the length of the first dimension matching the length of time_array
        if var.get_data().dtype.kind in {"U", "S", "O"}:  # Check if the data is string or object type
            binned_data = np.full((len(binned_time),), "", dtype=var.get_data().dtype)
        else:
            binned_data_shape = (len(binned_time),) + var.get_data().shape[1:]
            binned_data = np.full(binned_data_shape, np.nan)

        # Iterate over unique indices
        for i, unique_index in enumerate(unique_indices):
            bin_data = var.get_data()[indices_separation[i] : indices_separation[i + 1]]
            if len(bin_data) == 0 or not np.any(np.isfinite(bin_data)):
                continue  # no data found
            binned_value = time_bin_method_dict[key](bin_data)

            # Update the relevant slice of binned_data
            binned_data[unique_index, ...] = binned_value

        # Update relevant metadata fields
        # Ensure binned_data works for both numeric and string data
        if isinstance(binned_data[0], str):
            var.set_data(np.array(binned_data, dtype=object), "same")
        else:
            var.set_data(np.array(binned_data) , "same")

        # update metadata
        var.metadata.add_processing_note(f"Time binned with method {time_bin_method_dict[key].value} and cadence of {time_binning_cadence.total_seconds()/60} minutes")

    new_time_var = ep.Variable(data=binned_time, original_unit=ep.units.posixtime)
    new_time_var.metadata.add_processing_note("Created while time binning")

    return new_time_var


def _create_binned_time_and_bins(start_time:datetime,
                                 end_time:datetime,
                                 time_binning_cadence:timedelta,
                                 window_alignement:Literal["left", "center", "right"]) -> tuple[NDArray[np.floating], list[float]]:

    binned_time = np.arange(start_time.timestamp(), end_time.timestamp(), time_binning_cadence.total_seconds())

    # built bins
    total_duration = binned_time[-1] - binned_time[0]

    # Generate timestamps by stepping through the time range
    num_steps = (
        int(total_duration // time_binning_cadence.total_seconds()) + 1
    )  # Number of steps including start and end

    if window_alignement == "center":
        time_bins = [binned_time[0] - time_binning_cadence.total_seconds() / 2]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    elif window_alignement == "left":
        time_bins = [binned_time[0] - time_binning_cadence.total_seconds()]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    elif window_alignement == "right":
        time_bins = [binned_time[0]]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    else:
        msg = f"Encountered invalid window_alignment argument in time binning: {window_alignement}!"
        raise ValueError(msg)

    return binned_time, time_bins


def _calculate_index_iterables(timestamps: NDArray[np.floating], time_bins: list[float]) -> tuple[NDArray[np.intp], list[int]]:

    index_set = np.digitize(timestamps, time_bins)
    index_set = index_set - 1  # shift indices by one to match time array

    unique_indices = np.unique(index_set)

    indices_separation:list[int] = []
    cursor = 0

    for i in range(len(unique_indices)):
        # ignore values before the desired interval
        if unique_indices[i] == -1:
            continue

        while cursor < len(index_set) and index_set[cursor] != unique_indices[i]:
            cursor += 1

        indices_separation.append(cursor)

    indices_separation.append(len(index_set))
    unique_indices = np.delete(
        unique_indices, np.argwhere((unique_indices == -1) | (unique_indices == len(time_bins) - 1))
    )

    return unique_indices, indices_separation

