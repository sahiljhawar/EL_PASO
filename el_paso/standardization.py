from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from functools import wraps, partial
from multiprocessing import Pool
import warnings
from typing import Literal

from astropy import units as u
from el_paso.classes.variable import TimeBinMethod
import numpy as np

from el_paso.classes import Variable, TimeVariable, SourceFile
from el_paso.utils import timed_function, enforce_utc_timezone


def _create_binned_time_and_bins(start_time, end_time, time_binning_cadence, window_alignement: str):
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
        raise ValueError(f"Encountered invalid window_alignment argument in time binning: {window_alignement}!")

    return binned_time, time_bins


def _calculate_index_iterables(timestamps: np.ndarray, time_bins: list):
    index_set = np.digitize(timestamps, time_bins)
    # index_set = np.where(index_set == len(time_bins), 0, index_set) # remove values before and beyond time array; -1 will be ignored later on
    index_set = index_set - 1  # shift indices by one to match time array

    unique_indices = np.unique(index_set)

    indices_separation = []
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


def load_variables(
    source_files: SourceFile | Sequence[SourceFile], start_time: datetime, end_time: datetime
) -> dict[str, Variable]:
    if not isinstance(source_files, Sequence):
        source_files = [source_files]

    variables = {}

    for source_file in source_files:
        variables |= source_file.extract_variables(start_time, end_time)

    return variables


def convert_units_to_standard(variables: dict[str, Variable]) -> None:
    for var in variables.values():
        var.convert_to_standard_unit()


@timed_function()
def time_bin_all_variables(
    variables: dict[str, Variable],
    time_binning_cadence: timedelta,
    start_time: datetime,
    end_time: datetime,
    window_alignement: Literal["center", "left"] = "center",
) -> None:
    print("Time binning ...")

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    binned_time, time_bins = _create_binned_time_and_bins(start_time, end_time, time_binning_cadence, window_alignement)

    # Cache digitized indices for every time variable
    index_iterables = {}

    time_variables = {key: var for key, var in variables.items() if isinstance(var, TimeVariable)}
    non_time_variables = {key: var for key, var in variables.items() if not isinstance(var, TimeVariable)}

    for key, var in non_time_variables.items():
        # Just repeat in case of no time dependency
        if var.time_variable is None:
            if var.metadata.time_bin_method == TimeBinMethod.Repeat:
                var.data = np.repeat(var.data[np.newaxis, ...], len(binned_time), axis=0)
                var.time_variable = get_standard_variable_by_type("Epoch", variables)
            continue

        # check if time variable and data content sizes match
        if var.data.shape[0] != len(var.time_variable.data):
            raise ValueError(f"Variable {key}: size of dimension 0 does not match length of time variable!")

        # calculate bin indices for given time array if it has not been calculated before
        if var.time_variable not in index_iterables.keys():
            assert var.time_variable.metadata.unit == u.posixtime
            index_iterables[var.time_variable] = _calculate_index_iterables(var.time_variable.data, time_bins)

        unique_indices, indices_separation = index_iterables[var.time_variable]

        # Initialize binned_data as an array of np.nans with the same shape as self.data,
        # but with the length of the first dimension matching the length of time_array
        if var.data.dtype.kind in {"U", "S", "O"}:  # Check if the data is string or object type
            binned_data = np.full((len(binned_time),), "", dtype=var.data.dtype)
        else:
            binned_data_shape = (len(binned_time),) + var.data.shape[1:]
            binned_data = np.full(binned_data_shape, np.nan)

        # Iterate over unique indices
        for i, unique_index in enumerate(unique_indices):
            bin_data = var.data[indices_separation[i] : indices_separation[i + 1]]
            if len(bin_data) == 0 or not np.any(np.isfinite(bin_data)):
                continue  # no data found
            binned_value = var.metadata.time_bin_method(bin_data)

            # Update the relevant slice of binned_data
            binned_data[unique_index, ...] = binned_value

        # Update relevant metadata fields
        # Ensure binned_data works for both numeric and string data
        if isinstance(binned_data[0], str):
            var.data = np.array(binned_data, dtype=object)
        else:
            var.data = np.array(binned_data)

        # update metadata
        var.metadata.time_bin_interval = time_binning_cadence.total_seconds()

    # After we binned all other variables, we can set the time variables to the binned time array
    for var in time_variables.values():
        if var.do_time_binning:
            var.data = binned_time
            continue


def validate_standard(func):
    def validate_units(variables: dict[str, Variable]):
        for key, var in variables.items():
            if var.standard:
                assert (
                    var.metadata.unit == var.standard.standard_unit
                ), f"Variable {key} has wrong units! Actual unit: {var.metadata.unit}. Should be: {var.standard.standard_unit}."

    def validate_dimensions(variables: dict[str, Variable]):
        for key, var in variables.items():
            if var.standard:
                match var.standard.variable_type:
                    case "Epoch":
                        assert (
                            var.data.ndim == 1
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Flux":
                        if "DO" in var.standard_name:
                            expected_ndim = 2
                        elif "DU" in var.standard_name:
                            expected_ndim = 3
                        assert (
                            var.data.ndim == expected_ndim
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Energy":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "PitchAngle":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Position":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                        assert (
                            var.data.shape[1] == 3
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong shape: {var.data.shape}!"

                    case "MLT":
                        assert (
                            var.data.ndim == 1
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"


    @wraps(func)
    def with_validated_standard(*args, **kwargs):
        variables = args[0]
        assert isinstance(variables, dict)

        validate_units(variables)
        validate_dimensions(variables)

        return func(*args, **kwargs)

    return with_validated_standard


def get_standard_variable_by_name(standard_name: str, variables: dict[str, Variable]):
    num_vars_found = 0
    var_found = None

    for var in variables.values():
        if var.standard_name == standard_name:
            var_found = var
            num_vars_found += 1

    if num_vars_found == 0:
        raise ValueError(f"No variable found with standard name {standard_name}!")
    elif num_vars_found > 1:
        raise ValueError(
            f"Found {num_vars_found} variables with standard name {standard_name}! Please specify the variable yourself."
        )
    else:
        return var_found


def get_standard_variable_by_type(standard_type: str, variables: dict[str, Variable]):
    num_vars_found = 0
    var_found = None

    for var in variables.values():
        if var.standard and var.standard.variable_type == standard_type:
            var_found = var
            num_vars_found += 1

    if num_vars_found == 0:
        msg = f"No variable found with variable type {standard_type}!"
        raise ValueError(msg)

    if num_vars_found > 1:
        msg = f"Found {num_vars_found} variables with variable type {standard_type}! Please specify the variable yourself."
        raise ValueError(msg)

    return var_found
