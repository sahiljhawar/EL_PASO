from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat

from el_paso import SavingStrategy, Variable
from el_paso.utils import enforce_utc_timezone, timed_function


@timed_function()
def save(variables_dict: dict[str, Variable],
         saving_strategy: SavingStrategy,
         start_time: datetime,
         end_time: datetime,
         time_var: Variable,
         *,
         append:bool=False) -> None:

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    time_intervals_to_save = saving_strategy.get_time_intervals_to_save(start_time, end_time)

    for interval_start, interval_end in time_intervals_to_save:
        for output_file in saving_strategy.output_files:
            file_path = saving_strategy.get_file_path(interval_start, interval_end, output_file)

            target_variables = saving_strategy.get_target_variables(output_file, variables_dict, time_var, interval_start, interval_end)

            if len(target_variables) == 0:
                warnings.warn(
                    f"Saving attempted, but product is missing some required variables for output {output_file.name}!",
                    stacklevel=2,
                )
            else:
                _save_single_file(file_path, target_variables, append=append)

def _save_single_file(file_path:Path, target_variables:dict[str,Variable], *, append:bool=False):
    """Save specified variables into the specified file name.

    Args:
        in_file_name (str): The name of the file to save data into.
        target_list (list): A list of Variable objects to be saved.

    Raises:
        UnsupportedFormatError: If the file format is not supported.
    """
    print(f"Saving file {file_path.name}...")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    format_name = file_path.suffix.lower()

    # Create a dictionary to store data and metadata
    data_dict:dict[str,NDArray[np.generic]|dict[str,Any]] = {}
    metadata_dict:dict[Any,Any] = {}

    for save_name, variable in target_variables.items():
        # Save the data_content into a field named by save_name

        data_dict[save_name] = variable.get_data()

        data_content = variable.get_data()
        if data_content.size == 0:
            warnings.warn(f"Variable {save_name} does not hold any content! Skipping ...", stacklevel=2)
            continue
        if data_content.ndim == 1:
            data_content = data_content.reshape(-1, 1)
        data_dict[save_name] = data_content
        # Create metadata for each variable
        metadata_dict[save_name] = {
            "unit": str(variable.metadata.unit),
            "original_cadence_seconds": variable.metadata.original_cadence_seconds,
            "source_files": [],
            "description": variable.metadata.description,
            "processing_notes": variable.metadata.processing_notes,
        }

    # Add metadata to the dictionary
    data_dict["metadata"] = _sanitize_metadata_dict(metadata_dict)


    if format_name == ".mat":

        if append:
            msg = "Append functionality has not been added to .mat saving!"
            raise NotImplementedError(msg)

        # Save the dictionary into a .mat file
        savemat(str(file_path), data_dict)

    elif format_name == ".pickle":

        if file_path.exists() and append:
            with file_path.open("rb") as file:
                data_dict_old = pickle.load(file)
                data_dict = _concatenate_data_dicts(data_dict_old, data_dict)

        # Save the dictionary into a .npy file
        with file_path.open("wb") as file:
            pickle.dump(data_dict, file)

    elif format_name == ".h5":
        with h5py.File(file_path, "w") as file:
            for key, value in data_dict.items():
                if key == "metadata":
                    continue
                file.create_dataset(key, data=value, compression="gzip")

    # elif format_name == ".csv":
    #     data_arr = []

    #     column_names = []  # To store column names for variables

    #     for variable in target_list:
    #         # Check if variable.data is 1D

    #         if variable.data.ndim == 1:
    #             # Add the 1D array as a column

    #             data_arr.append(variable.data)

    #             # Check if variable has save_name, else use workspace_name

    #             if hasattr(variable, "save_name") and variable.save_name:
    #                 column_names.append(variable.save_name)

    #             else:
    #                 column_names.append(variable.workspace_name)

    #     # Convert data_arr to a 2D NumPy array where each column is from the 1D arrays

    #     # Convert data_arr to a structured NumPy array where each column retains its original dtype
    #     if data_arr:
    #         # Identify the dtypes of each column
    #         dtypes = []
    #         for col_data in data_arr:
    #             if np.issubdtype(col_data.dtype, np.number):
    #                 dtypes.append((column_names[len(dtypes)], col_data.dtype))  # e.g., float or int dtype
    #             else:
    #                 max_str_len = max([len(str(item)) for item in col_data])
    #                 dtypes.append((column_names[len(dtypes)], f"U{max_str_len}"))  # Unicode string with max length

    #         # Create the structured array
    #         structured_data = np.empty(len(data_arr[0]), dtype=dtypes)

    #         # Populate the structured array with the actual data
    #         for i, col_data in enumerate(data_arr):
    #             structured_data[column_names[i]] = col_data

    #         # Create the format string for savetxt
    #         fmt = ["%s" if np.issubdtype(dtype[1], np.str_) else "%.16f" for dtype in dtypes]
    #     else:
    #         raise ValueError("Attempting to save an empty data file!")

    #     # Prepare the header string

    #     header_str = ""

    #     # Check if self.save_header is True, append the main header

    #     if self.save_header:
    #         header_str += self.header  # Assume self.header is already a string

    #         header_str += "\n"  # Add a newline after the header

    #     # Check if self.save_columns is True, append the column names

    #     if self.save_columns:
    #         header_str += self.save_separator.join(column_names)  # Join column names with the delimiter

    #         header_str += "\n"  # Add a newline after the column names

    #     # Save the data using np.savetxt

    #     np.savetxt(
    #         in_file_name, structured_data, delimiter=self.save_separator, fmt=fmt, header=header_str, comments=""
    #     )
    # elif format_name == ".txt" or format_name == ".asc":
    #     raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
    # elif format_name == ".h5":
    #     raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
    else:
        msg = f"The '{format_name}' format is not implemented."
        raise NotImplementedError(msg)

def _sanitize_metadata_dict(metadata_dict:dict[Any,Any]) -> dict[Any,Any]:
    """Sanitize the metadata dictionary by replacing None type objects with empty arrays.

    Args:
        metadata_dict (dict): The dictionary of dictionaries to be sanitized.

    Returns:
        dict: The sanitized dictionary.

    """
    sanitized_dict:dict[Any,Any] = {}

    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized_dict[key] = _sanitize_metadata_dict(value)
        elif value is None:
            # Replace None with an empty numpy array
            sanitized_dict[key] = np.array([])
        else:
            # Retain other values as they are
            sanitized_dict[key] = value

    return sanitized_dict

def _concatenate_data_dicts(data_dict_1:dict[str,Any], data_dict_2:dict[str,Any]) -> dict[str, Any]:

    time_1 = np.squeeze(data_dict_1["time"])
    time_2 = np.squeeze(data_dict_2["time"])

    idx_to_insert = np.searchsorted(time_1, time_2[0])

    time_1_in_2 = np.squeeze(np.isin(time_1, time_2))

    for key, value_1 in data_dict_1.items():

        if key not in data_dict_2:
            msg = "Key missmatch when concatenating data dicts!"
            raise ValueError(msg)

        if isinstance(value_1, np.ndarray):
            value_1 = value_1[~time_1_in_2]

            value_2 = data_dict_2[key]

            concatenated_value = value_2 if value_1.size == 0 else np.insert(value_1, idx_to_insert, value_2, axis=0)

            if key == "time" and len(np.unique(concatenated_value)) != len(concatenated_value):
                msg = "Time values were not unique when concatinating arrays!"
                raise ValueError(msg)
            data_dict_2[key] = concatenated_value

        elif isinstance(value_1, dict): # this is the metadata dict
            continue

    return data_dict_2
