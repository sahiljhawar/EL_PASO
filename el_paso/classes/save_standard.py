import calendar
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from astropy import units as u
from scipy import io as sio

from el_paso.utils import enforce_utc_timezone, get_key_by_value


class SaveCadence(Enum):
    DAILY = 0
    MONTHLY = 1
    ONE_FILE = 2

@dataclass
class OutputFile:
    name: str
    variable_names_to_save: list
    variable_units: list

    def __post_init__(self):
        if len(self.variable_names_to_save) != len(self.variable_units):
            msg = "Variables names and variable units have different lengths!"
            raise ValueError(msg)


class SaveStandard(ABC):
    def __init__(
        self,
        mission: str,
        source: str,
        instrument: str,
        model: str,
        mfm: str,
        version: str,
        save_text_segments: Dict[str, List[str]],
        default_db: str,
        default_format: str,
        product_variable_names: Dict[str, str] = None,
        outputs: List[str] = [],
        files: Dict[str, List[str]] = {},
        file_variables: Dict[str, List[str]] = [],
    ):
        """
        Initializes a SaveStandard object.

        Args:
            mission (str): The mission associated with the save standard.
            source (str): The satellite associated with the save standard.
            instrument (str): The instrument associated with the save standard.
            model (str): The model associated with the save standard.
            mfm (str): The magnetic field model associated with the save standard.
            version (str): The version of the save standard.
            save_text_segments (Dict[str, List[str]]): A dictionary of text segments used for saving files.
            default_db (str): The default database associated with the save standard.
            default_format (str): The default format for saving files.
            outputs (List[str]): A list of output types associated with the save standard.
            files (Dict[str, List[str]]): A dictionary mapping file types to file paths.
            file_variables (Dict[str, List[str]]): A dictionary mapping file types to variables.
        """
        self.mission = mission
        self.source = source
        self.instrument = instrument
        self.model = model
        self.mfm = mfm
        self.version = version
        self.save_text_segments = save_text_segments
        self.default_db = default_db
        self.default_format = default_format
        self.outputs = outputs
        self.files = files
        self.file_variables = file_variables
        self.product_variable_names = product_variable_names
        self.number_precision = np.float32

        # This variables have to be specified by the derived class
        self.output_files: list[OutputFile] = []

    @abstractmethod
    def get_saved_file_name(
        self, start_time: datetime, end_time: datetime, output_file: OutputFile, external_text: Optional[str] = None
    ) -> str:
        """
        Get the saved file name based on a time string, output type, and optional external text.

        Args:
            time_string (str): The time string used to generate the file name.
            output_variable_name (str): The type of output for which the file name is being generated.
            external_text (str, optional): An optional external text to include in the file name.

        Returns:
            str: The generated file name.
        """
        pass

    def get_time_intervals_to_save(self, start_time, end_time):
        time_intervals = []

        match self.save_cadence:

            case SaveCadence.DAILY:
                current_time = start_time
                while current_time <= end_time:
                    day_start = datetime(
                        current_time.year, current_time.month, current_time.day, 0, 0, 0, tzinfo=timezone.utc,
                    )
                    day_end = datetime(
                        current_time.year, current_time.month, current_time.day, 23, 59, 59, tzinfo=timezone.utc,
                    )
                    time_intervals.append([day_start, day_end])
                    current_time += timedelta(days=1)
            case SaveCadence.MONTHLY:
                current_time = start_time.replace(day=1)
                while current_time <= end_time:
                    year = current_time.year
                    month = current_time.month
                    eom_day = calendar.monthrange(year, month)[1]

                    month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
                    month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=timezone.utc)
                    time_intervals.append([month_start, month_end])
                    current_time = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)

            case SaveCadence.ONE_FILE:
                time_intervals = [(start_time, end_time)]

        return time_intervals

    def save(self, start_time: datetime, end_time: datetime, variables_dict, saved_filename_extra_text=""):

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        time_intervals_to_save = self.get_time_intervals_to_save(start_time, end_time)

        for interval_start, interval_end in time_intervals_to_save:
            for output_file in self.output_files:
                # Get parent directory and create it if it does not exist yet
                file_name = Path(
                    self.get_saved_file_name(interval_start, interval_end, output_file, saved_filename_extra_text)
                )
                # Extract the directory path from the file path
                directory = file_name.parent
                # Create the directories if they don't exist
                if directory:  # Only create directories if there's a directory path (not an empty string)
                    directory.mkdir(exist_ok=True)

                target_variables = self._get_target_variables(output_file, variables_dict, start_time, end_time)

                if len(target_variables) == 0:
                    warnings.warn(
                        f"Saving attempted, but product is missing some required variables for output {output_file.name}!",
                        stacklevel=2,
                    )
                else:
                    self._save_single_file(file_name, target_variables)

    def _get_target_variables(self, output_file: OutputFile, variables_dict, start_time, end_time):
        target_variables = {}

        # if no variables have been specified, we save all of them
        if len(output_file.variable_names_to_save) == 0:
            for key, var in variables_dict.items():
                var_to_save = deepcopy(var)
                var_to_save.truncate(start_time.timestamp(), end_time.timestamp())
                if np.issubdtype(var_to_save.data.dtype, np.number):
                    var_to_save.data = var_to_save.data.astype(self.number_precision)
                target_variables[key] = var_to_save

            return target_variables

        for variable_unit, variable_name_to_save in zip(output_file.variable_units, output_file.variable_names_to_save):
            name_in_product = self.product_variable_names[variable_name_to_save]
            if name_in_product in variables_dict:
                var_to_save = deepcopy(variables_dict[name_in_product])
                key = get_key_by_value(self.product_variable_names, name_in_product)
                var_to_save.convert_to_unit(variable_unit)
                target_variables[key] = var_to_save
            else:
                warnings.warn(f"Could not find target variable {name_in_product}!", stacklevel=2)
                return []

        return target_variables

    def _sanitize_metadata_dict(self, metadata_dict):
        """
        Sanitize the metadata dictionary by replacing None type objects with empty arrays.

        Args:
            metadata_dict (dict): The dictionary of dictionaries to be sanitized.

        Returns:
            dict: The sanitized dictionary.
        """
        sanitized_dict = {}

        for key, value in metadata_dict.items():
            if isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized_dict[key] = self._sanitize_metadata_dict(value)
            elif value is None:
                # Replace None with an empty numpy array
                sanitized_dict[key] = np.array([])
            else:
                # Retain other values as they are
                sanitized_dict[key] = value

        return sanitized_dict

    def _save_single_file(self, in_file_name, target_variables):
        """
        Saves specified variables into the specified file name.

        Args:
            in_file_name (str): The name of the file to save data into.
            target_list (list): A list of Variable objects to be saved.

        Raises:
            UnsupportedFormatError: If the file format is not supported.
        """

        print(f"Saving file {in_file_name}...")

        in_file_name = Path(in_file_name)

        format_name = in_file_name.suffix.lower()

        if format_name == ".mat":
            # Create a dictionary to store data and metadata
            data_dict = {}
            metadata_dict = {}

            for save_name, variable in target_variables.items():
                # Save the data_content into a field named by save_name

                data_dict[save_name] = variable.data

                data_content = variable.data
                if data_content is None:
                    warnings.warn(f"Variable {variable.standard_name} does not hold any content! Skipping ...")
                    continue
                if data_content.ndim == 1:
                    data_content = data_content.reshape(-1, 1)
                data_dict[save_name] = data_content
                # Create metadata for each variable
                metadata_dict[save_name] = {
                    "unit": str(variable.metadata.unit),
                    "original_cadence_seconds": variable.metadata.original_cadence_seconds,
                    "source_files": variable.metadata.source_files,
                    "description": variable.metadata.description,
                    "processing_notes": variable.metadata.processing_notes,
                    "time_bin_method": variable.metadata.time_bin_method,
                    "time_bin_interval": variable.metadata.time_bin_interval,
                }

            # Add metadata to the dictionary
            data_dict["metadata"] = self._sanitize_metadata_dict(metadata_dict)

            # Save the dictionary into a .mat file
            sio.savemat(str(in_file_name), data_dict)

        elif format_name == ".csv":
            data_arr = []

            column_names = []  # To store column names for variables

            for variable in target_list:
                # Check if variable.data is 1D

                if variable.data.ndim == 1:
                    # Add the 1D array as a column

                    data_arr.append(variable.data)

                    # Check if variable has save_name, else use workspace_name

                    if hasattr(variable, "save_name") and variable.save_name:
                        column_names.append(variable.save_name)

                    else:
                        column_names.append(variable.workspace_name)

            # Convert data_arr to a 2D NumPy array where each column is from the 1D arrays

            # Convert data_arr to a structured NumPy array where each column retains its original dtype
            if data_arr:
                # Identify the dtypes of each column
                dtypes = []
                for col_data in data_arr:
                    if np.issubdtype(col_data.dtype, np.number):
                        dtypes.append((column_names[len(dtypes)], col_data.dtype))  # e.g., float or int dtype
                    else:
                        max_str_len = max([len(str(item)) for item in col_data])
                        dtypes.append((column_names[len(dtypes)], f"U{max_str_len}"))  # Unicode string with max length

                # Create the structured array
                structured_data = np.empty(len(data_arr[0]), dtype=dtypes)

                # Populate the structured array with the actual data
                for i, col_data in enumerate(data_arr):
                    structured_data[column_names[i]] = col_data

                # Create the format string for savetxt
                fmt = ["%s" if np.issubdtype(dtype[1], np.str_) else "%.16f" for dtype in dtypes]
            else:
                raise ValueError("Attempting to save an empty data file!")

            # Prepare the header string

            header_str = ""

            # Check if self.save_header is True, append the main header

            if self.save_header:
                header_str += self.header  # Assume self.header is already a string

                header_str += "\n"  # Add a newline after the header

            # Check if self.save_columns is True, append the column names

            if self.save_columns:
                header_str += self.save_separator.join(column_names)  # Join column names with the delimiter

                header_str += "\n"  # Add a newline after the column names

            # Save the data using np.savetxt

            np.savetxt(
                in_file_name, structured_data, delimiter=self.save_separator, fmt=fmt, header=header_str, comments=""
            )
        elif format_name == ".txt" or format_name == ".asc":
            raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
        elif format_name == ".h5":
            raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
        else:
            raise UnsupportedFormatError(f"The '{format_name}' format is not implemented.")
