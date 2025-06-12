from __future__ import annotations

import calendar
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as tz
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from numpy.typing import NDArray

from el_paso.classes import Variable
from el_paso.classes.extraction_functions import ExtractionInfo, extract_data_from_ascii, extract_varibles_from_cdf
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version, timed_function


class SourceFile:
    def __init__(
        self,
        download_path: str,
        download_url: str = "",
        download_arguments_prefixes: str = "",
        download_arguments_suffixes: str = "",
        file_cadence: Literal["daily", "monthly", "single_file"] = "daily",
    ) -> None:
        self.download_url = download_url
        self.download_arguments_prefixes = download_arguments_prefixes
        self.download_arguments_suffixes = download_arguments_suffixes
        self.download_path = download_path
        self.file_cadence = file_cadence

    def _wget_download(self, current_time:datetime):

        Path(fill_str_template_with_time(self.download_path, current_time)).parents[0].mkdir(exist_ok=True, parents=True)

        # Replace "yyyymmdd" or "YYYYMMDD" in url, prefix, and suffix with the parsed string
        url = fill_str_template_with_time(self.download_url, current_time)
        prefix = fill_str_template_with_time(self.download_arguments_prefixes, current_time)
        suffix = fill_str_template_with_time(self.download_arguments_suffixes, current_time)
        download_command = f"wget {prefix} {url} {suffix}"
        print(download_command)

        # Execute the download command
        try:
            os.system(download_command)
        except Exception as e:
            print(f"Error downloading file using command {download_command}: {e}")

    def download(self, start_time: datetime, end_time: datetime):
        """Downloads the product data according to the specified standard."""

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        curr_time = start_time

        match self.file_cadence:
            case "daily":
                while curr_time <= end_time:
                    self._wget_download(curr_time)
                    curr_time += timedelta(days=1)

            case "monthly":
                raise NotImplementedError

            case "single_file":
                self._wget_download(start_time)

            case _:
                raise NotImplementedError


    @timed_function()
    def extract_variables(self, start_time: datetime,
                          end_time: datetime,
                          extraction_infos: list[ExtractionInfo],
                          pd_read_csv_kwargs:dict[str, Any]|None=None)-> dict[str, Variable]:
        print("Extracting variables ...")

        if pd_read_csv_kwargs is None:
            pd_read_csv_kwargs = {}

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        files_list, _ = self._construct_downloaded_file_list(start_time, end_time)

        if len(files_list) == 1 and files_list[0] is None:
            raise ValueError("No file found to extract variables!")

        variable_data = {info.name_or_column: np.array([]) for info in extraction_infos}

        for file_path in files_list:
            if file_path is None:
                continue

            if file_path.suffix == ".cdf":
                new_data = extract_varibles_from_cdf(str(file_path), extraction_infos)
            elif file_path.suffix in [".txt", ".asc", ".csv", ".tab"]:
                new_data = extract_data_from_ascii(str(file_path), extraction_infos, pd_read_csv_kwargs)
            elif file_path.suffix == ".nc":
                self._load_nc_file_to_extract(file_path)
            elif file_path.suffix == ".h5":
                raise NotImplementedError("HDF5 reading is not supported yet!")
                # self._load_h5_file_to_extract(file_path, variables)
            elif file_path.suffix == ".json":
                self._extract_variables_from_json(file_path)
            elif file_path.suffix == ".mat":
                self._load_mat_file_to_extract(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

            # Update the data content of variables
            for info in extraction_infos:
                key = info.name_or_column
                if variable_data[key].size == 0:
                    variable_data[key] = new_data[key]
                elif info.is_time_dependent:
                    print(f"Concatenating data for {key} ...")
                    variable_data[key] = np.concatenate((variable_data[key], new_data[key]), axis=0)

        # create variables based on the extraction_infos
        variables:dict[str, Variable] = {}

        for info in extraction_infos:
            variables[info.result_key] = Variable(original_unit=info.unit, data=variable_data[info.name_or_column])
            variables[info.result_key].metadata.source_files = [path.name for path in files_list]

        return variables

    def _extract_variables_from_json(self, file_path: str) -> None:

        warnings.warn("Enountered JSON file. Please specify your dependent variables!", stacklevel=2)

        with Path(file_path).open("r") as f:
            data = json.load(f)

        json_df = pd.DataFrame(data)

        variable_data = {}

        for key, var in self.variables_to_extract.items():
            name_or_column_in_file = var.name_or_column_in_file
            if name_or_column_in_file in json_df:
                if len(var.dependent_variables) == 0 and var.time_variable is None:
                    variable_data[var.name_or_column_in_file] = np.array(pd.unique(json_df[name_or_column_in_file]))
                else:
                    dependent_data = [json_df[var.time_variable.name_or_column_in_file]]
                    dependent_data += [
                        json_df[dep_var.name_or_column_in_file] for dep_var in var.dependent_variables
                    ]

                    unique_values = [pd.unique(dep) for dep in dependent_data]
                    shape = tuple(len(uq) for uq in unique_values)

                    # Determine the correct dtype for the data_array based on the column data type
                    dtype = object if json_df[name_or_column_in_file].dtype == object else float

                    data_array = np.full(shape, np.nan, dtype=dtype)

                    # for dep_idx in range(len(dependent_data)):
                    #     for i, unique_value in enumerate(unique_values[dep_idx]):
                    #         data_array[]

                    for indices in np.ndindex(*shape):
                        mask = np.ones(len(json_df), dtype=bool)
                        for i, idx in enumerate(indices):
                            mask &= dependent_data[i] == unique_values[i][idx]
                        if mask.any():
                            data_array[indices] = json_df[name_or_column_in_file][mask].to_numpy()[0]

                    variable_data[var.name_or_column_in_file] = data_array
            else:
                warnings.warn(
                    f"Variable {key} with name_or_column_in_file: {var.name_or_column_in_file} was not found in file {file_path}!",
                )

        for var in self.variables_to_extract.values():
            if var.data is not None and (isinstance(var, TimeVariable) or var.time_variable is not None):
                var.data = np.concatenate((var.data, variable_data[var.name_or_column_in_file]), axis=0)
            else:
                var.data = variable_data[var.name_or_column_in_file]

    def _get_downloaded_file_name(self, time: datetime) -> Path:
        file_path = Path(fill_str_template_with_time(self.download_path, time))

        file_names_all_versions = file_path.parent.glob(file_path.name)

        file_path_latest = get_file_by_version(file_names_all_versions, version="latest")

        if file_path_latest is None:
            warnings.warn(
                f"No file found under path: {file_path.parent}! Have you called download()? Check your download_path argument.",
                stacklevel=2,
            )

        return file_path_latest

    def _construct_downloaded_file_list(self, start_time: datetime, end_time: datetime) -> tuple[list[Path], list[list[datetime]]]:

        file_paths = []
        time_intervals = []

        match self.file_cadence:

            case "daily":
                current_time = start_time
                while current_time <= end_time:

                    file_path = self._get_downloaded_file_name(current_time)
                    file_paths.append(file_path)

                    day_start = datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0, tzinfo=tz.utc)
                    day_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59, tzinfo=tz.utc)
                    time_intervals.append([day_start, day_end])

                    current_time += timedelta(days=1)

            case "monthly":
                current_time = start_time.replace(day=1)
                while current_time <= end_time:
                    year = current_time.year
                    month = current_time.month
                    eom_day = calendar.monthrange(year, month)[1]
                    time_str = f"{year:04d}{month:02d}01to{year:04d}{month:02d}{eom_day:02d}"
                    path_list = []
                    for path in self.download_paths:
                        file_path = self.get_source_file_name(path, time_str)
                        path_list.append(file_path)
                    file_paths.append(path_list)
                    month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=tz.utc)
                    month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=tz.utc)
                    time_intervals.append([month_start, month_end])
                    if month == 12:  # noqa: PLR2004
                        current_time = datetime(year + 1, 1, 1, tzinfo=tz.utc)
                    else:
                        current_time = datetime(year, month + 1, 1, tzinfo=tz.utc)

            case "single_file":
                file_paths.append(self._get_downloaded_file_name(start_time))

                day_start = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0, tzinfo=tz.utc)
                day_end = datetime(start_time.year, start_time.month, start_time.day, 23, 59, 59, tzinfo=tz.utc)
                time_intervals = [[start_time, datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59, tzinfo=tz.utc)]]

        return file_paths, time_intervals

    def _load_mat_file_to_access(self, file_path: str, variables):
        """Loads data from a .mat file and updates the variables.

        Args:
            file_path (str): The path to the .mat file.
            variables (List[Variable], optional): Specific variables to load from the file.

        Returns:
            return_variables (List[Variable]): list of Variable objects to return

        """
        mat_data = loadmat_nested(file_path)
        save_standard = self.current_save_standard  # Assuming self.save_standard is set
        if self.current_save_standard is None:
            save_standard = self.default_save_standard

        mat_data_fields = list(mat_data.keys())
        return_variables = []
        for var in self.variables:
            save_name = save_standard.variable_mapping(var)
            if save_name in mat_data_fields:
                self.set_variable_attribute(var.workspace_name, "data_content", mat_data[save_name])
                return_var = var
                return_var.workspace_name = save_name
                return_variables.append(return_var)
                mat_data_fields.remove(save_name)

        for field_name in mat_data_fields:
            #  if field_name != 'metadata' and not field_name.startswith('__'):
            if not field_name.startswith("__"):
                new_var = Variable(
                    data_content=mat_data[field_name], workspace_name=field_name, save_standard=save_standard,
                )
                return_variables.append(new_var)

        if variables is not None:
            return_variables = [variable for variable in return_variables if variable.workspace_name in variables]

        self.variables = return_variables
        return return_variables

    def _load_ascii_file_to_access(self, file_path: str, variables) -> None:
        """Loads data from an ASCII file and updates the variables.

        Args:
            file_path (str): The path to the ASCII file.
            variables (List[Variable], optional): Specific variables to load from the file.

        """
        return_variables = []
        # Step 1: Open the ASCII file and read the header rows manually if needed
        with open(file_path) as file:
            # Read the header rows (self.header_length defines how many rows to read)
            self.header = [next(file).strip() for _ in range(self.header_length)]

        # Step 2: Use pandas.read_csv to read the file from disk, skipping header rows
        # If save_columns is True, read column names from the file, otherwise read without column names
        if self.save_columns:
            df = pd.read_csv(file_path, delimiter=self.save_separator, skiprows=self.header_length)
            column_names = df.columns.tolist()
        else:
            df = pd.read_csv(file_path, delimiter=self.save_separator, skiprows=self.header_length, header=None)
            column_names = None

        # Step 3: Process the data based on the number of columns
        num_columns = df.shape[1]  # Number of columns in the dataframe

        if num_columns == len(self.target_variables):
            # Case 1: Data matches the number of target_variables
            for i, var_name in enumerate(self.target_variables):
                # Find the variable with the matching workspace_name
                variable = None
                for var in self.variables:
                    if var.workspace_name == var_name:
                        variable = var
                        break

                if variable:
                    # Set the data_content for the variable from the DataFrame column
                    variable.data = df.iloc[:, i].values
                    return_variables.append(variable)
        elif self.save_columns and column_names:
            # Case 2: Matching columns with save_name of variables or creating new ones
            for i, col_name in enumerate(column_names):
                # Try to find a variable with the matching save_name
                variable = None
                for var in self.variables:
                    # Check if 'save_name' exists and is not None, otherwise use 'workspace_name'
                    name_to_check = (
                        var.save_name if hasattr(var, "save_name") and var.save_name is not None else var.workspace_name
                    )
                    if name_to_check == col_name:
                        variable = var
                        break

                if variable:
                    # If a match is found, update its data_content from the DataFrame
                    variable.data = df[col_name].values
                    return_variables.append(variable)
                else:
                    # If no match, create a new variable with the column data
                    new_var = Variable(data_content=df[col_name].values, workspace_name=col_name)
                    return_variables.append(new_var)
        else:
            raise ValueError("Mismatch between the number of data columns and target variables or column names.")

        # Step 5: Filter return_variables based on the provided 'variables' argument, if any
        if variables is not None:
            return_variables = [variable for variable in return_variables if variable.workspace_name in variables]

        self.variables = return_variables

    def open_data_file_to_extract(self, file_path: str, variables) -> None:
        """Opens the data file and loads the content based on the file format.

        Args:
            file_path (str): The path to the data file.
            variables (List[Variable], optional): Variables to pull out from the file

        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension == ".cdf":
            self._load_cdf_file_to_extract(file_path, variables)
        elif file_extension in [".txt", ".asc", ".csv", ".tab"]:
            self._load_ascii_file_to_extract(file_path, variables)
        elif file_extension == ".nc":
            self._load_nc_file_to_extract(file_path, variables)
        elif file_extension == ".h5":
            raise NotImplementedError("HDF5 reading is not supported yet!")
            # self._load_h5_file_to_extract(file_path, variables)
        elif file_extension == ".json":
            self._load_json_file_to_extract(file_path, variables)
        elif file_extension == ".mat":
            self._load_mat_file_to_extract(file_path, variables)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def _load_mat_file_to_extract(self, file_path: str, variables) -> None:
        """Loads data from a .mat file and updates the variables.

        Args:
            file_path (str): The path to the .mat file.
            variables (List[Variable], optional): Specific variables to load from the file.

        """
        mat_data = sio.loadmat(file_path)
        save_standard = self.save_standard  # Assuming self.save_standard is set

        def update_variable(var_name: str, data_content) -> None:
            for var in self.variables:
                if var.save_standard == save_standard and var.standard_name == var_name:
                    if var.data is not None and var.data.any():
                        var.data = np.concatenate((var.data, data_content), axis=0)
                    else:
                        var.data = data_content
                    return
            new_var = Variable(data_content=data_content, workspace_name=var_name, save_standard=var_name)
            self.variables.append(new_var)

        if variables:
            for var in variables:
                save_name = var.save_standard
                if save_name in mat_data:
                    if var.data is not None and var.data.any():
                        var.data = np.concatenate((var.data, mat_data[save_name]), axis=0)
                    else:
                        var.data = mat_data[save_name]
        else:
            for var_name, data_content in mat_data.items():
                if not var_name.startswith("__"):  # Skipping meta variables in .mat files
                    update_variable(var_name, data_content)

    def _load_nc_file_to_extract(self, file_path: str) -> None:
        """Loads data from a NetCDF file and updates the variables.

        Args:
            file_path (str): The path to the NetCDF file.
            variables (List[Variable], optional): Specific variables to load from the file.

        """
        # Open the NetCDF file
        nc_file = sio.netcdf.NetCDFFile(file_path, "r")
        variable_data = {}

        # Extract data for each variable in self.variables
        for var in self.variables:
            if var.name_or_column_in_file in nc_file.variables:
                # Retrieve data corresponding to the variable name from the CDF file
                variable_data[var.name_or_column_in_file] = nc_file.variables[var.name_or_column_in_file]

        # Update the data content of variables
        for var in self.variables:
            if var.name_or_column_in_file in variable_data:
                if var.data is not None and var.data.any():
                    var.data = np.concatenate((var.data, variable_data[var.name_or_column_in_file]), axis=0)
                else:
                    var.data = variable_data[var.name_or_column_in_file]
