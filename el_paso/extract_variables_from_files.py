from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import cdflib
import numpy as np
import pandas as pd
from astropy import units as u
from numpy.typing import NDArray

from el_paso import Variable
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version, timed_function


@dataclass(frozen=True, slots=True, eq=False)
class ExtractionInfo:
    """Class to store information about the extraction of variables from a file."""

    result_key: str
    name_or_column: str|int
    unit: u.UnitBase
    is_time_dependent: bool = True
    dependent_variables: list[str]|None = None


@timed_function()
def extract_variables_from_files(start_time: datetime,
                                 end_time: datetime,
                                 file_cadence: Literal["daily", "monthly", "single_file"],
                                 data_path: Path|str,
                                 file_name_stem: str,
                                 extraction_infos: Iterable[ExtractionInfo],
                                 pd_read_csv_kwargs: dict[str, Any]|None=None,
                                 ) -> dict[str, Variable]:
    print("Extracting variables ...")

    if pd_read_csv_kwargs is None:
        pd_read_csv_kwargs = {}

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    data_path = Path(data_path)

    files_list, _ = _construct_file_list(start_time, end_time, file_cadence, data_path / file_name_stem)

    if len(files_list) == 0:
        msg = "No file found to extract variables!"
        raise ValueError(msg)

    variable_data = {info.name_or_column: np.array([]) for info in extraction_infos}

    for file_path in files_list:
        if file_path.suffix == ".cdf":
            new_data = _extract_data_from_cdf(str(file_path), tuple(extraction_infos))
        elif file_path.suffix in [".txt", ".asc", ".csv", ".tab"]:
            new_data = _extract_data_from_ascii(str(file_path), tuple(extraction_infos), pd_read_csv_kwargs)
        elif file_path.suffix == ".nc":
            raise NotImplementedError("NetCDF reading is not supported yet!")
        elif file_path.suffix == ".h5":
            raise NotImplementedError("HDF5 reading is not supported yet!")
        elif file_path.suffix == ".json":
            new_data = _extract_data_from_json(str(file_path), tuple(extraction_infos))
        elif file_path.suffix == ".mat":
            raise NotImplementedError("MATLAB .mat file reading is not supported yet!")
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


def _construct_file_list(start_time: datetime,
                         end_time: datetime,
                         file_cadence:Literal["daily", "monthly", "single_file"],
                         file_path:Path) -> tuple[list[Path], list[tuple[datetime,datetime]]]:

    file_paths:list[Path] = []
    time_intervals:list[tuple[datetime,datetime]] = []

    match file_cadence:

        case "daily":
            current_time = start_time
            while current_time <= end_time:

                file_path_current = _fill_file_name_and_check_version(current_time, file_path)
                if file_path_current is None:
                    logging.warning(
                        f"No file found for {current_time.strftime('%Y-%m-%d')} under path: {file_path}. Skipping this day.",
                        stacklevel=2,
                    )
                else:
                    file_paths.append(file_path_current)

                    day_start = datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0, tzinfo=timezone.utc)
                    day_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59, tzinfo=timezone.utc)
                    time_intervals.append((day_start, day_end))

                current_time += timedelta(days=1)

        case "monthly":
            raise NotImplementedError("Monthly file cadence is not implemented yet!")

        case "single_file":
            file_path_current = _fill_file_name_and_check_version(start_time, file_path)
            if file_path_current is None:
                raise FileNotFoundError(f"No file found under the specified path: {file_path}. Please check your data_path.")

            file_paths.append(file_path_current)

            # day_start = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0, tzinfo=timezone.utc)
            # day_end = datetime(start_time.year, start_time.month, start_time.day, 23, 59, 59, tzinfo=timezone.utc)
            # time_intervals = [[start_time, datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59, tzinfo=timezone.utc)]]
            time_intervals.append((start_time, end_time))

    return file_paths, time_intervals

def _fill_file_name_and_check_version(time: datetime, file_path:Path) -> Path|None:
    file_path = Path(fill_str_template_with_time(str(file_path), time))

    file_names_all_versions = _find_files_with_regex(file_path.parent, file_path.name)

    file_path_latest = get_file_by_version(file_names_all_versions, version="latest")

    if file_path_latest is None:
        warnings.warn(
            f"No file found under path: {file_path.parent}! Have you called download()? Check your download_path argument.",
            stacklevel=2,
        )

    return file_path_latest

def _extract_data_from_ascii(file_path: str, extraction_infos: tuple[ExtractionInfo], pd_read_csv_kwargs: dict) -> dict[str|int, NDArray[np.generic]]:
    """Load data from an ASCII file and updates the variables.

    Args:
        file_path (str): The path to the ASCII file.
        variables (List[Variable], optional): Specific variables to load from the file.

    """
    df = pd.read_csv(file_path, **pd_read_csv_kwargs)

    data:dict[str|int, NDArray[np.generic]] = {}

    for info in extraction_infos:
        if isinstance(info.name_or_column, int):
            data[info.name_or_column] = np.asarray(df.iloc[:, info.name_or_column].values)
        elif isinstance(info.name_or_column, str):
            data[info.name_or_column] = np.asarray(df.loc[:, info.name_or_column].values)
        else:
            msg = f"Encountered invalid name_or_column value: {info.name_or_column}! Must be int or str!"
            raise TypeError(msg)

    return data

def _extract_data_from_json(file_path: str, extraction_infos: tuple[ExtractionInfo]) -> dict[str|int, NDArray[np.generic]]:

    warnings.warn("Enountered JSON file. Please specify your dependent variables!", stacklevel=2)

    with Path(file_path).open("r") as f:
        data = json.load(f)

    json_df = pd.DataFrame(data)

    data:dict[str|int, NDArray[np.generic]] = {}

    for info in extraction_infos:
        if info.name_or_column in json_df:

            if info.dependent_variables is None:
                data[info.name_or_column] = np.array(pd.unique(json_df[info.name_or_column]))
            else:

                dependent_data = [json_df[dep_var_name] for dep_var_name in info.dependent_variables]

                unique_values = [pd.unique(dep) for dep in dependent_data]
                shape = tuple(len(uq) for uq in unique_values)

                # Determine the correct dtype for the data_array based on the column data type
                dtype = object if json_df[info.name_or_column].dtype == object else np.float64

                data_array = np.full(shape, np.nan, dtype=dtype)


                for indices in np.ndindex(*shape):
                    mask = np.ones(len(json_df), dtype=bool)
                    for i, idx in enumerate(indices):
                        mask &= dependent_data[i] == unique_values[i][idx]
                    if mask.any():
                        data_array[indices] = json_df[info.name_or_column][mask].to_numpy()[0]

                data[info.name_or_column] = data_array
        else:
            warnings.warn(
                f"Variable with name_or_column_in_file: {info.name_or_column} was not found in file {file_path}!",
            )

    return data

def _extract_data_from_cdf(file_path: str, extraction_infos: tuple[ExtractionInfo]) -> dict[str|int, NDArray[np.generic]]:
    # Open the CDF file
    cdf_file = cdflib.CDF(file_path)
    cdfinfo = cdf_file.cdf_info()
    variable_data:dict[str|int, NDArray[np.generic]] = {}

    # Extract data for each variable in self.variables
    for info in extraction_infos:
        if info.name_or_column in cdfinfo.zVariables:
            # Retrieve data corresponding to the variable name from the CDF file
            var_content = cdf_file.varget(info.name_or_column)
            if isinstance(var_content, str):
                # If the content is a string, convert it to a numpy array
                var_content = np.array([var_content])
            variable_data[info.name_or_column] = var_content
        else:
            logging.warn(f"Data with name {info.name_or_column} was not found in file {file_path}!", stacklevel=2)

    return variable_data

def _find_files_with_regex(directory:Path, regex_pattern:str) -> list[Path]:
    """Find files in a directory that match a given regex pattern."""
    compiled_regex = re.compile(regex_pattern) # Compile for efficiency

    return [file for file in directory.glob("*") if compiled_regex.search(str(file))]
