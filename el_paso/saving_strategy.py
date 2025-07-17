import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import h5py  # type: ignore[reportMissingTypeStubs]
from scipy.io import savemat  # type: ignore[reportMissingTypeStubs]

from el_paso import Variable

logger = logging.getLogger(__name__)

class OutputFile(NamedTuple):
    """Represents an output file with its name and a list of variable names to save.

    Attributes:
        name (str): The name of the output file.
        names_to_save (list[str]): List of variable names to be saved in the output file.
    """
    name: str
    names_to_save: list[str]

class SavingStrategy(ABC):
    """Abstract base class for defining strategies to save output files with specific time intervals and variables.

    Attributes:
        output_files (list[OutputFile]): List of output files to be managed by the saving strategy.

    Methods:
        get_time_intervals_to_save(start_time: datetime | None, end_time: datetime | None)
            -> list[tuple[datetime, datetime]]:
            Abstract method to determine the time intervals for saving data between start_time and end_time.

        get_file_path(interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
            Abstract method to generate the file path for a given time interval and output file.

        standardize_variable(variable: Variable, name_in_file: str) -> Variable:
            Abstract method to standardize a variable before saving, possibly renaming or formatting it.

        get_target_variables(output_file: OutputFile, variables_dict: dict[str, Variable], time_var: Variable | None,
                             start_time: datetime | None, end_time: datetime | None) -> dict[str, Variable] | None:
            Selects and prepares variables to be saved in the output file, optionally truncating them to a time range.

        save_single_file(file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False):
            Saves the provided dictionary to a file in the specified format (.mat, .pickle, .h5),
            optionally appending data.

        append_data(file_path: Path, dict_to_save: dict[str, Any]) -> dict[str, Any]:
            Abstract method to append data to an existing file; must be implemented by subclasses.
    """

    output_files:list[OutputFile]

    @abstractmethod
    def get_time_intervals_to_save(self,
                                   start_time:datetime|None,
                                   end_time:datetime|None) -> list[tuple[datetime, datetime]]:
        """Generates a list of time intervals to save between the specified start and end times.

        Args:
            start_time (datetime | None): The starting datetime for the intervals.
                                          If None, intervals may start from the earliest available time.
            end_time (datetime | None): The ending datetime for the intervals.
                                        If None, intervals may end at the latest available time.

        Returns:
            list[tuple[datetime, datetime]]: A list of tuples, each representing a time interval (start, end)
                                             to be saved.
        """

    @abstractmethod
    def get_file_path(self,
                      interval_start:datetime,
                      interval_end:datetime,
                      output_file:OutputFile) -> Path:
        """Generates a file path for saving variables based on the provided interval and output file information.

        Args:
            interval_start (datetime): The start of the interval for which the file is being generated.
            interval_end (datetime): The end of the interval for which the file is being generated.
            output_file (OutputFile): An OutputFile containing the name of the output file,
                                      and which variables should be saved in this file.

        Returns:
            Path: The generated file path where the output data should be saved.
        """

    @abstractmethod
    def standardize_variable(self, variable:Variable, name_in_file:str) -> Variable:
        """Standardizes the given variable according to the specified name in the file.

        Standardization may include checking of units, dimensions, and size consistency.

        Args:
            variable (Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it appears in the file.

        Returns:
            Variable: The standardized variable instance.
        """

    def get_target_variables(self,
                             output_file:OutputFile,
                             variables_dict:dict[str,Variable],
                             time_var:Variable|None,
                             start_time:datetime|None,
                             end_time:datetime|None) -> dict[str,Variable]|None:
        """Retrieves and processes target variables for saving based on the specified output file.

        Parameters:
            output_file (OutputFile): The output file configuration containing variable names to save.
            variables_dict (dict[str, Variable]): Dictionary mapping variable names to Variable objects.
            time_var (Variable | None): The time variable used for truncation, if applicable.
            start_time (datetime | None): The start time for truncating variables, if specified.
            end_time (datetime | None): The end time for truncating variables, if specified.

        Returns:
            dict[str, Variable] | None:
                - A dictionary of processed Variable objects keyed by their names,
                    or None if any specified variable name is not found in variables_dict.

        Notes:
            - If no variable names are specified in output_file, all variables in variables_dict are processed.
            - Variables are deep-copied before processing.
            - Each variable is standardized using the `standardize_variable` method.
            - If a requested variable name is not found, a warning is issued and None is returned.
        """
        target_variables:dict[str,Variable] = {}

        # if no variables have been specified, we save all of them
        if len(output_file.names_to_save) == 0:
            for key, var in variables_dict.items():
                var_to_save = deepcopy(var)

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                var_to_save = self.standardize_variable(var_to_save, key)

                target_variables[key] = var_to_save

            return target_variables

        for name_to_save in output_file.names_to_save:

            if name_to_save in variables_dict:
                var_to_save = deepcopy(variables_dict[name_to_save])

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                var_to_save = self.standardize_variable(var_to_save, name_to_save)

                target_variables[name_to_save] = var_to_save
            else:
                warnings.warn(f"Could not find target variable {name_to_save}!", stacklevel=2)
                return None

        return target_variables

    def save_single_file(self, file_path:Path, dict_to_save:dict[str,Any], *, append:bool=False) -> None:
        """Saves variable data to a single file in one of the supported formats (.mat, .pickle, .h5).

        Parameters:
            file_path (Path): The path to the file where the dictionary will be saved.
                              The file extension determines the format.
            dict_to_save (dict[str, Any]): The dictionary containing variable data to save.
            append (bool, optional): If True and the file exists, appends data to the existing file (if supported).
                                     Defaults to False.

        Raises:
            NotImplementedError: If the file format specified by the file extension is not supported.

        Supported formats:
            - .mat: Saves using scipy.io.savemat.
            - .pickle: Saves using pickle.dump.
            - .h5: Saves using h5py, with each key as a dataset (excluding "metadata").
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = file_path.suffix.lower()

        if file_path.exists() and append:
            dict_to_save = self.append_data(file_path, dict_to_save)

        if format_name == ".mat":
            # Save the dictionary into a .mat file
            savemat(str(file_path), dict_to_save)

        elif format_name == ".pickle":

            with file_path.open("wb") as file:
                pickle.dump(dict_to_save, file)

        elif format_name == ".h5":
            with h5py.File(file_path, "w") as file:
                for key, value in dict_to_save.items():
                    if key == "metadata":
                        continue
                    file.create_dataset(key, data=value, compression="gzip") # type: ignore[reportUnknownMemberType]

        else:
            msg = f"The '{format_name}' format is not implemented."
            raise NotImplementedError(msg)

    def append_data(self, file_path:Path, dict_to_save:dict[str,Any]) -> dict[str,Any]:
        """Appends variable data from the specified file to the provided dictionary.

        Args:
            file_path (Path): The path to the file where data should be appended.
            dict_to_save (dict[str, Any]): The dictionary containing data to append.

        Returns:
            dict[str, Any]: The updated dictionary after appending data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        msg = "This has to be overwritten for each Strategy!"
        raise NotImplementedError(msg)
