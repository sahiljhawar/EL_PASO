# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any

import netCDF4 as nC
import numpy as np

import el_paso as ep
from el_paso.saving_strategies.monthly_h5_strategy import MonthlyH5Strategy
from el_paso.saving_strategy import OutputFile

if typing.TYPE_CHECKING:
    from datetime import datetime

    from el_paso.data_standard import DataStandard
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral

logger = logging.getLogger(__name__)


class MonthlyNetCDFStrategy(MonthlyH5Strategy):
    """A saving strategy that saves data to monthly NetCDF files.

    This strategy organizes and saves processed scientific data into a series of
    NetCDF files, partitioned by month. It inherits from `MonthlyH5Strategy` but
    overrides the file saving logic to use the NetCDF format, which is widely used
    in climate and earth science for storing array-oriented scientific data.

    The strategy standardizes variables based on a provided `DataStandard` and
    structures the output files with a consistent naming convention that includes
    the file stem, date range, and magnetic field models used. It supports
    multiple magnetic field models and automatically configures the output files
    and their dependencies.
    """

    output_files: list[OutputFile]

    file_path: Path
    dependency_dict: dict[str, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        file_name_stem: str,
        mag_field: MagneticFieldLiteral | list[MagneticFieldLiteral],
        data_standard: DataStandard | None = None,
        root_metadata: dict[str, str] | None = None,
    ) -> None:
        """Initializes the monthly NetCDF saving strategy.

        Parameters:
            base_data_path (str | Path): The base directory where the output NetCDF files will be saved.
            file_name_stem (str): The base name for the output files (e.g., "my_data").
            mag_field (MagneticFieldLiteral | list[MagneticFieldLiteral]):
                A string or list of strings specifying the magnetic field models used.
            data_standard (DataStandard | None):
                An optional `DataStandard` instance to use for standardizing variables.
                If `None`, `ep.data_standards.PRBEMStandard` is used by default.
        """
        if isinstance(mag_field, str):
            mag_field = [mag_field]

        if data_standard is None:
            data_standard = ep.data_standards.PRBEMStandard()

        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field_list = mag_field
        self.standard = data_standard
        self.root_metadata = root_metadata

        output_file_entries = [
            "time",
            "flux/FEDU",
            "flux/FEDO",
            "flux/FEIU",
            "flux/alpha_eq",
            "flux/energy",
            "flux/alpha_local",
            "position/xGEO",
            "psd/PSD",
            "density/density_local",
        ]

        for single_mag_field in self.mag_field_list:
            output_file_entries.extend(
                [
                    f"position/{single_mag_field}/MLT",
                    f"position/{single_mag_field}/R0",
                    f"position/{single_mag_field}/Lstar",
                    f"position/{single_mag_field}/Lm",
                    f"mag_field/{single_mag_field}/B_eq",
                    f"mag_field/{single_mag_field}/B_local",
                    f"psd/{single_mag_field}/inv_mu",
                    f"psd/{single_mag_field}/inv_K",
                    f"density/{single_mag_field}/density_eq",
                ]
            )
        self.output_files = [
            OutputFile("full", output_file_entries, save_incomplete=True),
        ]

        self.dependency_dict = {
            "time": ["time"],
            "flux/FEDU": ["time", "energy", "alpha"],
            "flux/FEDO": ["time", "energy"],
            "flux/FEIU": ["time", "energy", "alpha"],
            "flux/alpha_eq": ["time", "alpha"],
            "flux/energy": ["time", "energy"],
            "flux/alpha_local": ["time", "alpha"],
            "position/xGEO": ["time", "xGEO_components"],
            "psd/PSD": ["time", "energy", "alpha"],
            "density/density_local": ["time"],
        }

        for single_mag_field in mag_field:
            self.dependency_dict |= {
                f"position/{single_mag_field}/MLT": ["time"],
                f"position/{single_mag_field}/R0": ["time"],
                f"position/{single_mag_field}/Lstar": ["time", "alpha"],
                f"position/{single_mag_field}/Lm": ["time", "alpha"],
                f"mag_field/{single_mag_field}/B_eq": ["time"],
                f"mag_field/{single_mag_field}/B_local": ["time"],
                f"psd/{single_mag_field}/inv_mu": ["time", "energy", "alpha"],
                f"psd/{single_mag_field}/inv_K": ["time", "alpha"],
                f"density/{single_mag_field}/density_eq": ["time"],
            }

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generates the file path for a monthly NetCDF file.

        The file name is constructed from the `file_name_stem`, the date range of the interval,
        and the specified magnetic field models, with a `.nc` extension.

        Parameters:
            interval_start (datetime): The start of the time interval.
            interval_end (datetime): The end of the time interval.
            output_file (OutputFile): The configuration for the output file.

        Returns:
            Path: The full file path for the NetCDF file.
        """
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = f"{self.file_name_stem}_{start_year_month_day}to{end_year_month_day}"

        for mag_field in self.mag_field_list:
            file_name += f"_{mag_field}"

        file_name += ".nc"

        return self.base_data_path / file_name

    def standardize_variable(
        self, variable: ep.Variable, name_in_file: str, *, first_call_of_interval: bool
    ) -> ep.Variable:
        """Standardizes a variable based on the configured `DataStandard`.

        This method delegates the standardization process to a `DataStandard` instance,
        ensuring that the variable's units and dimensions are consistent with the
        defined standard.

        Parameters:
            variable (ep.Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it will appear in the file.
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            ep.Variable: The standardized variable.
        """
        return self.standard.standardize_variable(
            name_in_file, variable, reset_consistency_check=first_call_of_interval
        )

    def append_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
        """Append only the new time slice to an existing NetCDF file.

        This avoids rewriting the whole file by opening it in append mode and writing into
        `[start_index:end_index, ...]` for each variable.

        Parameters:
            file_path (Path): The path to the existing NetCDF file to which data will be appended.
            data_dict_to_save (dict[str, Any]): The dictionary containing variable data to append. Must include a "time"
            key with the new time slice.

        Returns:
            dict[str, Any]: The same `data_dict_to_save` that was passed in, for consistency with the method signature.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the "time" key is missing from `data_dict_to_save`.
            ValueError: If the existing NetCDF file does not have an unlimited "time" dimension.

        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        if "time" not in data_dict_to_save:
            msg = "Cannot append: missing 'time' in data_dict_to_save."
            raise KeyError(msg)

        # NetCDF4 requires the dimension to be created as "unlimited" (size=None).
        with nC.Dataset(file_path, "a", format="NETCDF4") as file:
            time_dim = file.dimensions.get("time")
            if time_dim is None or not time_dim.isunlimited():
                msg = (
                    "Cannot append: the existing NetCDF file does not have an "
                    "unlimited 'time' dimension. Recreate the file with 'time' "
                    "created as unlimited (None)."
                )
                raise ValueError(msg)

            new_time = data_dict_to_save["time"]
            new_time_len = int(new_time.shape[0])
            if new_time_len == 0:
                return data_dict_to_save

            start_index = int(time_dim.size)
            end_index = start_index + new_time_len

            metadata_dict = typing.cast("dict[str, Any]", data_dict_to_save.get("metadata", {}))

            for path, value in data_dict_to_save.items():
                if path == "metadata":
                    continue
                if getattr(value, "size", 0) == 0:
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierachy: nC.Group | nC.Dataset = file
                for group in groups:
                    if group not in curr_hierachy.groups:
                        curr_hierachy = curr_hierachy.createGroup(group)  # type: ignore[reportUnknownVariableType]
                    else:
                        curr_hierachy = typing.cast("nC.Group", curr_hierachy[group])

                data_set = curr_hierachy.variables[dataset_name]
                # All variables defined in this strategy depend on the 'time' dimension first.
                if path == "time":
                    data_set[start_index:end_index, ...] = np.squeeze(value)
                else:
                    data_set[start_index:end_index, ...] = value

                if path in metadata_dict:
                    metadata = typing.cast("dict[str, Any]", metadata_dict[path])
                    data_set.units = metadata["unit"]
                    data_set.source = metadata["source_files"]
                    data_set.history = metadata["processing_notes"]
                    data_set.description = metadata["description"]

        return data_dict_to_save

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:  # noqa: C901, PLR0912
        """Saves a dictionary of variables to a single NetCDF file.

        This method creates a new NetCDF4 file, defines dimensions based on the data,
        and writes each variable as a dataset. It also attaches metadata as attributes
        to the datasets.

        Parameters:
            file_path (Path): The path to the file where the data will be saved.
            dict_to_save (dict[str, Any]): The dictionary containing variable data.
            append (bool, optional): If `True`, attempts to append data to an existing file.
                If the existing file has an unlimited `time` dimension, only the new time
                slice is appended.

        Note:
            For appending to work, the original file must have been created with an
            unlimited `time` dimension.
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and append:
            self.append_data(file_path, dict_to_save)
            return

        with nC.Dataset(file_path, "w", format="NETCDF4") as file:
            if self.root_metadata is not None:
                for key, value in self.root_metadata.items():
                    setattr(file, key, value)

            size_time = dict_to_save["time"].shape[0]
            if size_time == 0:
                logger.info(f"Skipping empty save for {file_path.name} (time has length 0).")
                return
            size_pitch_angle: int = 0
            size_energy: int = 0

            if "flux/alpha_eq" in dict_to_save and dict_to_save["flux/alpha_eq"].size > 0:
                size_pitch_angle = dict_to_save["flux/alpha_eq"].shape[1]
            elif "flux/alpha_local" in dict_to_save and dict_to_save["flux/alpha_local"].size > 0:
                size_pitch_angle = dict_to_save["flux/alpha_local"].shape[1]

            if "flux/energy" in dict_to_save and dict_to_save["flux/energy"].size > 0:
                size_energy = dict_to_save["flux/energy"].shape[1]

            # Make time unlimited so future runs can append without rewriting.
            file.createDimension("time", None)
            file.createDimension("alpha", size_pitch_angle)
            file.createDimension("energy", size_energy)

            if "position/xGEO" in dict_to_save and dict_to_save["position/xGEO"].size > 0:
                file.createDimension("xGEO_components", 3)

            for path, value in dict_to_save.items():
                if path == "metadata":
                    continue

                if value.size == 0:
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierachy = file
                for group in groups:
                    if group not in curr_hierachy.groups:
                        curr_hierachy = curr_hierachy.createGroup(group)  # type: ignore[reportUnknownVariableType]
                    else:
                        curr_hierachy = typing.cast("nC.Group", curr_hierachy[group])

                data_set = typing.cast(
                    "nC.Variable[Any]",
                    curr_hierachy.createVariable(  # type: ignore[reportUnknownMemberType]
                        dataset_name, "float64", self.dependency_dict[path], zlib=True, complevel=5, shuffle=True
                    ),
                )

                data_set[:, ...] = value

                if path in dict_to_save["metadata"]:
                    metadata = dict_to_save["metadata"][path]
                    data_set.units = metadata["unit"]
                    data_set.source = metadata["source_files"]
                    data_set.history = metadata["processing_notes"]
                    data_set.description = metadata["description"]
