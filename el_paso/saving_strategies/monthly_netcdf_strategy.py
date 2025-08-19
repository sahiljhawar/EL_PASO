from __future__ import annotations

import logging
import typing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import netCDF4 as nc
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.saving_strategies.monthly_h5_strategy import MonthlyH5Strategy
from el_paso.saving_strategy import ConsistencyCheck, OutputFile
from el_paso.utils import assert_n_dim

logger = logging.getLogger(__name__)

class MonthlyNetCDFStrategy(MonthlyH5Strategy):

    output_files:list[OutputFile]

    file_path:Path
    dependency_dict: dict[str,list[str]]

    def __init__(self,
                 base_data_path:str|Path,
                 file_name_stem:str,
                 mag_field:Literal["T89", "T96", "TS04"]|list[Literal["T89", "T96", "TS04"]]) -> None:

        if not isinstance(mag_field, list):
            mag_field = [mag_field]

        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field = mag_field

        output_file_entries = ["time", "flux/FEDU", "flux/FEDO", "flux/alpha_eq", "flux/energy", "flux/alpha_local",
                               "position/xGEO", "density/density_local"]

        for single_mag_field in mag_field:
            output_file_entries.extend([f"position/{single_mag_field}/MLT", f"position/{single_mag_field}/R0",
                                        f"position/{single_mag_field}/Lstar", f"position/{single_mag_field}/Lm",
                                        f"mag_field/{single_mag_field}/B_eq", f"mag_field/{single_mag_field}/B_local",
                                        f"psd/{single_mag_field}/inv_mu", f"psd/{single_mag_field}/inv_K",
                                        f"density/{single_mag_field}/density_eq"])
        self.output_files = [
            OutputFile("full", output_file_entries, save_incomplete=True),
        ]

        self.consistency_check = ConsistencyCheck()

        self.dependency_dict = {
            "time": ["time"],
            "flux/FEDU": ["time", "energy", "alpha"],
            "flux/FEDO": ["time", "energy"],
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

    def get_file_path(self, interval_start:datetime, interval_end:datetime, output_file:OutputFile) -> Path:

        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = f"{self.file_name_stem}_{start_year_month_day}to{end_year_month_day}.nc"

        return self.base_data_path / file_name

    def standardize_variable(self, variable: ep.Variable, name_in_file: str) -> ep.Variable:

        if name_in_file in ["flux/FEDU"]:
            variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1)) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[2], name_in_file)

        elif name_in_file == "flux/FEDO":
            variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1)) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)

        elif name_in_file in ["flux/alpha_local", "flux/alpha_eq"]:
            variable.convert_to_unit(u.deg)

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[1], name_in_file)

        elif name_in_file == "flux/energy":
            variable.convert_to_unit(u.MeV)

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)

        elif name_in_file == "position/xGEO":
            variable.convert_to_unit(ep.units.RE)

            assert_n_dim(variable, 2, name_in_file)
            self.consistency_check.check_time_size(variable.get_data().shape[0], name_in_file)

        elif name_in_file == f"position/{self.mag_field}/MLT":
            variable.convert_to_unit(u.hour)

            assert_n_dim(variable, 1, name_in_file)
            self.consistency_check.check_time_size(variable.get_data().shape[0], name_in_file)

        elif "R0" in name_in_file:
            variable.convert_to_unit(ep.units.RE)

            assert_n_dim(variable, 1, name_in_file)
            self.consistency_check.check_time_size(variable.get_data().shape[0], name_in_file)

        elif "Lstar" in name_in_file or "lm" in name_in_file:
            variable.convert_to_unit(u.dimensionless_unscaled)

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[1], name_in_file)

        elif "B_eq" in name_in_file or "B_local" in name_in_file:
            variable.convert_to_unit(u.nT)

            assert_n_dim(variable, 1, name_in_file)
            self.consistency_check.check_time_size(variable.get_data().shape[0], name_in_file)

        elif name_in_file == "psd/PSD":
            variable.convert_to_unit((u.m * u.kg * u.m / u.s)**(-3)) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[2], name_in_file)

        elif "inv_mu" in name_in_file:
            variable.convert_to_unit(u.MeV/u.G) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[2], name_in_file)

        elif "inv_K" in name_in_file:
            variable.convert_to_unit(u.RE * u.G**0.5) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[1], name_in_file)

        elif "density" in name_in_file:
            variable.convert_to_unit(u.cm**(-3))

            assert_n_dim(variable, 1, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)

        return variable

    def save_single_file(self, file_path:Path, dict_to_save:dict[str,Any], *, append:bool=False) -> None:  # noqa: C901
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

        if file_path.exists() and append:
            dict_to_save = self.append_data(file_path, dict_to_save)

        with nc.Dataset(file_path, "w", format="NETCDF4") as file:

            size_time = dict_to_save["time"].shape[0]
            size_pitch_angle:int = 0
            size_energy:int = 0

            if "flux/alpha_eq" in dict_to_save and dict_to_save["flux/alpha_eq"].size > 0:
                size_pitch_angle = dict_to_save["flux/alpha_eq"].shape[1]
            elif "flux/alpha_local" in dict_to_save and dict_to_save["flux/alpha_local"].size > 0:
                size_pitch_angle = dict_to_save["flux/alpha_local"].shape[1]

            if "flux/energy" in dict_to_save and dict_to_save["flux/energy"].size > 0:
                size_energy = dict_to_save["flux/energy"].shape[1]

            file.createDimension("time", size_time)
            file.createDimension("alpha", size_pitch_angle)
            file.createDimension("energy", size_energy)

            if "position/xGEO" in dict_to_save and dict_to_save["position/xGEO"].size > 0:
                file.createDimension("xGEO_components", 3)

            for path, value in dict_to_save.items():

                if path == "metadata":
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierachy = file
                for group in groups:
                    if group not in curr_hierachy.groups:
                        curr_hierachy = curr_hierachy.createGroup(group) # type: ignore[reportUnknownVariableType]
                    else:
                        curr_hierachy = typing.cast("nc.Group", curr_hierachy[group])

                data_set = typing.cast("nc.Variable[Any]", curr_hierachy.createVariable( # type: ignore[reportUnknownMemberType]
                    dataset_name,
                    "f4",
                    self.dependency_dict[path],
                    zlib=True, complevel=5, shuffle=True))
                if value.size > 0:
                    data_set[:,...] = value

                if path in dict_to_save["metadata"]:
                    metadata = dict_to_save["metadata"][path]
                    data_set.units = metadata["unit"]
                    data_set.source = metadata["source_files"]
                    data_set.history = metadata["processing_notes"]
                    data_set.description = metadata["description"]
