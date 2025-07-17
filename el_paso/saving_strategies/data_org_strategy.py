from __future__ import annotations

import calendar
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from el_paso import Variable
from el_paso.saving_strategy import OutputFile, SavingStrategy


class DataOrgStrategy(SavingStrategy):

    output_files:list[OutputFile]

    file_path:Path

    def __init__(self,
                 base_data_path:str|Path,
                 mission:str,
                 satellite:str,
                 instrument:str,
                 kext:str,
                 file_format:Literal[".mat", ".pickle"]=".mat") -> None:

        self.base_data_path = Path(base_data_path)
        self.mission = mission
        self.satellite = satellite
        self.instrument = instrument
        self.kext = kext
        self.file_format = file_format

        self.output_files = [
            OutputFile("flux", ["time", "Flux"]),
            OutputFile("alpha_and_energy",["time", "alpha_local", "alpha_eq_model", "energy_channels"]),
            OutputFile("mlt", ["time", "MLT"]),
            OutputFile("lstar", ["time", "Lstar"]),
            OutputFile("lm", ["time", "Lm"]),
            OutputFile("psd", ["time", "PSD"]),
            OutputFile("xGEO", ["time", "xGEO"]),
            OutputFile("invmu_and_invk", ["time", "InvMu", "InvK"]),
            OutputFile("bfield", ["time", "B_eq", "B_local"]),
            OutputFile("R0", ["time", "R0"]),
            OutputFile("density", ["time", "density"]),
        ]

    def standardize_variable(self, variable: Variable, name_in_file: str) -> Variable:

        match name_in_file:
            case "time":
                variable.convert_to_unit(u.datenum)
                assert variable.get_data().ndim == 1
            case "Flux":
                variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1))
                assert variable.get_data().ndim == 3
            case "alpha_local":
                variable.convert_to_unit(u.radian)
                assert variable.get_data().ndim == 2
            case "alpha_eq_model":
                variable.convert_to_unit(u.radian)
                assert variable.get_data().ndim == 2
            case "energy_channels":
                variable.convert_to_unit(u.MeV)
                assert variable.get_data().ndim == 2
            case "MLT":
                variable.convert_to_unit(u.hour)
                assert variable.get_data().ndim == 1
            case "Lstar":
                variable.convert_to_unit(u.dimensionless_unscaled)
                assert variable.get_data().ndim == 2
            case "Lm":
                variable.convert_to_unit(u.dimensionless_unscaled)
                assert variable.get_data().ndim == 2
            case "xGEO":
                variable.convert_to_unit(u.RE)
                assert variable.get_data().ndim == 2
            case "B_eq":
                variable.convert_to_unit(u.nT)
                assert variable.get_data().ndim == 1
            case "B_local":
                variable.convert_to_unit(u.nT)
                assert variable.get_data().ndim == 1
            case "R0":
                variable.convert_to_unit(u.RE)
                assert variable.get_data().ndim == 1
            case "density":
                variable.convert_to_unit(u.cm**(-3))
                assert variable.get_data().ndim == 1
            case "PSD":
                variable.convert_to_unit((u.m * u.kg * u.m / u.s)**(-3))
                assert variable.get_data().ndim == 3
            case "InvMu":
                variable.convert_to_unit(u.MeV/u.G)
                assert variable.get_data().ndim == 3
            case "InvK":
                variable.convert_to_unit(u.RE * u.G**0.5)
                assert variable.get_data().ndim == 3
            case _:
                msg = f"Encountered invalid name_in_file: {name_in_file}!"
                raise ValueError(msg)

        return variable

    def get_time_intervals_to_save(self,
                                   start_time:datetime|None,
                                   end_time:datetime|None) -> list[tuple[datetime, datetime]]:
        time_intervals:list[tuple[datetime, datetime]] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for DataOrgStrategy!"
            raise ValueError(msg)

        current_time = start_time.replace(day=1)
        while current_time <= end_time:
            year = current_time.year
            month = current_time.month
            eom_day = calendar.monthrange(year, month)[1]

            month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
            month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=timezone.utc)
            time_intervals.append((month_start, month_end))
            current_time = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)

        return time_intervals

    def get_file_path(self, interval_start:datetime, interval_end:datetime, output_file:OutputFile) -> Path:

        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = f"{self.satellite.lower()}_{self.instrument.lower()}_{start_year_month_day}to{end_year_month_day}_{output_file.name}"

        if output_file.name in ["alpha_and_energy", "lstar", "lm", "invmu_and_invk", "mlt", "bfield", "R0"]:
            file_name += f"_n4_4_{self.kext}"

        file_name += "_ver4" + self.file_format

        return self.base_data_path / self.mission.upper() / self.satellite.lower() / "Processed_Mat_Files" / file_name

    def append_data(self, file_path:Path, data_dict_to_save:dict[str,Any]) -> dict[str, Any]:

        with file_path.open("rb") as file:
            data_dict_old = pickle.load(file)  # noqa: S301

            time_1 = np.squeeze(data_dict_old["time"])
            time_2 = np.squeeze(data_dict_to_save["time"])

            idx_to_insert = np.searchsorted(time_1, time_2[0])

            time_1_in_2 = np.squeeze(np.isin(time_1, time_2))

            for key, value_1 in data_dict_old.items():

                if key not in data_dict_to_save:
                    msg = "Key missmatch when concatenating data dicts!"
                    raise ValueError(msg)

                if isinstance(value_1, np.ndarray):
                    value_1 = value_1[~time_1_in_2]

                    value_2 = data_dict_to_save[key]

                    concatenated_value = value_2 if value_1.size == 0 else np.insert(value_1, idx_to_insert, value_2, axis=0)

                    if key == "time" and len(np.unique(concatenated_value)) != len(concatenated_value):
                        msg = "Time values were not unique when concatinating arrays!"
                        raise ValueError(msg)
                    data_dict_to_save[key] = concatenated_value

                elif isinstance(value_1, dict): # this is the metadata dict
                    continue

            return data_dict_to_save
