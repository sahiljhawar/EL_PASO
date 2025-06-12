from __future__ import annotations

import calendar
from datetime import datetime, timezone
from pathlib import Path

from astropy import units as u

from el_paso import Variable
from el_paso.saving_strategy import OutputFile, SavingStrategy


class DataOrgStrategy(SavingStrategy):

    output_files:list[OutputFile]

    file_path:Path

    def __init__(self, base_data_path:str|Path, mission:str, satellite:str, instrument:str, kext:str) -> None:

        self.base_data_path = Path(base_data_path)
        self.mission = mission
        self.satellite = satellite
        self.instrument = instrument
        self.kext = kext

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
                variable.convert_to_unit(u.dimensionless_angles)
                assert variable.get_data().ndim == 2
            case "alpha_eq_model":
                variable.convert_to_unit(u.dimensionless_angles)
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
            case _:
                msg = "Encountered invalid name_in_file!"
                raise ValueError(msg)

        return variable

    def get_time_intervals_to_save(self, start_time:datetime, end_time:datetime) -> list[tuple[datetime, datetime]]:
        time_intervals:list[tuple[datetime, datetime]] = []

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
            file_name += f"_{self.kext}"

        file_name += "_ver4.mat"

        return self.base_data_path / self.mission.upper() / self.satellite.lower() / "Processed_Mat_Files" / file_name
