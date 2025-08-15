from __future__ import annotations

import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.saving_strategies.monthly_h5_strategy import MonthlyH5Strategy
from el_paso.saving_strategy import ConsistencyCheck, OutputFile
from el_paso.utils import assert_n_dim


class MonthlyNetCDFStrategy(MonthlyH5Strategy):

    output_files:list[OutputFile]

    file_path:Path
    dependency_dict: dict[str,list[str]]

    def __init__(self,
                 base_data_path:str|Path,
                 file_name_stem:str,
                 mag_field:Literal["T89", "T96", "TS04"]) -> None:

        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field = mag_field

        self.output_files = [
            OutputFile("full", ["time",
                                "flux/FEDU", "flux/FEDO", "flux/alpha_eq", "flux/energy", "flux/alpha_local",
                                "position/xGEO", f"position/{mag_field}/MLT", f"position/{mag_field}/R0",
                                f"position/{mag_field}/Lstar", f"position/{mag_field}/Lm",
                                f"mag_field/{mag_field}/B_eq", f"mag_field/{mag_field}/B_local",
                                "psd/PSD", f"psd/{mag_field}/inv_mu", f"psd/{mag_field}/inv_K",
                                "density/density_local", f"density/{mag_field}/density_eq",
            ], save_incomplete=True),
        ]

        self.consistency_check = ConsistencyCheck()

        self.dependency_dict = {
            "time": ["time"],
            "flux/FEDU": ["time", "energy", "alpha"],
            "flux/FEDO": ["time", "energy"],
            "flux/alpha_eq": ["time", "alpha"],
            "flux/energy": ["time", "energy"],
            "flux/alpha_local": ["time", "alpha"],
            "position/xGEO": ["time"],
            f"position/{mag_field}/MLT": ["time"],
            f"position/{mag_field}/R0": ["time"],
            f"position/{mag_field}/Lstar": ["time", "alpha"],
            f"position/{mag_field}/Lm": ["time", "alpha"],
            f"mag_field/{mag_field}/B_eq": ["time"],
            f"mag_field/{mag_field}/B_local": ["time"],
            "psd/PSD": ["time"],
            f"psd/{mag_field}/inv_mu": ["time", "energy", "alpha"],
            f"psd/{mag_field}/inv_K": ["time", "alpha"],
            "density/density_local": ["time"],
            f"density/{mag_field}/density_eq": ["time"],
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

        elif name_in_file == f"position/{self.mag_field}/R0":
            variable.convert_to_unit(ep.units.RE)

            assert_n_dim(variable, 1, name_in_file)
            self.consistency_check.check_time_size(variable.get_data().shape[0], name_in_file)

        elif name_in_file in [f"position/{self.mag_field}/Lstar", f"position/{self.mag_field}/Lm"]:
            variable.convert_to_unit(u.dimensionless_unscaled)

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[1], name_in_file)

        elif name_in_file in [f"mag_field/{self.mag_field}/B_eq", f"mag_field/{self.mag_field}/B_local"]:
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

        elif name_in_file == f"psd/{self.mag_field}/inv_mu":
            variable.convert_to_unit(u.MeV/u.G) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_energy_size(shape[1], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[2], name_in_file)

        elif name_in_file == f"psd/{self.mag_field}/inv_K":
            variable.convert_to_unit(u.RE * u.G**0.5) # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)
            self.consistency_check.check_pitch_angle_size(shape[2], name_in_file)

        elif name_in_file in ["density/density_local", f"density/{self.mag_field}/density_eq"]:
            variable.convert_to_unit(u.cm**(-3))

            assert_n_dim(variable, 1, name_in_file)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], name_in_file)

        return variable
