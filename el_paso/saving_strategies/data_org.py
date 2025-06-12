import operator
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional

from astropy import units as u

from el_paso.classes.save_standard import OutputFile, SaveCadence, SaveStandard
from el_paso.classes.variable import Variable


class DataorgPMF(SaveStandard):
    def __init__(
        self,
        product_variable_names,
        mission: str = None,
        source: str = None,
        instrument: str = None,
        model: str = None,
        mfm: str = None,
        version: str = None,
        save_text_segments: List[str] = None,
        default_db: str = None,
        file_format: Literal[".mat", ".pickle"] = ".mat",
    ):
        """Initialize a DataorgPMF object.

        Args:
            mission (str): The mission associated with the save standard.
            source (str): The data source (satellite/instrument/model) associated with the save standard.
            instrument (str): The instrument associated with the save standard.
            model (str): The model associated with the save standard.
            mfm (str): The multi-function module associated with the save standard.
            version (str): The version of the save standard.
            save_text_segments (List[str]): The list of text segments used for saving files.
            default_db (str): The default database associated with the save standard.
            default_format (str): The default format for saving files.
        """
        super().__init__(
            mission,
            source,
            instrument,
            model,
            mfm,
            version,
            save_text_segments,
            default_db,
            file_format,
            product_variable_names,
        )

        variable_names_in_file = [
            "time",
            "Flux",
            "alpha_local",
            "alpha_eq_model",
            "energy_channels",
            "MLT",
            "Lstar",
            "Lm",
            "PSD",
            "xGEO",
            "InvMu",
            "InvK",
            "B_eq",
            "B_local",
            "R0",
            "density",
        ]

        assert all(
            [x in list(product_variable_names.keys()) for x in variable_names_in_file]
        ), f"Variable name(s) {[x for x in variable_names_in_file if x not in list(product_variable_names.keys())]} not provided! Add them to the 'varnames' argument."

        self.save_cadence = SaveCadence.MONTHLY

        # TODO fix
        self.output_files = [
            OutputFile("flux", ["time", "Flux"], [u.datenum, (u.cm**2 * u.s * u.sr * u.keV) ** (-1)]),
            OutputFile(
                "alpha_and_energy",
                ["time", "alpha_local", "alpha_eq_model", "energy_channels"],
                [u.datenum, u.dimensionless_unscaled, u.dimensionless_unscaled, u.MeV],
            ),
            OutputFile("mlt", ["time", "MLT"], [u.datenum, u.hour]),
            OutputFile("lstar", ["time", "Lstar"], [u.datenum, ""]),
            OutputFile("lm", ["time", "Lm"], [u.datenum, ""]),
            OutputFile("psd", ["time", "PSD"], [u.datenum, u.s**3 / (u.kg**3 * u.m**6)]),
            OutputFile("xGEO", ["time", "xGEO"], [u.datenum, u.RE]),
            OutputFile("invmu_and_invk", ["time", "InvMu", "InvK"], [u.datenum, u.MeV / u.G, u.RE * u.G ** (1 / 2)]),
            OutputFile("bfield", ["time", "B_eq", "B_local"], [u.datenum, u.G, u.G]),
            OutputFile("R0", ["time", "R0"], [u.datenum, u.RE]),
            OutputFile("density", ["time", "density"], [u.datenum, u.cm**(-3)]),
        ]

    def get_saved_file_name(
        self, start_time: datetime, end_time: datetime, output_file: OutputFile, external_text: Optional[str] = None) -> str:
        """Get the saved file name based on a time string, output type, and optional external text.

        Args:
            time_string (str): The time string used to generate the file name.
            output_type (str): The type of output for which the file name is being generated.
            external_text (str, optional): An optional external text to include in the file name.

        Returns:
            str: The generated file name.

        """
        start_year_month_day = start_time.strftime("%Y%m%d")
        end_year_month_day = end_time.strftime("%Y%m%d")

        file_folder_name = (
            f"{self.mission.upper()}/{self.save_text_segments[0]}/{self.save_text_segments[1]}/Processed_Mat_Files/"
        )

        if output_file.name in ["flux", "psd", "xGEO", "density"]:
            os.makedirs(file_folder_name, exist_ok=True)
            file_name = (
                f"{file_folder_name}"
                f"{self.save_text_segments[1]}_{self.instrument}_{start_year_month_day}to{end_year_month_day}_{output_file.name}_"
                f"{self.save_text_segments[5]}{self.file_format}"
            )
        elif output_file.name in ["alpha_and_energy", "lstar", "lm", "invmu_and_invk", "mlt", "bfield", "R0"]:
            os.makedirs(file_folder_name, exist_ok=True)
            file_name = (
                f"{file_folder_name}"
                f"{self.save_text_segments[1]}_{self.instrument}_{start_year_month_day}to{end_year_month_day}_"
                f"{output_file.name}_{self.save_text_segments[2]}_{self.save_text_segments[3]}_"
                f"{self.save_text_segments[4]}_{self.save_text_segments[5]}{self.file_format}"
            )
        else:
            raise ValueError(f"Output file name '{output_file.name}' is not supported.")

        return file_name


class DataorgNflux(SaveStandard):
    def __init__(
        self,
        mission: str = None,
        satellite: str = None,
        instrument: str = None,
        model: str = None,
        mfm: str = None,
        version: str = None,
        save_text_segments: List[str] = None,
        default_db: str = None,
        default_format: str = None,
    ):
        """
        Initializes a DataorgNflux object.

        Args:
            mission (str): The mission associated with the save standard.
            satellite (str): The satellite associated with the save standard.
            instrument (str): The instrument associated with the save standard.
            model (str): The model associated with the save standard.
            mfm (str): The multi-function module associated with the save standard.
            version (str): The version of the save standard.
            save_text_segments (List[str]): The list of text segments used for saving files.
            default_db (str): The default database associated with the save standard.
            default_format (str): The default format for saving files.
        """
        super().__init__(
            mission, satellite, instrument, model, mfm, version, save_text_segments, default_db, default_format
        )
        self.outputs = ["Nflux"]
        self.files = ["combined_flux_file"]
        self.file_variables = {
            "Nflux": [
                "Epoch_posixtime",
                "Energy_FEDU",
                "FEDU",
                "source_satellite",
                "xGEO",
                "PA_local_OBS_FEDU",
                "PA_eq_T89",
                "Lstar",
                "InvMu",
                "InvK",
                "FEDU",
                "PSD_FEDU",
                "B_eq",
                "B_loc",
                "MLT",
            ]
        }

    def get_saved_file_name(self, time_string: str, output_type: str, external_text: Optional[str] = None) -> str:
        """
        Get the saved file name based on a time string, output type, and optional external text.

        Args:
            time_string (str): The time string used to generate the file name.
            output_type (str): The type of output for which the file name is being generated.
            external_text (str, optional): An optional external text to include in the file name.

        Returns:
            str: The generated file name.
        """
        # Split time_string around the word "to"
        time_part = time_string.split("to")[0].strip()
        # Convert the first part from YYYYMMDD to datetime.datetime
        time_obj = datetime.strptime(time_part, "%Y%m%d")
        year_month = time_obj.strftime("%Y%m")
        year_month_day = time_obj.strftime("%Y%m%d")
        if output_type == "Nflux":
            file_folder_name = f"{self.save_text_segments[0]}/{year_month}"
            os.makedirs(file_folder_name, exist_ok=True)
            file_name = (
                f"{self.save_text_segments[0]}/{year_month}/{self.satellite}_{self.save_text_segments[1]}_"
                f"{year_month_day}_{self.save_text_segments[2]}_{self.save_text_segments[3]}.mat"
            )
        else:
            raise ValueError(f"Output type '{output_type}' is not supported.")

        return file_name

    def variable_mapping(self, in_variable: Variable) -> str:
        """
        Maps a variable to a specific database.

        Args:
            in_variable (Variable): The input variable name.

        Returns:
            str: The mapped variable name.
        """
        if in_variable.standard_name == "Energy_FEDU":
            out_var_name = "Energy"
        elif in_variable.standard_name == "PA_local_OBS_FEDU":
            out_var_name = "Pitch_Angles"
        elif in_variable.standard_name == "Epoch_posixtime":
            out_var_name = "newtime"
        elif in_variable.standard_name == "PA_local_T89":
            out_var_name = "alpha_loc"
        elif in_variable.standard_name == "PA_eq_T89":
            out_var_name = "alpha_eq"
        elif in_variable.standard_name == "Lstar_T89_irbem":
            out_var_name = "Lstar"
        elif in_variable.standard_name == "InvMu_T89":
            out_var_name = "InvMu"
        elif in_variable.standard_name == "InvK_T89":
            out_var_name = "InvK"
        elif in_variable.standard_name == "FEDU":
            out_var_name = "Flux"
        elif in_variable.standard_name == "PSD_FEDU":
            out_var_name = "PSD"
        elif in_variable.standard_name == "B_eq_T89":
            out_var_name = "B_eq"
        elif in_variable.standard_name == "B_local_T89":
            out_var_name = "B_loc"
        elif in_variable.standard_name == "MLT_T89":
            out_var_name = "MLT"
        elif in_variable.standard_name == "xGEO":
            out_var_name = "Position_GEO"
        else:
            if in_variable.standard_name is not None and in_variable.standard_name:
                out_var_name = in_variable.standard_name
            else:
                out_var_name = in_variable.workspace_name
        return out_var_name
