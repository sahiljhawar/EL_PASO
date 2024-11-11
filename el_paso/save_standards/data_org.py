import os
from datetime import datetime
from el_paso.classes.variable import Variable
from el_paso.classes.save_standard import SaveStandard
from typing import List, Optional, Dict

class DataorgPMF(SaveStandard):
    def __init__(self, mission: str = None, source: str = None, instrument: str = None,
                 model: str = None, mfm: str = None,
                 version: str = None, save_text_segments: List[str] = None, default_db: str = None,
                 default_format: str = None, varnames = None):
        """
        Initializes a DataorgPMF object.

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
        super().__init__(mission, source, instrument, model, mfm, version, save_text_segments,
                         default_db, default_format, varnames)
        self.outputs = ["flux", "alpha_and_energy", "mlt", "lstar", "lm", "psd", "xGEO", "invmu_and_invk", "bfield", "R_eq"]
        self.files = ["flux", "alpha_and_energy", "mlt", "lstar", "lm", "psd", "xGEO", "invmu_and_invk", "bfield", "R_eq"]
        self.file_variables = {"flux": [f"{varnames['time']}", f"{varnames['Flux']}"],
                               "alpha_and_energy": [f"{varnames['time']}", f"{varnames['PA_local']}", f"{varnames['PA_eq']}", f"{varnames['Energy']}"],
                               "mlt": [f"{varnames['time']}", f"{varnames['MLT']}"],
                               "lstar": [f"{varnames['time']}", f"{varnames['Lstar']}"],
                               "lm": [f"{varnames['time']}", f"{varnames['Lm']}"],
                               "psd": [f"{varnames['time']}", f"{varnames['PSD']}"],
                               "xGEO": [f"{varnames['time']}", f"{varnames['xGEO']}"],
                               "R_eq": [f"{varnames['time']}", f"{varnames['R_eq']}"],
                               "bfield": [f"{varnames['time']}", f"{varnames['B_eq']}", f"{varnames['B_local']}"],
                               "invmu_and_invk": [f"{varnames['time']}", f"{varnames['InvMu']}", f"{varnames['InvK']}"]
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
        if 'to' in time_string:
            time_part2 = time_string.split("to")[1].strip()
            time_obj2 = datetime.strptime(time_part2, "%Y%m%d")
            year_month2 = time_obj2.strftime("%Y%m")
            year_month_day2 = time_obj2.strftime("%Y%m%d")
        else:
            year_month2 = year_month
            year_month_day2 = year_month_day

        file_folder_name = f"{self.mission.lower()}/{self.save_text_segments[0]}/{self.save_text_segments[1]}/Processed_Mat_Files/"

        if output_type in ["flux", "psd", "mlt", "xGEO"]:
            os.makedirs(file_folder_name, exist_ok=True)
            file_name = (f"{file_folder_name}"
                         f"{self.save_text_segments[1]}_{self.instrument}_{year_month_day}to{year_month_day2}_{output_type}_"
                         f"{self.save_text_segments[5]}.mat")
        elif output_type in ["alpha_and_energy", "lstar", "lm", "invmu_and_invk", "bfield", "R_eq"]:
            os.makedirs(file_folder_name, exist_ok=True)
            file_name = (f"{file_folder_name}"
                         f"{self.save_text_segments[1]}_{self.instrument}_{year_month_day}to{year_month_day2}_"
                         f"{output_type}_{self.save_text_segments[2]}_{self.save_text_segments[3]}_"
                         f"{self.save_text_segments[4]}_{self.save_text_segments[5]}.mat")
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
        if "Energy" in in_variable.standard_name:
            out_var_name = "Energy"
        elif "Epoch" in in_variable.standard_name:
            out_var_name = "time"
        elif "PA_local" in in_variable.standard_name:
            out_var_name = "alpha_local"
        elif "PA_eq" in in_variable.standard_name:
            out_var_name = "alpha_eq_model"
        elif "R_eq" in in_variable.standard_name:
            out_var_name = "R_eq"
        elif "Lstar" in in_variable.standard_name:
            out_var_name = "Lstar"
        elif "Lm" in in_variable.standard_name:
            out_var_name = "Lm"
        elif "InvMu" in in_variable.standard_name:
            out_var_name = "InvMu"
        elif "InvK" in in_variable.standard_name:
            out_var_name = "InvK"
        elif (in_variable.standard_name == "FPDU" or in_variable.standard_name == "FEDU" or
              in_variable.standard_name == "FHEDU" or in_variable.standard_name == "FODU"):
            out_var_name = "Flux"
        elif "PSD" in in_variable.standard_name:
            out_var_name = "PSD"
        elif "MLT" in in_variable.standard_name:
            out_var_name = "MLT"
        elif in_variable.standard_name == "xGEO":
            out_var_name = "xGEO"
        else:
            if in_variable.standard_name is not None and in_variable.standard_name:
                out_var_name = in_variable.standard_name
            else:
                out_var_name = in_variable.workspace_name
        return out_var_name

class DataorgNflux(SaveStandard):
    def __init__(self, mission: str = None, satellite: str = None, instrument: str = None,
                 model: str = None, mfm: str = None,
                 version: str = None, save_text_segments: List[str] = None, default_db: str = None,
                 default_format: str = None):
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
        super().__init__(mission, satellite, instrument, model, mfm, version, save_text_segments,
                         default_db, default_format)
        self.outputs = ["Nflux"]
        self.files = ["combined_flux_file"]
        self.file_variables = {"Nflux": ["Epoch_posixtime", "Energy_FEDU", "FEDU", "source_satellite",
                                                 "xGEO", "PA_local_OBS_FEDU", "PA_eq_T89", "Lstar", "InvMu", "InvK",
                                                 "FEDU", "PSD_FEDU", "B_eq", "B_loc", "MLT"]}

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
            file_name = (f"{self.save_text_segments[0]}/{year_month}/{self.satellite}_{self.save_text_segments[1]}_"
                         f"{year_month_day}_{self.save_text_segments[2]}_{self.save_text_segments[3]}.mat")
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
