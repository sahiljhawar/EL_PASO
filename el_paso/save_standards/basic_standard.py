from datetime import datetime
from el_paso.classes.variable import Variable
from el_paso.classes.save_standard import SaveStandard
from typing import List, Optional, Dict

class BasicStandard(SaveStandard):
    def __init__(self, mission: str = None, source: str = None, instrument: str = None,
                 model: str = None, mfm: str = None,
                 version: str = None, save_text_segments: Dict[str, List[str]] = None, default_db: str = None,
                 default_format: str = None):
        """
        Initializes a RealtimeMat object.

        Args:
            mission (str): The mission associated with the save standard.
            source (str): The data source associated with the save standard.
            instrument (str): The instrument associated with the save standard.
            model (str): The model associated with the save standard.
            mfm (str): The multi-function module associated with the save standard.
            version (str): The version of the save standard.
            save_text_segments (List[str]): The list of text segments used for saving files.
            default_db (str): The default database associated with the save standard.
            default_format (str): The default format for saving files.
        """
        super().__init__(mission, source, instrument, model, mfm, version, save_text_segments,
                         default_db, default_format)
        self.outputs = ["output_file"]
        self.files = ["output_file"]
        self.file_variables = {"output_file": []}

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
        if 'to' in time_string:
            time_part2 = time_string.split("to")[1].strip()
            time_obj2 = datetime.strptime(time_part2, "%Y%m%d")
            year_month2 = time_obj2.strftime("%Y%m")
            year_month_day2 = time_obj2.strftime("%Y%m%d")
        # Convert the first part from YYYYMMDD to datetime.datetime
        time_obj = datetime.strptime(time_part, "%Y%m%d")
        year_month = time_obj.strftime("%Y%m")
        year_month_day = time_obj.strftime("%Y%m%d")

        if "YYYYMMDD" in external_text or "yyyymmdd" in external_text:
            yyyymmdd_str = year_month_day
            # Replace "yyyymmdd" or "YYYYMMDD" in external_text with the appropriate value
            file_name = external_text.replace("yyyymmdd", yyyymmdd_str).replace("YYYYMMDD", yyyymmdd_str).replace("YYYYMM", year_month)
        elif "YYYYMMD1" in external_text and "YYYYMMD2" in external_text and 'to' in time_string:
            file_name = external_text.replace("YYYYMMD1", year_month_day).replace("YYYYMMD2", year_month_day2).replace("YYYYMM", year_month2)
        else:
            raise ValueError(f"The externally provided file name does not have the appropriate placeholders in it!")

        return file_name

    def variable_mapping(self, in_variable: Variable) -> str:
        """
        Maps a variable to a specific database.

        Args:
            in_variable (Variable): The input variable name.

        Returns:
            str: The mapped variable name.
        """

        if in_variable.standard_name is not None and in_variable.standard_name and in_variable.product.use_standard:
            out_var_name = in_variable.standard_name
        else:
            out_var_name = in_variable.workspace_name
        return out_var_name
