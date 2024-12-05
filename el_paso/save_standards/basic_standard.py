from datetime import datetime
from el_paso.classes.variable import Variable
from el_paso.classes.save_standard import SaveStandard, OutputFile
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
        
        self.output_files = [
            OutputFile("everything", [], [])
        ]

    def get_saved_file_name(self, start_time:datetime, end_time:datetime, output_file: OutputFile, external_text: Optional[str] = None) -> str:
        """
        Get the saved file name based on a time string, output type, and optional external text.

        Args:
            time_string (str): The time string used to generate the file name.
            output_type (str): The type of output for which the file name is being generated.
            external_text (str, optional): An optional external text to include in the file name.

        Returns:
            str: The generated file name.
        """

        start_year_month_day = start_time.strftime("%Y%m%d")
        end_year_month_day = end_time.strftime("%Y%m%d")

        file_name = ''
        for text_segment in self.save_text_segments:
            file_name += text_segment + '_'

        file_name += start_year_month_day + 'to' + end_year_month_day + '.mat'

        return file_name
