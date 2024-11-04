from typing import List, Optional, Dict
from abc import ABC, abstractmethod

class SaveStandard(ABC):
    def __init__(self, mission: str, source: str, instrument: str, model: str, mfm: str,
                 version: str, save_text_segments: Dict[str, List[str]], default_db: str, default_format: str,
                 varnames: Dict[str, str] = None, outputs: List[str] = [], files: Dict[str, List[str]] = {}, file_variables: Dict[str, List[str]] = []):
        """
        Initializes a SaveStandard object.

        Args:
            mission (str): The mission associated with the save standard.
            source (str): The satellite associated with the save standard.
            instrument (str): The instrument associated with the save standard.
            model (str): The model associated with the save standard.
            mfm (str): The magnetic field model associated with the save standard.
            version (str): The version of the save standard.
            save_text_segments (Dict[str, List[str]]): A dictionary of text segments used for saving files.
            default_db (str): The default database associated with the save standard.
            default_format (str): The default format for saving files.
            outputs (List[str]): A list of output types associated with the save standard.
            files (Dict[str, List[str]]): A dictionary mapping file types to file paths.
            file_variables (Dict[str, List[str]]): A dictionary mapping file types to variables.
        """
        self.mission = mission
        self.source = source
        self.instrument = instrument
        self.model = model
        self.mfm = mfm
        self.version = version
        self.save_text_segments = save_text_segments
        self.default_db = default_db
        self.default_format = default_format
        self.outputs = outputs
        self.files = files
        self.file_variables = file_variables
        self.varnames=varnames

    @abstractmethod
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
        pass

    @abstractmethod
    def variable_mapping(self, in_variable: str, in_db: str) -> str:
        """
        Map a variable to a specific database.

        Args:
            in_variable (str): The input variable name.
            in_db (str): The input database name.

        Returns:
            str: The mapped variable name.
        """
        pass
