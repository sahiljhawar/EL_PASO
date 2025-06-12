from __future__ import annotations

from datetime import datetime
from pathlib import Path

from el_paso import Variable
from el_paso.saving_strategy import OutputFile, SavingStrategy


class SingleFileStrategy(SavingStrategy):

    map_standard_name:dict[str,str]
    output_files:list[OutputFile]

    file_path:Path

    def __init__(self, file_path:str|Path) -> None:

        self.file_path = Path(file_path)
        self.output_files = [OutputFile(self.file_path.name, [])]

        self.map_standard_name = {}

    def get_time_intervals_to_save(self, start_time:datetime, end_time:datetime) -> list[tuple[datetime, datetime]]:
        return [(start_time, end_time)]

    def get_file_path(self, interval_start:datetime, interval_end:datetime, output_file:OutputFile) -> Path:
        return self.file_path

    def standardize_variable(self, variable: Variable, name_in_file: str) -> Variable:
        return variable
