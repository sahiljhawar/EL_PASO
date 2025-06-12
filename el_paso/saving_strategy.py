import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from el_paso.classes import Variable


class OutputFile(NamedTuple):
    name: str
    names_to_save: list[str]

class SavingStrategy(ABC):

    output_files:list[OutputFile]

    @abstractmethod
    def get_time_intervals_to_save(self, start_time:datetime, end_time:datetime) -> list[tuple[datetime, datetime]]:
        pass

    @abstractmethod
    def get_file_path(self,
                      interval_start:datetime,
                      interval_end:datetime,
                      output_file:OutputFile) -> Path:
        pass

    @abstractmethod
    def standardize_variable(self, variable:Variable, name_in_file:str) -> Variable:
        pass

    def get_target_variables(self,
                             output_file:OutputFile,
                             variables_dict:dict[str,Variable],
                             time_var:Variable,
                             start_time:datetime,
                             end_time:datetime) -> dict[str,Variable]:
        target_variables:dict[str,Variable] = {}

        # if no variables have been specified, we save all of them
        if len(output_file.names_to_save) == 0:
            for key, var in variables_dict.items():
                var_to_save = deepcopy(var)
                var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                target_variables[key] = var_to_save

            return target_variables

        for name_to_save in output_file.names_to_save:

            if name_to_save in variables_dict:
                var_to_save = deepcopy(variables_dict[name_to_save])
                var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                var_to_save = self.standardize_variable(var_to_save, name_to_save)

                target_variables[name_to_save] = var_to_save
            else:
                warnings.warn(f"Could not find target variable {name_to_save}!", stacklevel=2)
                return {}

        return target_variables
