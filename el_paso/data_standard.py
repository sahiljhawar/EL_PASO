# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache 2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import el_paso as ep


class DataStandard(ABC):

    @abstractmethod
    def standardize_variable(self, standard_name: str, variable:ep.Variable) -> ep.Variable:
        pass

class _SizeAttr(NamedTuple):
    name:str = ""
    size:int = 0

@dataclass
class ConsistencyCheck:
    len_time: _SizeAttr|None = None
    len_pitch_angle: _SizeAttr|None = None
    len_energy: _SizeAttr|None = None

    def check_time_size(self, provided_len_time:int, name_in_file:str) -> None:
        if self.len_time is None:
            self.len_time = _SizeAttr(name_in_file, provided_len_time)
        elif self.len_time.size != provided_len_time:
                msg = (f"Time length missmatch! Time length of variable {self.len_time.name}: {self.len_time.size}",
                       f"and of variable {name_in_file}: {provided_len_time}")
                raise ValueError(msg)

    def check_pitch_angle_size(self, provided_len_pitch_angle:int, name_in_file:str) -> None:
        if self.len_pitch_angle is None:
            self.len_pitch_angle = _SizeAttr(name_in_file, provided_len_pitch_angle)
        elif self.len_pitch_angle.size != provided_len_pitch_angle:
                msg = (f"Pitch angle length missmatch! Pitch angle length of variable {self.len_pitch_angle.name}:"
                       f"{self.len_pitch_angle.size} and of variable {name_in_file}: {provided_len_pitch_angle}")
                raise ValueError(msg)

    def check_energy_size(self, provided_len_energy:int, name_in_file:str) -> None:
        if self.len_energy is None:
            self.len_energy = _SizeAttr(name_in_file, provided_len_energy)
        elif self.len_energy.size != provided_len_energy:
                msg = (f"Energy length missmatch! Energy length of variable {self.len_energy.name}:"
                       f"{self.len_energy.size} and of variable {name_in_file}: {provided_len_energy}")
                raise ValueError(msg)
