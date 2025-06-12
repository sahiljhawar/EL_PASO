from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from el_paso.utils import enforce_utc_timezone


class TimeBinMethod(Enum):
    """Enum for time binning methods."""

    Mean = "Mean"
    NanMean = "NanMean"
    Median = "Median"
    NanMedian = "NanMedian"
    Merge = "Merge"
    NanMax = "NanMax"
    NanMin = "NanMin"
    NoBinning = "NoBinning"
    Repeat = "Repeat"

    def __call__(self, x:NDArray[np.float64]) -> NDArray[np.float64]:
        """Call the binning method on the provided data.

        :param x: Numpy array which is binned
        :type x: np.ndarray
        :return: A numpy array holding the binned data
        :rtype: np.ndarray
        """
        binned_array = None

        match self.value:
            case "Mean":
                binned_array = np.mean(x, axis=0)
            case "NanMean":
                binned_array = np.nanmean(x, axis=0)
            case "Median":
                binned_array = np.nanmedian(x, axis=0)
            case "NanMedian":
                binned_array = np.nanmedian(x, axis=0)
            case "Merge":
                binned_array = np.concatenate(x, axis=0)
            case "NanMax":
                binned_array = np.nanmax(x, axis=0)
            case "NanMin":
                binned_array = np.nanmin(x, axis=0)
            case "NoBinning":
                binned_array = x
            case "Repeat":
                binned_array = x

        return binned_array


@dataclass
class VariableMetadata:
    """A class holding the metadata of a variable.

    :param unit: The unit of the variable.
    :type unit: u.UnitBase
    :param original_cadence_seconds: The original cadence of the data in seconds.
    :type original_cadence_seconds: float
    :param source_files: The list of SourceFiles, which variable contains data from.
    :type source_files: list[SourceFile]
    :param description: The description of the variable explaining what kind of data this variable contains.
    :type description: str
    :param processing_notes: The processing notes of the variable explaining all steps done to achieve the final result.
    :type processing_notes: str
    :param standard_name: The name of the standard variable this variable complies to.
    :type standard_name: str
    """

    unit: u.UnitBase = u.dimensionless_unscaled
    original_cadence_seconds: float = 0
    source_files: list[str] = field(default_factory=list)
    description: str = ""
    processing_notes: str = ""
    standard_name: str = ""

    def __post_init__(self) -> None:
        self.processing_steps_counter = 1

    def add_processing_note(self, processing_note:str) -> None:

        processing_note = f"{self.processing_steps_counter}) {processing_note}\n"

        self.processing_notes += processing_note
        self.processing_steps_counter += 1

class Variable:
    """Variable class holding data and metadata."""

    __slots__ = "standard_name", "dependent_variables", "_data", "metadata", "backup_for_reset"

    _data:NDArray[np.float64]
    metadata:VariableMetadata

    def __init__(
        self,
        original_unit: u.UnitBase,
        data:NDArray[np.float64]|None = None,
        description: str = "",
        processing_notes: str = "",
        dependent_variables: list[Variable]|None = None,
    ) -> None:

        self._data = np.array([], dtype=np.float64) if data is None else data
        self.dependent_variables = dependent_variables if dependent_variables else []

        self.metadata = VariableMetadata(
            unit=original_unit,
            description=description,
            processing_notes=processing_notes,
        )

    def __repr__(self) -> str:
        return f"Variable holding {self._data.shape} data points with metadata: {self.metadata}"

    def convert_to_unit(self, target_unit:u.UnitBase|str) -> None:
        """Convert the data to a given unit.

        :param target_unit: The unit the data should be converted to.
        :type target_unit: u.UnitBase
        :raises ValueError: if the 'unit' attribute of the metadata has not been set
        """
        if self.metadata.unit is None:
            msg = f"Unit has not been set for this Variable! Standard name: {self.standard_name}"
            raise ValueError(msg)

        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        if self.metadata.unit != target_unit:
            data_with_unit = u.Quantity(self._data, self.metadata.unit)
            self._data = data_with_unit.to_value(target_unit)

            self.metadata.unit = target_unit

    def get_data(self, target_unit:u.UnitBase|str|None=None) -> NDArray[np.float64]:
        """Get the data of the variable.

        :return: The data of the variable.
        :rtype: NDArray[np.float64]
        """
        if target_unit is None:
            return self._data

        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        return (self._data * self.metadata.unit).to_value(target_unit)

    def set_data(self, data:NDArray[np.float64], unit:Literal["same"]|str|u.UnitBase) -> None:
        self._data = data

        if isinstance(unit, str):
            if unit != "same":
                self.metadata.unit = u.Unit(unit)
        elif isinstance(unit, u.UnitBase):
            self.metadata.unit = unit
        else:
            msg = "unit must be either a str or a astropy unit!"
            raise TypeError(msg)

    def transpose_data(self, seq: list[int]|tuple[int,...]):
        self._data = np.transpose(self._data, axes=seq)

    def apply_thresholds_on_data(self, lower_threshold: float = -np.inf, upper_threshold: float = np.inf):
        self._data = np.where((self._data > lower_threshold) & (self._data < upper_threshold), self._data, np.nan)

    def truncate(self, time_variable:Variable, start_time:float|datetime, end_time:float|datetime) -> None:

        if isinstance(start_time, datetime):
            start_time = enforce_utc_timezone(start_time).timestamp()
        if isinstance(end_time, datetime):
            end_time = enforce_utc_timezone(end_time).timestamp()

        if self._data.shape[0] != time_variable.get_data().shape[0]:
            msg = "Encountered length missmatch between variable and time variable!"
            raise ValueError(msg)

        time_var_data = time_variable.get_data(u.posixtime)

        self._data = self._data[(time_var_data >= start_time) & (time_var_data <= end_time)]
