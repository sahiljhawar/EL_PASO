from __future__ import annotations

import typing
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, overload

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.utils import enforce_utc_timezone

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


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

    def __call__(self, data:NDArray[np.generic]) -> NDArray[np.generic]:  # noqa: C901
        """Applyies the binning method to the provided data.

        Args:
            data (NDArray[np.generic]): The input data array to be binned or aggregated.

        Returns:
            NDArray[np.generic]: The resulting array after applying the selected binning or aggregation method.

        Raises:
            TypeError: If the selected binning method requires numeric types and the input data is not numeric.
        """
        binned_array:NDArray[np.generic]

        if self.value in ["Mean", "NanMean", "Median", "NanMedian", "NanMax", "NanMin"] \
            and not np.issubdtype(data.dtype, np.number):
                msg = f"{self.value} time bin method is only supported for numeric types!"
                raise TypeError(msg)

        match self.value:
            case "Mean":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.mean(data, axis=0)
            case "NanMean":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmean(data, axis=0)
            case "Median":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmedian(data, axis=0)
            case "NanMedian":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmedian(data, axis=0)
            case "Merge":
                binned_array = np.concatenate(data, axis=0)
            case "NanMax":
                binned_array = np.nanmax(data, axis=0)
            case "NanMin":
                binned_array = np.nanmin(data, axis=0)
            case "NoBinning":
                binned_array = data
            case "Repeat":
                binned_array = data

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
    source_files: list[str] = field(default_factory=list[str])
    description: str = ""
    processing_notes: str = ""
    standard_name: str = ""

    def __post_init__(self) -> None:
        """Initializes the processing_steps_counter attribute to 1 after the dataclass has been instantiated.

        This method is automatically called by the dataclass after the __init__ method.
        """
        self.processing_steps_counter = 1

    def add_processing_note(self, processing_note:str) -> None:
        """Adds a processing note to the metadata.

        The note is prefixed with the current processing steps counter and a newline character is appended.
        The processing steps counter is then incremented.

        Args:
            processing_note (str): The note to be added to the processing notes.
        """
        processing_note = f"{self.processing_steps_counter}) {processing_note}\n"

        self.processing_notes += processing_note
        self.processing_steps_counter += 1

class Variable:
    """Variable class holding data and metadata."""

    __slots__ = "_data", "metadata"

    _data:NDArray[np.generic]
    metadata:VariableMetadata

    def __init__(
        self,
        original_unit: u.UnitBase,
        data:NDArray[np.generic]|None = None,
        description: str = "",
        processing_notes: str = "",
    ) -> None:

        self._data = np.array([], dtype=np.generic) if data is None else data

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
        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        if self.metadata.unit != target_unit:
            data_with_unit = u.Quantity(self._data, self.metadata.unit)
            self._data = typing.cast("NDArray[np.generic]", data_with_unit.to_value(target_unit)) #type: ignore

            self.metadata.unit = target_unit

    @overload
    def get_data(self, target_unit:u.UnitBase|str) -> NDArray[np.floating|np.integer]:
        ...

    @overload
    def get_data(self, target_unit:None=None) -> NDArray[np.generic]:
        ...

    def get_data(self, target_unit:u.UnitBase|str|None=None) -> NDArray[np.generic]:
        """Get the data of the variable.

        :return: The data of the variable.
        :rtype: NDArray[np.generic]
        """
        if target_unit is None:
            return self._data

        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        if not np.issubdtype(self._data.dtype, np.number):
            msg = f"Unit conversion is only supported for numeric types! Encountered for variable {self}."
            raise TypeError(msg)

        return typing.cast("NDArray[np.generic]", u.Quantity(self._data, self.metadata.unit).to_value(target_unit)) #type: ignore

    def set_data(self, data:NDArray[np.generic], unit:Literal["same"]|str|u.UnitBase) -> None:
        self._data = data

        if isinstance(unit, str):
            if unit != "same":
                self.metadata.unit = u.Unit(unit)
        elif isinstance(unit, u.UnitBase): #type: ignore
            self.metadata.unit = unit
        else:
            msg = "unit must be either a str or a astropy unit!"
            raise TypeError(msg)

    def transpose_data(self, seq: list[int]|tuple[int,...]):
        self._data = np.transpose(self._data, axes=seq)

    def apply_thresholds_on_data(self, lower_threshold: float = -np.inf, upper_threshold: float = np.inf):

        if not np.issubdtype(self._data.dtype, np.number):
            msg = f"Thresholds are only supported for numeric types! Encountered for variable {self}."
            raise TypeError(msg)
        self._data = typing.cast("NDArray[np.number]", self._data)

        self._data = np.where((self._data > lower_threshold) & (self._data < upper_threshold), self._data, np.nan)

    def truncate(self, time_variable:Variable, start_time:float|datetime, end_time:float|datetime) -> None:

        if isinstance(start_time, datetime):
            start_time = enforce_utc_timezone(start_time).timestamp()
        if isinstance(end_time, datetime):
            end_time = enforce_utc_timezone(end_time).timestamp()

        if self._data.shape[0] != time_variable.get_data().shape[0]:
            msg = "Encountered length missmatch between variable and time variable!"
            raise ValueError(msg)

        time_var_data = time_variable.get_data(ep.units.posixtime)

        self._data = self._data[(time_var_data >= start_time) & (time_var_data <= end_time)]
