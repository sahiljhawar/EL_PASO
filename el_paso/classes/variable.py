from __future__ import annotations

import os
import typing
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.orm import sessionmaker

from el_paso.metadata.models import StandardVariable
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

    def convert_to_standard_unit(self) -> None:
        """Convert the data to the units defined by the variable's standard."""
        if self.standard:
            self.convert_to_unit(self.standard.standard_unit)

    def convert_to_standard(self, standard_name:str) -> None:
      # Access database
        home_path = os.getenv("HOME")
        if home_path is None:
            raise ValueError("HOME environmental variable not set!")
        db_path = Path(home_path) / ".el_paso" / "metadata_database.db"
        table_name = "StandardVariable"

        standard_variable_info = self._get_standard_info_from_db(
            db_path, table_name, "standard_name", standard_name
        )

        # If database reading is successful, construct and store the relevant StandardVariable
        if standard_variable_info:
            standard_unit = u.Unit(standard_variable_info.get("standard_unit"))

            # Constructing the StandardVariable object
            standard = StandardVariable(
                id=standard_variable_info["id"],
                standard_id=standard_variable_info["standard_id"],
                variable_type=standard_variable_info["variable_type"],
                standard_name=standard_variable_info["standard_name"],
                standard_description=standard_variable_info.get("standard_description"),
                standard_notes=standard_variable_info.get("standard_notes"),
                standard_unit=standard_unit,
            )
        else:
            msg = f"Standard info could not be loaded for variable: {standard_name}"
            raise ValueError(msg)

        self.convert_to_unit(u.Unit(standard.standard_unit))

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

    def get_standard_info(self,target_name: str) -> Any:
        """Gets information (field name target_name) for this variable from the standard.

        Args:
            standard (str): The name of the standard.
            target_name (str): The name of the target attribute to retrieve.

        Returns:
            Any: The value of the target attribute.
        """

        # Access database
        db_path = f"{os.getenv('RT_SCRIPTS_DIR')}/Metadata/metadata_database.db"
        table_name = "StandardVariable"  # Replace with your actual table name
        standard_variable_info = self._get_standard_info_from_db(
            db_path, table_name, "standard_name", self.standard_name
        )
        # If database reading is successful, construct and store the relevant StandardVariable
        if standard_variable_info:
            # Constructing the StandardVariable object
            self.standard = StandardVariable(
                id=standard_variable_info["id"],
                standard_id=standard_variable_info["standard_id"],
                variable_type=standard_variable_info["variable_type"],
                standard_name=standard_variable_info["standard_name"],
                standard_description=standard_variable_info.get("standard_description"),
                standard_notes=standard_variable_info.get("standard_notes"),
                standard_unit=standard_variable_info.get("standard_unit"),
            )
            return getattr(self.standard, target_name, None)
        return None

    def _get_standard_info_from_db(self, db_path: str, table_name: str, column_name: str, filter_value: str) -> dict:
        """Helper method to get standard information from the database.

        Args:
            db_path (str): The path to the database.
            table_name (str): The name of the table.
            column_name (str): The name of the column to filter by.
            filter_value (str): The value to filter the column by.

        Returns:
            dict: The dictionary of the target attribute.
        """
        engine = create_engine(f"sqlite:///{db_path}")
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table = metadata.tables.get(table_name)

        if table is None:
            print(f"Table {table_name} does not exist.")
            return None

        session_engine = sessionmaker(bind=engine)
        session = session_engine()

        # Create a select statement for all columns in the table
        stmt = select(table).where(table.c[column_name] == filter_value)

        # Execute the query and fetch all results
        result = session.execute(stmt).mappings().first()

        session.close()

        return dict(result) if result else None

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
