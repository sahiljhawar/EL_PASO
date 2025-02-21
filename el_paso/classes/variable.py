from __future__ import annotations

import os
import typing
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, List

import cdflib
import numpy as np
from astropy import units as u
from numpy.typing import NDArray
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.orm import sessionmaker

from el_paso.metadata.models import StandardVariable

if typing.TYPE_CHECKING:
    from el_paso.classes import SourceFile


class TimeBinMethod(Enum):
    """Enum for time binning methods."""

    Mean = "Mean"
    NanMean = "NanMean"
    Median = "Median"
    NanMedian = "NanMedian"
    Merge = "Merge"
    MergeString = "MergeString"
    NanMax = "NanMax"
    NanMin = "NanMin"
    NoBinning = "NoBinning"
    Repeat = 0

    def __call__(self, x:np.ndarray) -> np.ndarray:
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

    unit: u.UnitBase = ""
    original_cadence_seconds: float = 0
    source_files: list[SourceFile] = field(default_factory=list)
    description: str = ""
    processing_notes: str = ""
    standard_name: str = ""
    time_bin_method: TimeBinMethod = TimeBinMethod.NoBinning
    time_bin_interval: timedelta = None


class Variable:
    """Variable class holding data and metadata."""

    __slots__ = "time_variable", "name_or_column_in_file", "standard_name", \
                "dependent_variables", "data", "standard", "metadata", "backup_for_reset"

    data:NDArray[np.float64]
    metadata:VariableMetadata

    def __init__(
        self,
        time_variable: TimeVariable|None,
        original_unit: u.UnitBase,
        time_bin_method: TimeBinMethod = TimeBinMethod.NoBinning,
        name_or_column_in_file: str = "",
        standard_name: str = "",
        description: str = "",
        processing_notes: str = "",
        dependent_variables: list|None = None
    ) -> None:

        self.data = None
        self.name_or_column_in_file = name_or_column_in_file
        self.standard_name = standard_name
        self.time_variable = time_variable
        self.dependent_variables = dependent_variables if dependent_variables else []

        self.standard = None
        self.metadata = VariableMetadata(
            unit=original_unit,
            time_bin_method=time_bin_method,
            description=description,
            processing_notes=processing_notes,
        )

        # Access database
        db_path = Path(os.getenv("HOME")) / ".el_paso" / "metadata_database.db"
        table_name = "StandardVariable"

        if self.standard_name != "":
            standard_variable_info = self._get_standard_info_from_db(
                db_path, table_name, "standard_name", self.standard_name
            )

            # If database reading is successful, construct and store the relevant StandardVariable
            if standard_variable_info:
                standard_unit = u.Unit(standard_variable_info.get("standard_unit"))

                # Constructing the StandardVariable object
                self.standard = StandardVariable(
                    id=standard_variable_info["id"],
                    standard_id=standard_variable_info["standard_id"],
                    variable_type=standard_variable_info["variable_type"],
                    standard_name=standard_variable_info["standard_name"],
                    standard_description=standard_variable_info.get("standard_description"),
                    standard_notes=standard_variable_info.get("standard_notes"),
                    standard_unit=standard_unit,
                )
            else:
                msg = f"Standard info could not be loaded for variable: {self.standard_name}"
                raise ValueError(msg)

        self.backup_for_reset = self.get_slots_dict()

    def get_slots_dict(self) -> dict[str,Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__ if slot != "backup_for_reset"}

    def reset(self) -> None:
        """Reset the variable to its default state."""
        for key, value in self.backup_for_reset.items():
            # do not make a copy of the time variable as this variable is holding a reference
            if key == "time_variable":
                continue
            setattr(self, key, deepcopy(value))

    def convert_to_standard_unit(self) -> None:
        """Convert the data to the units defined by the variable's standard."""
        if self.standard:
            self.convert_to_unit(self.standard.standard_unit)

    def convert_to_unit(self, target_unit:u.UnitBase) -> None:
        """Convert the data to a given unit.

        :param target_unit: The unit the data should be converted to.
        :type target_unit: u.UnitBase
        :raises ValueError: if the 'unit' attribute of the metadata has not been set
        """
        if self.metadata.unit is None:
            msg = f"Unit has not been set for this Variable! Standard name: {self.standard_name}"
            raise ValueError(msg)

        if self.metadata.unit != target_unit:
            data_with_unit = self.data * self.metadata.unit
            self.data = data_with_unit.to_value(target_unit)

            self.metadata.unit = target_unit

    def get_standard_info(self,target_name: str) -> Any:
        """Gets information (field name target_name) for this variable from the standard.

        Args:
            standard (str): The name of the standard.
            target_name (str): The name of the target attribute to retrieve.

        Returns:
            Any: The value of the target attribute.
        """
        if self.standard is not None:
            return getattr(self.standard, target_name, None)
        else:
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

    def transpose_data(self, seq: Iterable[int]):
        self.data = np.transpose(self.data, seq)

    def apply_thresholds_on_data(self, lower_threshold: float = -np.inf, upper_threshold: float = np.inf):
        self.data = np.where((self.data > lower_threshold) & (self.data < upper_threshold), self.data, np.nan)

    def truncate(self, start_timestamp, end_timestamp):
        assert isinstance(start_timestamp, float)
        assert isinstance(end_timestamp, float)

        if self.time_variable is None:
            raise ValueError(f"Time variable has not been set for this Variable! Standard name: {self.standard_name}")

        self.data = self.data[(self.time_variable.data >= start_timestamp) & (self.time_variable.data <= end_timestamp)]


class TimeVariable(Variable):

    def __init__(
        self,
        original_unit: u.UnitBase,
        standard_name: str = "",
        name_or_column_in_file: str = "",
        do_time_binning: bool = True,
    ):
        super().__init__(
            original_unit=original_unit,
            time_variable=None,
            time_bin_method=TimeBinMethod.NoBinning,
            name_or_column_in_file=name_or_column_in_file,
            standard_name=standard_name,
        )

        self.do_time_binning = do_time_binning

    def truncate(self, start_timestamp, end_timestamp):
        assert isinstance(start_timestamp, float)
        assert isinstance(end_timestamp, float)

        self.data = self.data[(self.data >= start_timestamp) & (self.data <= end_timestamp)]


class DerivedVariable(Variable):
    def __init__(
        self,
        standard_name: str = "",
    ):
        super().__init__(
            original_unit="",
            time_variable=None,
            standard_name=standard_name,
        )
