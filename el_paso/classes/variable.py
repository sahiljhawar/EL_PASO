from typing import List, Any
from pathlib import Path
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass

import numpy as np
import cdflib
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import sessionmaker
from astropy import units as u

from el_paso.metadata.models import StandardVariable, SaveVariable

class TimeBinMethod(Enum):
    Mean        = lambda x: np.mean(x, axis=0)
    NanMean     = lambda x: np.nanmean(x, axis=0)
    Median      = lambda x: np.median(x, axis=0)
    NanMedian   = lambda x: np.nanmedian(x, axis=0)
    Merge       = lambda x: np.concatenate(x, axis=0)
    MergeString = lambda x: '\n'.join(map(str, x.flatten()))
    NanMax      = lambda x: np.nanmax(x, axis=0)
    NanMin      = lambda x: np.nanmin(x, axis=0)
    NanMedian   = lambda x: np.min(x, axis=0)
    NoBinning   = lambda: np.nan
    Repeat      = 0

    def __call__(self, x):
        return self.value(x)

@dataclass
class VariableMetadata:
    unit:u.UnitBase = ''
    save_name:str = ''
    cadence_seconds:int = 0
    source_files:list = None
    description:str = ''
    processing_notes:str = ''
    time_bin_method:TimeBinMethod = TimeBinMethod.NoBinning
    time_bin_interval:timedelta = None

class Variable:
    def __init__(
            self,
            time_variable,
            original_unit: u.UnitBase,
            target_unit: u.UnitBase = None,
            time_bin_method: TimeBinMethod = TimeBinMethod.NoBinning,
            name_or_column_in_file: str = '',
            standard_name: str = '',
            dependent_variables: List[str] = None,
            description: str = '',
            processing_notes: str = ''
            ):
        
        """Initializes a Variable object.

        Args:
            workspace_name (str): The name the variable is loaded as in the workspace.
            data_content (Any, optional): The data content of the variable.
            product_name (str, optional): The name of the Product object the variable belongs to. Defaults to ''.
            name_or_column_in_file (Union[str, int], optional): The name or column the variable is stored at in
            the original files. Defaults to ''.
            standard_name (str, optional): Name of the variable in the chosen data standard. Defaults to ''.
            save_standard (str, optional): Name assigned to the variable by the chosen saving standard. Defaults to ''.
            internal_standard (str, optional): The internal data standard in use. Defaults to ''.
            current_units (str, optional): Units of the variable in the source files. Defaults to ''.
            source_files (List[str], optional): The list of source files used to produce the data content of this
            variable. Defaults to None.
            current_units (str, optional): The current units of the variable. Defaults to ''.
            description (str, optional): Description of the variable. Defaults to ''.
            source_notes (str, optional): Notes about this variable from the source files. Defaults to ''.
            standard_notes (str, optional): Notes about the variable from the chosen data standard. Defaults to ''.
            processing_notes (str, optional): Notes about this variable from the processing. Defaults to ''.
            original_cadence_seconds (int, optional): The original cadence of this variable in seconds.
                                                      Defaults to None.
            current_cadence_seconds (int, optional): The current cadence of this variable in seconds. Defaults to None.
            fill_method (str, optional): The method used to fill this variable during data gaps. Defaults to ''.
            time_bin_method_default (str, optional): The default time binning method for this variable. Defaults to ''.
            dependent_variables (List[str], optional): The list of variables this variable depends on. Defaults to None.
            related_variables (List[str], optional): List of variables this variable is related to. Defaults to None.
            time_bin_method_current (str, optional): Current time binning method. Defaults to None.
            time_bin_interval_current (str, optional): Current time binning interval. Defaults to None.
            speasy_tree (str, optional): Data tree for accessing this product through Speasy. Defaults to None.
            spase_description (str, optional): SPASE-compatible metadata description. Defaults to None.
            standard_variable (StandardVariable, optional): Content associated with this variable in
                                                            the chosen metadata database. Defaults to None.
            save_variable (SaveVariable, optional): Content associated with this variable in the chosen
                                                    save standard database. Defaults to None.
            header_length (str, optional): Number of header rows in original ASCII file. Defaults to None.
            columns (bool, optional): Whether original ASCII file contains column names. Defaults to None.
        """
        self.data_content = None
        self.name_or_column_in_file = name_or_column_in_file
        self.standard_name = standard_name
        self.time_variable = time_variable
        self.target_unit = target_unit if target_unit is not None else original_unit
        self.dependent_variables = dependent_variables if dependent_variables is not None else []

        self.metadata = VariableMetadata(unit=original_unit, time_bin_method=time_bin_method, description=description, processing_notes=processing_notes)

        # Access database
        db_path = Path(os.getenv('HOME'))/ '.el_paso' / 'metadata_database.db'
        table_name = "StandardVariable"

        standard_variable_info = self._get_standard_info_from_db(db_path, table_name, "standard_name",
                                                                    self.standard_name)
        # If database reading is successful, construct and store the relevant StandardVariable
        if standard_variable_info:
            # Constructing the StandardVariable object
            self.standard = StandardVariable(
                id=standard_variable_info['id'],
                standard_id=standard_variable_info['standard_id'],
                variable_type=standard_variable_info['variable_type'],
                standard_name=standard_variable_info['standard_name'],
                standard_description=standard_variable_info.get('standard_description'),
                standard_notes=standard_variable_info.get('standard_notes'),
                standard_unit=standard_variable_info.get('standard_unit')
            )
        else:
            self.standard = None

    def convert_to_target_unit(self):
        if self.metadata.unit != self.target_unit:
            data_with_unit = self.data_content * self.metadata.unit
            self.data_content = data_with_unit.to_value(self.target_unit)

            self.metadata.unit = self.target_unit

    def get_standard_info(self, standard: str, target_name: str) -> Any:
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
            standard_variable_info = self._get_standard_info_from_db(db_path, table_name, "standard_name",
                                                                     self.standard_name)
            # If database reading is successful, construct and store the relevant StandardVariable
            if standard_variable_info:
                # Constructing the StandardVariable object
                self.standard = StandardVariable(
                    id=standard_variable_info['id'],
                    standard_id=standard_variable_info['standard_id'],
                    variable_type=standard_variable_info['variable_type'],
                    standard_name=standard_variable_info['standard_name'],
                    standard_description=standard_variable_info.get('standard_description'),
                    standard_notes=standard_variable_info.get('standard_notes'),
                    standard_unit=standard_variable_info.get('standard_unit')
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
        engine = create_engine(f'sqlite:///{db_path}')
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

    def get_attribute(self, attr_name: str) -> Any:
        """Gets an attribute by name."""
        return getattr(self, attr_name)

    def get_item(self, item_name: str) -> Any:
        """Gets an item by name."""
        return self.__dict__.get(item_name)

    def set_item(self, item_name: str, value: Any) -> None:
        """Sets an item by name."""
        self.__dict__[item_name] = value

class TimeVariable(Variable):

    def __init__(
            self,
            original_unit: u.UnitBase,
            name_or_column_in_file: str = ''
            ):
        
        super().__init__(
            original_unit=original_unit,
            time_variable=None,
            target_unit=u.epoch_timestamp,
            time_bin_method=TimeBinMethod.NoBinning,
            name_or_column_in_file=name_or_column_in_file,
            standard_name='Epoch',
            dependent_variables=None)
        
class DerivedVariable(Variable):

    def __init__(
            self,
            standard_name: str,
            time_variable: TimeVariable,
            target_unit: u.UnitBase
    ):
        
        super().__init__(
            original_unit='',
            time_variable=time_variable,
            target_unit=target_unit,
            standard_name=standard_name,
            dependent_variables=None
        )