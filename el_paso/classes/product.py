from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import itertools
import calendar
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from el_paso.classes import SaveStandard
    from el_paso.classes import Variable, TimeVariable

from el_paso.save_standards.data_org import DataorgPMF
from el_paso.utils import save_into_file
from el_paso.derived_variables import compute_PSD, get_local_B_field, construct_maginput, get_MLT, get_magequator, get_Lstar, compute_invariant_mu, compute_invariant_K, get_mirror_point

class Product(ABC):
    def __init__(
            self,
            irbem_lib_path:str,
            source_files: list = [],
            derived_variables: dict = {},
            save_cadence: str='daily',
            perform_time_binning:bool=False,
            time_binning_cadence:timedelta=None,
            default_save_standard: SaveStandard = DataorgPMF,
            saved_filename_extra_text: str = '',
            irbem_options = [1, 0, 0, 0, 0]
            ):
        """
        Initializes a Product object.

        Args:
            name (str): The name of the product.
            dataset (DataSet, optional): The DataSet object the product belongs to. Defaults to None.
            variables (List[Variable], optional): The list of Variable objects associated with the product.
                                                Defaults to None.
            instrument (str, optional): The instrument associated with the product. Defaults to None.
            internal_standard (str, optional): The internal data standard in use. Defaults to None.
            save_standard (str, optional): The saving standard used for the product. Defaults to None.
            save_paths (List[str], optional): The paths where the product is saved. Defaults to None.
            download_paths (List[str], optional): The paths where the source file product are downloaded.
                                                Defaults to None.
            speasy_tree (str, optional): The Speasy data tree for accessing this product. Defaults to None.
            filetype(str, optional): "daily" for daily files, "monthly" for monthly files
            spase_description (str, optional): SPASE-compatible metadata description for the product. Defaults to None.
            download_urls (List[str], optional): The URLs for downloading the source files for the product.
                                                Defaults to None.
            download_arguments_prefixes (List[str], optional): The arguments required for downloading the source files for
                                                the product.
                                                Defaults to None.
            download_arguments_suffixes (List[str], optional): The arguments required for downloading the source files for
                                                the product.
                                                Defaults to None.
            download_command (str, optional): The command used for downloading the product, if different from default.
                                                Defaults to None.
        """
        self.irbem_lib_path = irbem_lib_path
        self.source_files = source_files
        self.derived_variables = derived_variables
        self.default_save_standard = default_save_standard
        self.save_cadence = save_cadence
        self.perform_time_binning = perform_time_binning
        self.saved_filename_extra_text = saved_filename_extra_text
        self.irbem_options = irbem_options

        if self.perform_time_binning:
            if time_binning_cadence is None:
                raise ValueError("asdfasdf")
            self.time_binning_cadence = time_binning_cadence

        self.standardized_variables = {}

    @abstractmethod
    def process_original_file(self) -> None:
        """Processes the variables from the original files for the product."""
        pass

    @abstractmethod
    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""
        pass

    @abstractmethod
    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""
        pass

    def download(self, start_time:datetime, end_time:datetime) -> None:
        """Downloads the product data according to the specified standard."""

        for source_file in self.source_files:
            source_file.download(start_time, end_time)

    def _extract_variables(self, start_time:datetime, end_time:datetime):

        self.extracted_variables = {}

        for source_file in self.source_files:
            self.extracted_variables |= source_file.extract_variables(start_time, end_time) # merge dictionaries

    def _convert_units(self):
        for var in self.extracted_variables.values():
            var.convert_to_target_unit()

    def _time_bin(
            self,
            time_array: list[float]
            ) -> None:
        """Bins the time series data according to the specified inputs.

        Args:
            bin_method (str, optional): The method to use for binning. Defaults to self.time_bin_method_default.
            time_array (List[float], optional): The array of time points to bin the data by.
            time_variable (Variable, optional): Another variable to use for binning.
            index_set (np.ndarray, optional): Index set for grouping data.
            bin_cadence (int, optional): Cadence value for binning.
            default_bin_cadence (int, optional): Default cadence value for binning.
            start_time (float, optional): Start time for binning.
            end_time (float, optional): End time for binning.
        """
        from el_paso.classes import TimeVariable

        # Build index sets for every time variable
        index_sets = {}

        for key, var in self.extracted_variables.items():

            # do not update time variables, since we still need the data for binning the other variables
            if isinstance(var, TimeVariable):
                continue

            # built bins
            total_duration = time_array[-1] - time_array[0]

            # Generate datetime objects by stepping through the time range
            num_steps = int(total_duration // self.time_binning_cadence.total_seconds())  # Number of steps including start and end
            time_bins = [time_array[0], time_array[0] + self.time_binning_cadence.total_seconds()/2]
            for i in range(num_steps):
                current_time = time_bins[-1] + self.time_binning_cadence.total_seconds()
                time_bins.append(current_time)

            # Just repeat in case of no time dependency
            if var.time_variable is None:
                var.data_content = np.repeat(var.data_content[np.newaxis, ...], len(time_array), axis=0)
                continue

            # check if time variable and data content sizes match
            if var.data_content.shape[0] != len(var.time_variable.data_content):
                raise ValueError(f'Variable {key}: size of dimension 0 does not match length of time variable!')

            # calculate bin indices for given time array if it has not been calculated before
            if var.time_variable not in index_sets.keys():
                index_sets[var.time_variable] = np.digitize(var.time_variable.data_content, time_array)

            index_set = index_sets[var.time_variable]

            # Initialize binned_data as an array of np.nans with the same shape as self.data_content,
            # but with the length of the first dimension matching the length of time_array
            if var.data_content.dtype.kind in {'U', 'S', 'O'}:  # Check if the data is string or object type
                binned_data = np.full((len(time_array),), "", dtype=var.data_content.dtype)
            else:
                binned_data_shape = (len(time_array),) + var.data_content.shape[1:]
                binned_data = np.full(binned_data_shape, np.nan)

            # Iterate over unique indices
            unique_indices = np.unique(index_set)
            unique_indices = unique_indices[(unique_indices != -1) & (unique_indices != len(time_array))]

            for unique_index in unique_indices:

                bin_data = var.data_content[index_set == unique_index, ...]
                binned_value = var.metadata.time_bin_method(bin_data)

                # Update the relevant slice of binned_data
                binned_data[unique_index, ...] = binned_value

            # Update relevant metadata fields
            # Ensure binned_data works for both numeric and string data
            if isinstance(binned_data[0], str):
                var.data_content = np.array(binned_data, dtype=object)
            else:
                var.data_content = np.array(binned_data)

            # update metadata
            var.metadata.cadence_seconds = self.time_binning_cadence.total_seconds()

        # After we binned all other variables, we can set the time variables to the binned time array
        for var in self.extracted_variables.values():
            if isinstance(var, TimeVariable):
                var.data_content = time_array
                continue

    def compute_derived_variables(self):
        
        # collect magnetic_field results in this dictionary
        magnetic_field_results = {}
        maginput = construct_maginput(self.get_standard_variable('Epoch_posixtime').data_content)

        for var in self.derived_variables.values():

            # check if the value has been calculated already in the case of magnetic field calculations
            if var.standard_name in magnetic_field_results.keys():
                var.data_content, var.metadata.unit = magnetic_field_results[var.standard_name]
            else:
                magnetic_field_str = var.standard_name.split('_')[-1]

                if var.standard.variable_type == 'PhaseSpaceDensity':
                    print('\tCalculating phase space density ...')
                    var.data_content, var.metadata.unit = compute_PSD(self)
                    continue

                elif 'B_local' in var.standard_name:
                    magnetic_field_results |= self._get_B_local(var, magnetic_field_str, maginput)

                elif 'MLT' in var.standard_name:
                    magnetic_field_results |= self._get_MLT(var, magnetic_field_str, maginput)

                elif 'R_eq' in var.standard_name or 'B_eq' in var.standard_name:
                    magnetic_field_results |= self._get_B_eq_and_R_eq(var, magnetic_field_str, maginput)

                elif 'Lstar_' in var.standard_name:
                    magnetic_field_results |= self._get_L_star(var, magnetic_field_str, maginput)

                elif 'PA_eq' in var.standard_name:
                    magnetic_field_results |= self._get_pa_eq(var, magnetic_field_str, maginput, magnetic_field_results)

                elif 'invMu' in var.standard_name:
                    magnetic_field_results |= self._get_invariant_mu(var, magnetic_field_str, maginput, magnetic_field_results)
                    
                elif 'invK' in var.standard_name:
                    magnetic_field_results |= self._get_invariant_K(var, magnetic_field_str, maginput, magnetic_field_results)

            var.data_content, var.metadata.unit = magnetic_field_results[var.standard_name]


    def process(self,
                start_time: str = None,
                end_time: str = None,
                fieldmodel: str = None) -> None:
        """Processes the product data according to the specified standard."""

        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
            start_time = start_time.replace(tzinfo=timezone.UTC)
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d')
            end_time = end_time.replace(hour=23, minute=59, second=59, tzinfo=timezone.UTC)
        
        time_intervals_to_save = self.get_time_intervals_to_save(start_time, end_time)

        for time_interval in time_intervals_to_save:

            # STAGE 1: converting extracted variables to standard variables
            # After this stage, we are only working with standard variables anymore

            self._extract_variables(time_interval[0], time_interval[1])

            self._convert_units()

            self.process_original_file()

            # self.check_dimensions()

            if self.perform_time_binning:
                print("Starting time binning...")
                time_array = np.arange(time_interval[0].timestamp(), time_interval[1].timestamp(), self.time_binning_cadence.total_seconds())
                self._time_bin(time_array)
                print("Time binning done.")

            # STAGE 2: calculating of derived quantities
            # We only access variables by their standard name

            self.process_after_binning()

            if len(self.derived_variables) > 0:
                print("Starting calculation of derived variables...")
                self.compute_derived_variables()
                print("Calculation of derived variables done.")

            # if os.getenv('COMPUTE_INVARIANTS') and (hasattr(self, 'uses_invariants') and self.uses_invariants):
            #     print("Starting invariant calculation...")
            #     onera_lib_file = f"{os.getenv('IRBEM_CODE_DIR')}/libirbem.so"
            #     self.compute_adiabatic_invariants(fieldmodel=fieldmodel)
            #     print("Invariant calculation done.")
            # if hasattr(self, 'post_process') and callable(self.post_process):
            #     self.post_process(fieldmodel=fieldmodel)
            self.save(time_interval[0], time_interval[1])


    def get_time_intervals_to_save(self, start_time, end_time):
        
        time_intervals = []

        if self.save_cadence == 'daily':
            current_time = start_time
            while current_time <= end_time:
                day_start = datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0, tzinfo=timezone.utc)
                day_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59, tzinfo=timezone.utc)
                time_intervals.append([day_start, day_end])
                current_time += timedelta(days=1)
        elif self.save_cadence == 'monthly':
            current_time = start_time.replace(day=1)
            while current_time <= end_time:
                year = current_time.year
                month = current_time.month
                eom_day = calendar.monthrange(year, month)[1]

                month_start = datetime(year, month, 1, 0, 0, 0)
                month_end = datetime(year, month, eom_day, 23, 59, 59)
                time_intervals.append([month_start, month_end])
                if month == 12:
                    current_time = datetime(year + 1, 1, 1)
                else:
                    current_time = datetime(year, month + 1, 1)

        return time_intervals        

    def get_variables_by_type(self, type_name: str):
        """
        Retrieve a list of variables that match the specified variable type.

        Args:
            type_name (str): The type of variable to filter by.

        Returns:
            list: A list of variables that match the specified type.
        """

        return [variable for variable in (self.extracted_variables | self.derived_variables).values()
                if variable.standard is not None and variable.standard.variable_type == type_name]

    def compute_adiabatic_invariants(self, fieldmodel: str = "T89") -> None:
        """
        Runs adiabatic invariant calculation for all epoch-type variables in the workspace.
        Args:
            fieldmodel (str): B field model to calculate with, out of T89, T04s, OP77Q

        """
        # Find kext for IRBEM
        if fieldmodel == "T89":
            kext = 4
        elif fieldmodel == "T04s":
            kext = 11
        elif fieldmodel == "OP77Q":
            kext = 5
        else:
            raise ValueError(f"B field model {fieldmodel} has not been added to the code yet!")

        print('Loading maginput...')
        for time_var in self.get_variables_by_type('Epoch'):
            '''
            maginput = construct_maginput_basic(time_var.data_content,
                                                sw_path=os.getenv('RT_ACE_PROC_DIR'),
                                                kp_path=os.getenv('KP_READER_PATH'))
                                                '''
            maginput = construct_maginput_basic(time_var.data_content,
                                                sw_path=os.getenv('RT_ACE_PROC_DIR'),
                                                kp_path=self.kp_path, kp_type=self.kp_type)
            maginput = maginput.T

            position_var_geo = self.get_variable_for_epoch_by_string(time_var, 'GEO')
            PA_var = self.get_variable_for_epoch_by_string(time_var, 'PA_local')
            print('Maginput loaded.')
            print('Calculating adiabatic invariants...')
            returned_results = make_invariants_basic(
                maginput, time_var.data_content, position_var_geo.data_content, PA_var.data_content,
                verbose=False, debug=False, r_zero=1, num_cores=12,
                options=[1, 1, 4, 4, 0], kext=kext, sysaxes=1,
                onera_lib_file=None)
            self.variables.append(Variable(workspace_name='Lstar', name_or_column_in_file='',
                                           standard_name=f'Lstar_{fieldmodel}_irbem', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['Lstar']))
            self.variables.append(Variable(workspace_name='Lm', name_or_column_in_file='',
                                           standard_name=f'Lm_{fieldmodel}_irbem', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['Lm']))
            self.variables.append(Variable(workspace_name='Xj', name_or_column_in_file='',
                                           standard_name=f'invI_{fieldmodel}_irbem', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['Xj']))
            self.variables.append(Variable(workspace_name='alpha_loc', name_or_column_in_file='',
                                           standard_name=f'PA_local_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['alpha_loc']))
            self.variables.append(Variable(workspace_name='alpha_eq', name_or_column_in_file='',
                                           standard_name=f'PA_eq_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['alpha_eq']))
            self.variables.append(Variable(workspace_name='B_mirr', name_or_column_in_file='',
                                           standard_name='', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['B_mirr']))
            self.variables.append(Variable(workspace_name='B_eq', name_or_column_in_file='',
                                           standard_name=f'B_eq_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name],
                                           data_content=returned_results['B_eq']))
            self.variables.append(Variable(workspace_name='B_loc', name_or_column_in_file='',
                                           standard_name=f'B_local_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name],
                                           data_content=returned_results['B_loc']))
            self.variables.append(Variable(workspace_name='MLT', name_or_column_in_file='',
                                           standard_name=f'MLT_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name],
                                           data_content=returned_results['MLT']))
            self.variables.append(Variable(workspace_name='InvK', name_or_column_in_file='',
                                           standard_name=f'InvK_{fieldmodel}', source_units='', source_notes='',
                                           original_cadence_seconds=None, fill_method='', time_bin_method_default='',
                                           dependent_variables=[time_var.workspace_name, PA_var.workspace_name],
                                           data_content=returned_results['InvK']))

    def save(self, start_time, end_time, save_standard: SaveStandard = None) -> None:
        """Saves file corresponding to this product in the specified format"""

        if save_standard is None:
            save_standard = self.default_save_standard

        # Check and convert start_time if it's a string
        if isinstance(start_time, str):
            start_time_obj = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        else:
            start_time_obj = start_time  # Already a datetime object

        # Check and convert end_time if it's a string
        if isinstance(end_time, str):
            end_time_obj = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
        else:
            end_time_obj = end_time  # Already a datetime object

        start_time_str = start_time_obj.strftime("%Y%m%d")
        end_time_str = end_time_obj.strftime("%Y%m%d")

        time_string = f"{start_time_str}to{end_time_str}"

        for output_type in save_standard.outputs:
            file_name = Path(save_standard.get_saved_file_name(time_string, output_type, self.saved_filename_extra_text))
            # Extract the directory path from the file path
            directory = file_name.parent
            # Create the directories if they don't exist
            if directory:  # Only create directories if there's a directory path (not an empty string)
                directory.mkdir(exist_ok=True)

            if len(save_standard.file_variables[output_type]) == 0:
                if hasattr(self, 'target_variables') and self.target_variables:
                    target_variables = [self.get_variable_by_workspace_name(var_name) for var_name in self.target_variables]
                    save_into_file(file_name, target_variables)
                else:
                    target_variables = self.variables
                    print(f"Chosen save standard has no target variables, saving all variables into output file...")
                    save_into_file(file_name, target_variables)
                continue

            target_variables = []

            for key, var in (self.extracted_variables | self.derived_variables).items():
                if key in save_standard.file_variables[output_type]:
                    var.metadata.save_name = save_standard.variable_mapping(var)
                    target_variables.append(var)

            print([x.metadata.save_name for x in target_variables])
            print(save_standard.file_variables[output_type])

            if len(target_variables) == len(save_standard.file_variables[output_type]):                
                save_into_file(file_name, target_variables)
            else:
                warnings.warn(f"Saving attempted, but product is missing some required "
                                f"variables for output {output_type}! "
                                f"IGNORE if processing a product that saves multiple output files")

    def get_standard_variable(self, standard_name: str) -> Variable:

        for var in (self.extracted_variables | self.derived_variables).values():
            if var.standard_name == standard_name:
                return var
            
    def _get_L_star(self, var, magnetic_field_str, maginput):
        print('\tCalculating Lstar ...')
        assert self.get_standard_variable('PA_local').metadata.unit == ''
        pa_local = self.get_standard_variable('PA_local').data_content

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_Lstar(xGEO, var.time_variable.data_content, pa_local, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results
    
    def _get_B_local(self, var, magnetic_field_str, maginput):
        print('\tCalculating local magnetic field ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_local_B_field(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results

    def _get_MLT(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic local time ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_MLT(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str)

        return results

    def _get_B_eq_and_R_eq(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic field and radial distance at the equator ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_magequator(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results

    def _get_pa_eq(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating equatorial pitch angle ...')

        assert self.get_standard_variable('PA_local').metadata.unit == ''
        pa_local = self.get_standard_variable('PA_local').data_content

        results = magnetic_field_results

        if not ('B_eq_' + magnetic_field_str) in results.keys():
            results |= self._get_B_eq_and_R_eq(var, magnetic_field_str, maginput)
        if not ('B_local_' + magnetic_field_str) in results.keys():
            results |= self._get_B_local(var, magnetic_field_str, maginput)

        B_eq = results['B_eq_' + magnetic_field_str][0]
        B_local = results['B_local_' + magnetic_field_str][0]

        results[var.standard_name] = (np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:,np.newaxis]), '')

        return results
    
    def _get_invariant_mu(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating invariant mu ...')
        assert self.get_standard_variable('PA_local').metadata.unit == ''
        pa_local = self.get_standard_variable('PA_local').data_content

        energy_vars = self.get_variables_by_type('Energy')
        assert len(energy_vars) == 1, f'We assume that there is exactly ONE energy variable available for calculating invariant mu. Found: {len(energy_vars)}!'
        energy_var = energy_vars[0]

        assert energy_var.metadata.unit == u.MeV
        energy = energy_var.data_content

        species_char = energy_var.standard.standard_name[-3]

        results = magnetic_field_results
        if not ('B_local_' + magnetic_field_str) in results.keys():
            results |= self._get_B_local(var, magnetic_field_str, maginput)

        B_local = results['B_local_' + magnetic_field_str][0]

        results[var.standard_name] = (compute_invariant_mu(energy, pa_local, B_local, species_char), u.MeV/u.G)

        return results

    def _get_invariant_K(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating invariant K ...')

        results = magnetic_field_results

        if not ('B_mirr_' + magnetic_field_str) in results.keys():
            results |= self._get_B_mirr(var, magnetic_field_str, maginput)
        if not ('XJ_' + magnetic_field_str) in results.keys():
            results |= self._get_L_star(var, magnetic_field_str, maginput)

        B_mirr = results['B_mirr_' + magnetic_field_str][0]
        XJ = results['XJ_' + magnetic_field_str][0]

        results['invK_' + magnetic_field_str] = (compute_invariant_K(B_mirr, XJ), u.RE*u.G**0.5)
        
        return results

    def _get_B_mirr(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic field of mirror points ...')

        assert self.get_standard_variable('PA_local').metadata.unit == ''
        pa_local = self.get_standard_variable('PA_local').data_content

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_mirror_point(xGEO, var.time_variable.data_content, pa_local, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results
