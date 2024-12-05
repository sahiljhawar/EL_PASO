from __future__ import annotations
from datetime import datetime, timedelta, timezone
import calendar

import numpy as np
from astropy import units as u

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from el_paso.classes import SaveStandard
    from el_paso.classes import Variable

# from el_paso.save_standards.data_org import DataorgPMF
# from el_paso.derived_variables import compute_PSD, get_local_B_field, construct_maginput, get_MLT, get_magequator, get_Lstar, compute_invariant_mu, compute_invariant_K, get_mirror_point
# from el_paso.utils import timed_function

class Product():
    def __init__(
            self,
            irbem_lib_path:str,
            source_files: list = [],
            derived_variables: dict = {},
            save_cadence: str='daily',
            perform_time_binning:bool=False,
            time_binning_cadence:timedelta=None,
            save_standard: SaveStandard = DataorgPMF,
            saved_filename_extra_text: str = '',
            irbem_options = [1, 0, 0, 0, 0],
            num_cores=1
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
        self.save_standard = save_standard
        self.save_cadence = save_cadence
        self.perform_time_binning = perform_time_binning
        self.saved_filename_extra_text = saved_filename_extra_text
        self.irbem_options = irbem_options
        self.num_cores = num_cores

        if self.perform_time_binning:
            if time_binning_cadence is None:
                raise ValueError("asdfasdf")
            self.time_binning_cadence = time_binning_cadence

        self.custom_variables = {}

    # This function can be overloaded by the user
    def standardize_variables_custom(self) -> None:
        """Processes the variables from the original files for the product."""
        pass

    # This function can be overloaded by the user
    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""
        pass

    # This function can be overloaded by the user
    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""
        pass

    def download(self, start_time:datetime, end_time:datetime) -> None:
        """Downloads the product data according to the specified standard."""

        for source_file in self.source_files:
            source_file.download(start_time, end_time)

    def get_all_variables(self):
        return self.extracted_variables | self.custom_variables | self.derived_variables

    def _extract_variables(self, start_time:datetime, end_time:datetime):

        self.extracted_variables = {}

        for source_file in self.source_files:
            self.extracted_variables |= source_file.extract_variables(start_time, end_time) # merge dictionaries

    def _convert_to_standard_units(self):
        for key,var in self.extracted_variables.items():
            if var.metadata.unit and var.standard:
                var.convert_to_standard_unit()

    def _validate_dimensions(self):
        for key, var in self.extracted_variables.items():
            
            if var.standard:
                match var.standard.variable_type:
                    case 'Epoch':
                        assert var.data_content.ndim == 1, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data_content.ndim}!'
                    case 'Flux':
                        assert var.data_content.ndim == 3, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data_content.ndim}!'
                    case 'Energy':
                        assert var.data_content.ndim == 2, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data_content.ndim}!'
                    case 'PitchAngle':
                        assert var.data_content.ndim == 2, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data_content.ndim}!'
                    case 'Position':
                        assert var.data_content.ndim == 2, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data_content.ndim}!'
                        assert var.data_content.shape[1] == 3, f'Variable {key} with type {var.standard.variable_type} has wrong dimensions:!'

    @timed_function()
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

        for key, var in (self.extracted_variables | self.custom_variables).items():

            # do not update time variables, since we still need the data for binning the other variables
            if isinstance(var, TimeVariable):
                continue

            # built bins
            total_duration = time_array[-1] - time_array[0]

            # Generate datetime objects by stepping through the time range
            num_steps = int(total_duration // self.time_binning_cadence.total_seconds()) + 1 # Number of steps including start and end
            time_bins = [time_array[0] - self.time_binning_cadence.total_seconds()/2]
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
                index_set = np.digitize(var.time_variable.data_content, time_bins)
                index_set = np.where(index_set == len(time_bins), 0, index_set) # remove values before and beyond time array; -1 will be ignored later on
                index_sets[var.time_variable] = index_set-1 # shift indices by one to match time array

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

            for unique_index in unique_indices:
                if unique_index == -1:
                    continue  # skip data out of range

                bin_data = var.data_content[index_set == unique_index, ...]
                if len(bin_data) == 0 or not np.any(np.isfinite(bin_data)):
                    continue # no data found
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
            if isinstance(var, TimeVariable) and var.do_time_binning:
                var.data_content = time_array
                continue

    @timed_function('Calculating derived variables')
    def _compute_derived_variables(self):
        
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

                else:
                    continue

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

            print(f'Processing {str(time_interval[0])} to {str(time_interval[1])}...')

            # STAGE 1: converting extracted variables to standard variables
            # After this stage, we are only working with standard variables anymore

            print('Extracting variables...')
            self._extract_variables(time_interval[0], time_interval[1])

            print('Converting to standard units...')
            self._convert_to_standard_units()

            self.standardize_variables_custom()
            
            if self.perform_time_binning:
                print("Starting time binning...")
                time_array = np.arange(time_interval[0].timestamp(), time_interval[1].timestamp(), self.time_binning_cadence.total_seconds())
                self._time_bin(time_array)

            # STAGE 2: calculating of derived quantities
            # We only access variables by their standard name

            self.process_after_binning()

            self._validate_dimensions()

            if len(self.derived_variables) > 0:
                print("Starting calculation of derived variables...")
                self._compute_derived_variables()

            self.post_process(fieldmodel=fieldmodel)
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

                month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
                month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=timezone.utc)
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

        return [variable for variable in self.get_all_variables().values()
                if variable.standard is not None and variable.standard.variable_type == type_name]

    def save(self, start_time:datetime, end_time:datetime, save_standard: SaveStandard = None) -> None:
        """Saves file corresponding to this product in the specified format"""

        self.save_standard.save(start_time, end_time, self.get_all_variables())

    def get_standard_variable(self, standard_name: str) -> Variable:

        for var in self.get_all_variables().values():
            if var.standard_name == standard_name:
                return var

    @timed_function()  
    def _get_L_star(self, var, magnetic_field_str, maginput):
        print('\tCalculating Lstar ...')
        assert self.get_standard_variable('PA_local').metadata.unit == 'deg'
        pa_local = self.get_standard_variable('PA_local').data_content

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_Lstar(xGEO, var.time_variable.data_content, pa_local, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput, self.num_cores)

        return results
    
    @timed_function()
    def _get_B_local(self, var, magnetic_field_str, maginput):
        print('\tCalculating local magnetic field ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_local_B_field(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results

    @timed_function()
    def _get_MLT(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic local time ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_MLT(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str)

        return results
    
    @timed_function()
    def _get_B_eq_and_R_eq(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic field and radial distance at the equator ...')

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_magequator(xGEO, var.time_variable.data_content, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput)

        return results
    
    @timed_function('Calculating equatorial pitch angle')
    def _get_pa_eq(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating equatorial pitch angle ...')

        assert self.get_standard_variable('PA_local').metadata.unit == u.deg
        pa_local = np.deg2rad(self.get_standard_variable('PA_local').data_content)

        results = magnetic_field_results

        if not ('B_eq_' + magnetic_field_str) in results.keys():
            results |= self._get_B_eq_and_R_eq(var, magnetic_field_str, maginput)
        if not ('B_local_' + magnetic_field_str) in results.keys():
            results |= self._get_B_local(var, magnetic_field_str, maginput)

        B_eq = results['B_eq_' + magnetic_field_str][0]
        B_local = results['B_local_' + magnetic_field_str][0]

        results[var.standard_name] = (np.rad2deg(np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:,np.newaxis])), u.deg)

        return results
    
    @timed_function('Invariant mu calculation')
    def _get_invariant_mu(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating invariant mu ...')
        assert self.get_standard_variable('PA_local').metadata.unit == 'deg'
        pa_local = self.get_standard_variable('PA_local').data_content
        pa_local = np.deg2rad(self.get_standard_variable('PA_local').data_content)

        energy_vars = self.get_variables_by_type('Energy')
        assert len(energy_vars) == 1, f'We assume that there is exactly ONE energy variable available for calculating invariant mu. Found: {len(energy_vars)}!'
        energy_var = energy_vars[0]

        energy = (energy_var.data_content * energy_var.metadata.unit).to_value(u.MeV)

        species_char = energy_var.standard.standard_name[-3]

        results = magnetic_field_results
        if not ('B_local_' + magnetic_field_str) in results.keys():
            results |= self._get_B_local(var, magnetic_field_str, maginput)

        # load needed data and convert to correct units
        B_local = results['B_local_' + magnetic_field_str][0]
        B_local = (B_local * results['B_local_' + magnetic_field_str][1]).to_value(u.G)

        results[var.standard_name] = compute_invariant_mu(energy, pa_local, B_local, species_char)

        return results

    @timed_function('Invariant K calculation')
    def _get_invariant_K(self, var, magnetic_field_str, maginput, magnetic_field_results):
        print('\tCalculating invariant K ...')

        results = magnetic_field_results

        if not ('XJ_' + magnetic_field_str) in results.keys():
            results |= self._get_L_star(var, magnetic_field_str, maginput)
        if not ('B_mirr_' + magnetic_field_str) in results.keys():
            results |= self._get_B_mirr(var, magnetic_field_str, maginput)

        # load needed data and convert to correct units
        B_mirr = results['B_mirr_' + magnetic_field_str][0]
        B_mirr = (B_mirr * results['B_mirr_' + magnetic_field_str][1]).to_value(u.G)
        XJ = results['XJ_' + magnetic_field_str][0]

        results['invK_' + magnetic_field_str] = (compute_invariant_K(B_mirr, XJ), u.RE*u.G**0.5)
        
        return results

    @timed_function()
    def _get_B_mirr(self, var, magnetic_field_str, maginput):
        print('\tCalculating magnetic field of mirror points ...')

        assert self.get_standard_variable('PA_local').metadata.unit == 'deg'
        pa_local = self.get_standard_variable('PA_local').data_content

        assert self.get_standard_variable('xGEO').metadata.unit == u.RE
        xGEO = self.get_standard_variable('xGEO').data_content

        results = get_mirror_point(xGEO, var.time_variable.data_content, pa_local, self.irbem_lib_path, self.irbem_options, magnetic_field_str, maginput, self.num_cores)

        return results
