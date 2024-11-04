import os
from datetime import datetime, timedelta
from datetime import timezone as tz
from pathlib import Path

import numpy as np
import cdflib

from el_paso.classes import TimeVariable
from el_paso.utils import fill_str_template_with_time, get_file_by_version

class SourceFile():

    def __init__(
            self,
            download_url:str,
            download_arguments_prefixes:str,
            download_arguments_suffixes:str,
            download_path:str,
            variables_to_extract:dict,
            file_cadence:str='daily'):
        
        self.download_url = download_url
        self.download_arguments_prefixes = download_arguments_prefixes
        self.download_arguments_suffixes = download_arguments_suffixes
        self.download_path = download_path
        self.variables_to_extract = variables_to_extract
        self.file_cadence = file_cadence

    def download(self, start_time:datetime, end_time:datetime):
        """Downloads the product data according to the specified standard."""

        if self.file_cadence == 'daily':
            # Parse start_time into a YYYYMMDD string
            if isinstance(start_time, str):
                start_time = datetime.strptime(start_time, '%Y-%m-%d')
                start_time = start_time.replace(tzinfo=tz.UTC)
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, '%Y-%m-%d')
                end_time = end_time.replace(hour=23, minute=59, second=59, tzinfo=tz.UTC)

            # Replace "yyyymmdd" or "YYYYMMDD" in url, prefix, and suffix with the parsed string
            url = fill_str_template_with_time(self.download_url)
            prefix = fill_str_template_with_time(self.download_arguments_prefixes)
            suffix = fill_str_template_with_time(self.download_arguments_suffixes)
            download_command = f'wget {prefix} {url} {suffix}'
            print(download_command)

            # Execute the download command
            try:
                os.system(download_command)
            except Exception as e:
                print(f"Error downloading file using command {download_command}: {e}")

        else:
            raise NotImplementedError(f"proper handling of {self.filetype} type of files is not yet implemented!")

    def reset_variable_contents(self):
        for var in self.variables_to_extract.values():
            var.data_content = None

    def extract_variables(self, start_time:datetime, end_time:datetime):
        
        self.reset_variable_contents()

        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
            start_time = start_time.replace(tzinfo=tz.UTC)
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d')
            end_time = end_time.replace(hour=23, minute=59, second=59, tzinfo=tz.UTC)
        
        files_list, _ = self._construct_downloaded_file_list(start_time, end_time, self.file_cadence)

        print(files_list)

        for file_path in files_list:

            if file_path.suffix == '.cdf':
                self._extract_varibles_from_cdf(file_path)
            elif file_path.suffix in ['.txt', '.asc', '.csv', '.tab']:
                self._load_ascii_file_to_extract(file_path)
            elif file_path.suffix == '.nc':
                self._load_nc_file_to_extract(file_path)
            elif file_path.suffix == '.h5':
                raise NotImplementedError("HDF5 reading is not supported yet!")
                #self._load_h5_file_to_extract(file_path, variables)
            elif file_path.suffix == '.json':
                self._load_json_file_to_extract(file_path)
            elif file_path.suffix == '.mat':
                self._load_mat_file_to_extract(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        return self.variables_to_extract
    
    def _extract_varibles_from_cdf(self, file_path: str):
        """
        Loads data from a CDF file and updates the variables.

        Args:
            file_path (str): The path to the CDF file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """

        # Open the CDF file
        cdf_file = cdflib.CDF(file_path)
        cdfinfo = cdf_file.cdf_info()
        globalattrs = cdf_file.globalattsget()
        variable_data = {}

        # Extract data for each variable in self.variables
        for var in self.variables_to_extract.values():
            if var.name_or_column_in_file in cdfinfo.zVariables:
                # Retrieve data corresponding to the variable name from the CDF file
                variable_data[var.name_or_column_in_file] = cdf_file.varget(var.name_or_column_in_file)

        # Update the data content of variables
        for var in self.variables_to_extract.values():
            if var.name_or_column_in_file in variable_data:
                if var.data_content is not None and var.data_content.any():
                    var.data_content = np.concatenate((var.data_content, variable_data[var.name_or_column_in_file]),
                                                      axis=0)
                else:
                    var.data_content = variable_data[var.name_or_column_in_file]


    def _get_downloaded_file_name(self, time:datetime):

        file_path = Path(fill_str_template_with_time(self.download_path, time))
        file_names_all_versions = file_path.parent.glob(file_path.stem)

        return get_file_by_version(file_names_all_versions, version='latest')


    def _construct_downloaded_file_list(self, start_time: datetime, end_time: datetime, file_cadence: str):
        """
        Constructs the list of source files and corresponding times for the given interval.

        Args:
            start_time (datetime): The start time of the interval.
            end_time (datetime): The end time of the interval.
            filetype (str): 'daily' or 'monthly'

        Returns:
            Tuple[List[List[str]], List[Tuple[datetime, datetime]]]: List of list of source file paths
                                                                        and corresponding time intervals.
        """
        file_paths = []
        time_intervals = []

        if file_cadence == 'daily':
            current_time = start_time
            while current_time <= end_time:
                time_str = current_time.strftime('%Y%m%d')
                
                file_path = self._get_downloaded_file_name(current_time)
                file_paths.append(file_path)

                day_start = datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0)
                day_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)
                time_intervals.append([day_start, day_end])
                
                current_time += timedelta(days=1)
        elif file_cadence == 'custom-one-file':
            current_time = start_time
            path_list = []
            while current_time <= end_time:
                time_str = current_time.strftime('%Y%m%d')
                for path in self.download_paths:
                    file_path = self.get_source_file_name(path, time_str)
                    # Check if file_path is a list
                    if isinstance(file_path, list):
                        path_list.extend(file_path)  # Use extend() if file_path is a list
                    else:
                        path_list.append(file_path)  # Use append() if file_path is a single element
                file_paths.append(path_list)
                day_start = datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0)
                day_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)
                current_time += timedelta(days=1)
            time_intervals = [[start_time, datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59)]]
        elif file_cadence == 'monthly':
            current_time = start_time.replace(day=1)
            while current_time <= end_time:
                year = current_time.year
                month = current_time.month
                eom_day = calendar.monthrange(year, month)[1]
                time_str = f"{year:04d}{month:02d}01to{year:04d}{month:02d}{eom_day:02d}"
                path_list = []
                for path in self.download_paths:
                    file_path = self.get_source_file_name(path, time_str)
                    path_list.append(file_path)
                file_paths.append(path_list)
                month_start = datetime(year, month, 1, 0, 0, 0)
                month_end = datetime(year, month, eom_day, 23, 59, 59)
                time_intervals.append([month_start, month_end])
                if month == 12:
                    current_time = datetime(year + 1, 1, 1)
                else:
                    current_time = datetime(year, month + 1, 1)

        return file_paths, time_intervals
    

    def _load_mat_file_to_access(self, file_path: str, variables):
        """
        Loads data from a .mat file and updates the variables.

        Args:
            file_path (str): The path to the .mat file.
            variables (List[Variable], optional): Specific variables to load from the file.
        Returns:
            return_variables (List[Variable]): list of Variable objects to return
        """
        mat_data = loadmat_nested(file_path)
        save_standard = self.current_save_standard  # Assuming self.save_standard is set
        if self.current_save_standard is None:
            save_standard = self.default_save_standard

        mat_data_fields = list(mat_data.keys())
        return_variables = []
        for var in self.variables:
            save_name = save_standard.variable_mapping(var)
            if save_name in mat_data_fields:
                self.set_variable_attribute(var.workspace_name, 'data_content',
                                            mat_data[save_name])
                return_var = var
                return_var.workspace_name = save_name
                return_variables.append(return_var)
                mat_data_fields.remove(save_name)

        for field_name in mat_data_fields:
            #  if field_name != 'metadata' and not field_name.startswith('__'):
            if not field_name.startswith('__'):
                new_var = Variable(data_content=mat_data[field_name], workspace_name=field_name,
                                   save_standard=save_standard)
                return_variables.append(new_var)

        if variables is not None:
            return_variables = [variable for variable in return_variables if variable.workspace_name in variables]

        self.variables = return_variables
        return return_variables

    def _load_ascii_file_to_access(self, file_path: str, variables) -> None:
        """
        Loads data from an ASCII file and updates the variables.

        Args:
            file_path (str): The path to the ASCII file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """
        return_variables = []
        # Step 1: Open the ASCII file and read the header rows manually if needed
        with open(file_path, 'r') as file:
            # Read the header rows (self.header_length defines how many rows to read)
            self.header = [next(file).strip() for _ in range(self.header_length)]

        # Step 2: Use pandas.read_csv to read the file from disk, skipping header rows
        # If save_columns is True, read column names from the file, otherwise read without column names
        if self.save_columns:
            df = pd.read_csv(file_path, delimiter=self.save_separator, skiprows=self.header_length)
            column_names = df.columns.tolist()
        else:
            df = pd.read_csv(file_path, delimiter=self.save_separator, skiprows=self.header_length, header=None)
            column_names = None

        # Step 3: Process the data based on the number of columns
        num_columns = df.shape[1]  # Number of columns in the dataframe

        if num_columns == len(self.target_variables):
            # Case 1: Data matches the number of target_variables
            for i, var_name in enumerate(self.target_variables):
                # Find the variable with the matching workspace_name
                variable = None
                for var in self.variables:
                    if var.workspace_name == var_name:
                        variable = var
                        break

                if variable:
                    # Set the data_content for the variable from the DataFrame column
                    variable.data_content = df.iloc[:, i].values
                    return_variables.append(variable)
        elif self.save_columns and column_names:
            # Case 2: Matching columns with save_name of variables or creating new ones
            for i, col_name in enumerate(column_names):
                # Try to find a variable with the matching save_name
                variable = None
                for var in self.variables:
                    # Check if 'save_name' exists and is not None, otherwise use 'workspace_name'
                    name_to_check = var.save_name if hasattr(var,
                                                             'save_name') and var.save_name is not None else var.workspace_name
                    if name_to_check == col_name:
                        variable = var
                        break

                if variable:
                    # If a match is found, update its data_content from the DataFrame
                    variable.data_content = df[col_name].values
                    return_variables.append(variable)
                else:
                    # If no match, create a new variable with the column data
                    new_var = Variable(data_content=df[col_name].values, workspace_name=col_name)
                    return_variables.append(new_var)
        else:
            raise ValueError("Mismatch between the number of data columns and target variables or column names.")

        # Step 5: Filter return_variables based on the provided 'variables' argument, if any
        if variables is not None:
            return_variables = [variable for variable in return_variables if variable.workspace_name in variables]

        self.variables = return_variables


    def open_data_file_to_extract(self, file_path: str, variables) -> None:
        """
        Opens the data file and loads the content based on the file format.

        Args:
            file_path (str): The path to the data file.
            variables (List[Variable], optional): Variables to pull out from the file
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.cdf':
            self._load_cdf_file_to_extract(file_path, variables)
        elif file_extension in ['.txt', '.asc', '.csv', '.tab']:
            self._load_ascii_file_to_extract(file_path, variables)
        elif file_extension == '.nc':
            self._load_nc_file_to_extract(file_path, variables)
        elif file_extension == '.h5':
            raise NotImplementedError("HDF5 reading is not supported yet!")
            #self._load_h5_file_to_extract(file_path, variables)
        elif file_extension == '.json':
            self._load_json_file_to_extract(file_path, variables)
        elif file_extension == '.mat':
            self._load_mat_file_to_extract(file_path, variables)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def _load_mat_file_to_extract(self, file_path: str, variables) -> None:
        """
        Loads data from a .mat file and updates the variables.

        Args:
            file_path (str): The path to the .mat file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """
        mat_data = sio.loadmat(file_path)
        save_standard = self.save_standard  # Assuming self.save_standard is set

        def update_variable(var_name: str, data_content) -> None:
            for var in self.variables:
                if var.save_standard == save_standard and var.standard_name == var_name:
                    if var.data_content is not None and var.data_content.any():
                        var.data_content = np.concatenate((var.data_content, data_content),
                                                          axis=0)
                    else:
                        var.data_content = data_content
                    return
            new_var = Variable(data_content=data_content, workspace_name=var_name, save_standard=var_name)
            self.variables.append(new_var)

        if variables:
            for var in variables:
                save_name = var.save_standard
                if save_name in mat_data:
                    if var.data_content is not None and var.data_content.any():
                        var.data_content = np.concatenate((var.data_content, mat_data[save_name]),
                                                          axis=0)
                    else:
                        var.data_content = mat_data[save_name]
        else:
            for var_name, data_content in mat_data.items():
                if not var_name.startswith('__'):  # Skipping meta variables in .mat files
                    update_variable(var_name, data_content)

    def _load_json_file_to_extract(self, file_path: str, variables) -> None:
        """
        Loads data from a .json file and updates the variables.

        Args:
            file_path (str): The path to the .json file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        json_df = pd.DataFrame(data)

        if variables is None:
            variables = self.variables

        variable_data: Dict[str, np.ndarray] = {}

        for var in variables:
            name_or_column_in_file = var.name_or_column_in_file
            if name_or_column_in_file in json_df:
                if not var.dependent_variables:
                    variable_data[var.name_or_column_in_file] = np.array(pd.unique(json_df[name_or_column_in_file]))
                else:
                    dependent_data = [
                        json_df[self.get_variable_by_source_name(dep_var).name_or_column_in_file]
                        for dep_var in var.dependent_variables
                    ]
                    unique_values = [pd.unique(dep) for dep in dependent_data]
                    shape = tuple(len(uq) for uq in unique_values)

                    # Determine the correct dtype for the data_array based on the column data type
                    dtype = object if json_df[name_or_column_in_file].dtype == object else float

                    data_array = np.full(shape, np.nan, dtype=dtype)

                    for indices in np.ndindex(*shape):
                        mask = np.ones(len(json_df), dtype=bool)
                        for i, idx in enumerate(indices):
                            mask &= (dependent_data[i] == unique_values[i][idx])
                        if mask.any():
                            data_array[indices] = json_df[name_or_column_in_file][mask].values[0]

                    variable_data[var.name_or_column_in_file] = data_array

        for var in self.variables:
            if var.data_content is not None and var.data_content.any():
                var.data_content = np.concatenate((var.data_content, variable_data[var.name_or_column_in_file]),
                                                  axis=0)
            else:
                var.data_content = variable_data[var.name_or_column_in_file]

    def _split_line(self, line: str):
        """
        Splits a line based on the specified separator.

        Args:
            line (str): The line to be split.

        Returns:
            List[str]: A list of fields extracted from the line.
        """
        if self.separator == ' ':
            # Use regular expression to split by any number of whitespace characters
            return re.split(r'\s+', line.strip())
        else:
            # Split by the specified separator
            return line.strip().split(self.separator)

    def _load_ascii_file_to_extract(self, file_path: str) -> None:
        """
        Loads data from an ASCII file and updates the variables.

        Args:
            file_path (str): The path to the ASCII file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """

        variable_data = {}

        delimiter = self.separator
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)

            # Skip but store the header lines
            for _ in range(self.header_length):
                header_line = file.readline().strip()
                self.file_header.append(header_line)

            # Handle column names if self.columns is True
            if self.columns:
                headers = self._split_line(file.readline())
                for var in self.variables:
                    if var.name_or_column_in_file in headers:
                        variable_data[var.name_or_column_in_file] = []
            else:
                for var in self.variables:
                    variable_data[var.workspace_name] = []

            # Read the data content
            for line in file:
                row = self._split_line(line)
                if isinstance(row, list) and len(row) < 2:
                    row = ''.join(row)  # Join list elements into a single string
                    if self.separator is not None:
                        row = row.split(self.separator)
                if self.columns:
                    for var in self.variables:
                        if var.name_or_column_in_file in headers:
                            index = headers.index(var.name_or_column_in_file)
                            variable_data[var.name_or_column_in_file].append(row[index])
                else:
                    for var in self.variables:
                        if isinstance(var.name_or_column_in_file, int) and var.name_or_column_in_file < len(row):
                            variable_data[var.workspace_name].append(row[var.name_or_column_in_file])

        # Update the variables' data content
        for var in self.variables:
            if self.columns:
                key = var.name_or_column_in_file
            else:
                key = var.workspace_name

            if key in variable_data:
                # Retrieve the data content from variable_data
                data_content = variable_data[key]

                # Convert the data content to a NumPy array if it's a list
                if isinstance(data_content, list):
                    # Check if all elements in the list can be converted to numeric (int or float)
                    if all([isinstance(x, (int, float)) for x in data_content]):
                        data_content = np.array(data_content)

                # Ensure the data content is a 1D array along axis 0
                data_content = np.squeeze(data_content)

                # Assign the processed data content back to the variable's data_content attribute
                if var.data_content is not None and var.data_content.any():
                    var.data_content = np.concatenate((var.data_content, data_content),
                                                      axis=0)
                else:
                    var.data_content = data_content

    def _load_nc_file_to_extract(self, file_path: str) -> None:
        """
        Loads data from a NetCDF file and updates the variables.

        Args:
            file_path (str): The path to the NetCDF file.
            variables (List[Variable], optional): Specific variables to load from the file.
        """

        # Open the NetCDF file
        nc_file = sio.netcdf.NetCDFFile(file_path, 'r')
        variable_data = {}

        # Extract data for each variable in self.variables
        for var in self.variables:
            if var.name_or_column_in_file in nc_file.variables:
                # Retrieve data corresponding to the variable name from the CDF file
                variable_data[var.name_or_column_in_file] = nc_file.variables[var.name_or_column_in_file]

        # Update the data content of variables
        for var in self.variables:
            if var.name_or_column_in_file in variable_data:
                if var.data_content is not None and var.data_content.any():
                    var.data_content = np.concatenate((var.data_content, variable_data[var.name_or_column_in_file]),
                                                      axis=0)
                else:
                    var.data_content = variable_data[var.name_or_column_in_file]
