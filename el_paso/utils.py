from datetime import datetime
from pathlib import Path
import re
import warnings

from packaging import version as version_pkg
import numpy as np
from astropy import units as u
from scipy import io as sio

from el_paso.classes import TimeVariable

def fill_str_template_with_time(input:str, time:datetime):

    yyyymmdd_str = time.strftime('%Y%m%d')
    yyyy_str = time.strftime('%Y')
    MM_str = time.strftime('%m')
    DD_str = time.strftime('%d')

    return input.replace("yyyymmdd", yyyymmdd_str) \
                .replace("YYYYMMDD", yyyymmdd_str) \
                .replace("YYYY", yyyy_str) \
                .replace("MM", MM_str) \
                .replace("DD", DD_str)
    

def extract_version(file_name):
    """
    Extracts the version string from the file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        tuple: A tuple containing the base file name without the version and the parsed version object.
    """
    
    # convert to str in case of Path object
    file_name = str(file_name)

    # Regular expression to find the version part (_v* or _v*.*-*.*) before the file extension
    match = re.search(r"_(v[\d._-]+)(?=\.\w+$)", file_name)
    if match:
        base_name = file_name[:match.start()]
        ver_str = match.group(1)
        # Normalize the version string by replacing separators with dots
        normalized_ver_str = re.sub(r"[_-]", ".", ver_str.replace("v", ""))
        return base_name, version_pkg.parse(normalized_ver_str)
    else:
        return file_name, version_pkg.parse("0")

def get_file_by_version(file_paths:Path, version:str):
    """
    Filters the list of files to keep only the one with the highest version or matching version.

    Args:
        file_names (list): List of file paths.
        version (str, optional): Specific version to match. If provided, only files with this version are returned.

    Returns:
        list: List of file paths with the highest version or matching version.
    """
    latest_file = None

    if version != 'latest':
        normalized_version = re.sub(r"[_-]", ".", version.replace("v", ""))
        target_version = version_pkg.parse(normalized_version)
    else:
        target_version = None

    for file in file_paths:
        _, ver_obj = extract_version(file)

        # Check if the current file matches the target version if specified
        if target_version:
            if ver_obj == target_version:
                return file

        # If no specific version is targeted, find the highest version
        if latest_file is None:
            latest_file = file
        else:
            # Compare versions and keep the file with the highest version
            if ver_obj > extract_version(latest_file)[1]:
                latest_file = file

    # Extract the file names from the dictionary
    return latest_file

def sanitize_metadata_dict(metadata_dict):
    """
    Sanitize the metadata dictionary by replacing None type objects with empty arrays.

    Args:
        metadata_dict (dict): The dictionary of dictionaries to be sanitized.

    Returns:
        dict: The sanitized dictionary.
    """
    sanitized_dict = {}

    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized_dict[key] = sanitize_metadata_dict(value)
        elif value is None:
            # Replace None with an empty numpy array
            sanitized_dict[key] = np.array([])
        else:
            # Retain other values as they are
            sanitized_dict[key] = value

    return sanitized_dict


def save_into_file(in_file_name, target_list):
    """
    Saves specified variables into the specified file name.

    Args:
        in_file_name (str): The name of the file to save data into.
        target_list (list): A list of Variable objects to be saved.

    Raises:
        UnsupportedFormatError: If the file format is not supported.
    """

    in_file_name = Path(in_file_name)

    format_name = in_file_name.suffix.lower()

    if format_name == ".mat":
        # Create a dictionary to store data and metadata
        data_dict = {}
        metadata_dict = {}

        for variable in target_list:
            # Save the data_content into a field named by save_name

            # Check if the data_content consists of datetime.datetime objects
            if isinstance(variable, TimeVariable):
                data_dict[variable.metadata.save_name] = (variable.data_content * variable.metadata.unit).to_value(u.epoch_datenum)
            else:
                data_content = variable.data_content
                if data_content is None:
                    warnings.warn(f"Variable {variable.standard_name} does not hold any content! Skipping ...")
                    continue
                if data_content.ndim == 1:
                    data_content = data_content.reshape(-1, 1)
                data_dict[variable.metadata.save_name] = data_content
                # Create metadata for each variable
                metadata_dict[variable.metadata.save_name] = {
                    'unit': variable.metadata.unit,
                    'cadence_seconds': variable.metadata.cadence_seconds,
                    'source_files': variable.metadata.source_files,
                    'description': variable.metadata.description,
                    'processing_notes': variable.metadata.processing_notes,
                    'time_bin_method': variable.metadata.time_bin_method,
                    'time_bin_interval': variable.metadata.time_bin_interval,
                }

        # Add metadata to the dictionary
        data_dict['metadata'] = sanitize_metadata_dict(metadata_dict)

        # Save the dictionary into a .mat file
        sio.savemat(str(in_file_name), data_dict)


    elif format_name == ".csv":

        data_arr = []

        column_names = []  # To store column names for variables

        for variable in target_list:

            # Check if variable.data_content is 1D

            if variable.data_content.ndim == 1:

                # Add the 1D array as a column

                data_arr.append(variable.data_content)

                # Check if variable has save_name, else use workspace_name

                if hasattr(variable, 'save_name') and variable.save_name:

                    column_names.append(variable.save_name)

                else:

                    column_names.append(variable.workspace_name)

        # Convert data_arr to a 2D NumPy array where each column is from the 1D arrays

        # Convert data_arr to a structured NumPy array where each column retains its original dtype
        if data_arr:
            # Identify the dtypes of each column
            dtypes = []
            for col_data in data_arr:
                if np.issubdtype(col_data.dtype, np.number):
                    dtypes.append((column_names[len(dtypes)], col_data.dtype))  # e.g., float or int dtype
                else:
                    max_str_len = max([len(str(item)) for item in col_data])
                    dtypes.append(
                        (column_names[len(dtypes)], f'U{max_str_len}'))  # Unicode string with max length

            # Create the structured array
            structured_data = np.empty(len(data_arr[0]), dtype=dtypes)

            # Populate the structured array with the actual data
            for i, col_data in enumerate(data_arr):
                structured_data[column_names[i]] = col_data

            # Create the format string for savetxt
            fmt = ['%s' if np.issubdtype(dtype[1], np.str_) else '%.16f' for dtype in dtypes]
        else:
            raise ValueError("Attempting to save an empty data file!")

        # Prepare the header string

        header_str = ""

        # Check if self.save_header is True, append the main header

        if self.save_header:
            header_str += self.header  # Assume self.header is already a string

            header_str += "\n"  # Add a newline after the header

        # Check if self.save_columns is True, append the column names

        if self.save_columns:
            header_str += self.save_separator.join(column_names)  # Join column names with the delimiter

            header_str += "\n"  # Add a newline after the column names

        # Save the data using np.savetxt

        np.savetxt(in_file_name, structured_data, delimiter=self.save_separator,
                    fmt=fmt, header=header_str, comments='')
    elif format_name == ".txt" or format_name == ".asc":
        raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
    elif format_name == ".h5":
        raise UnsupportedFormatError(f"The '{format_name}' format is not supported yet.")
    else:
        raise UnsupportedFormatError(f"The '{format_name}' format is not implemented.")