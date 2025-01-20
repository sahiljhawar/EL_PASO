from datetime import datetime, timezone
from pathlib import Path
import re
import timeit
import logging
from functools import wraps

from packaging import version as version_pkg
import numpy as np
from astropy import units as u

from el_paso.classes import Variable

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

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

def timed_function(func_name=None):
    def timed_function_(f):
        @wraps(f)
        def wrap(*args, **kw):
            tic = timeit.default_timer()
            result = f(*args, **kw)
            toc = timeit.default_timer()
            if func_name:
                logging.info(f"\t\t{func_name} finished in {toc-tic:0.3f} seconds")
            else:
                logging.info(f"\t\tFinished in {toc-tic:0.3f} seconds")

            return result
        return wrap
    return timed_function_

def enforce_utc_timezone(time:datetime) -> datetime:
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time

def get_variable_by_standard_name(standard_name: str, variables: dict[str, Variable]):
    num_vars_found = 0
    var_found = None

    for var in variables.values():
        if var.standard_name == standard_name:
            var_found = var
            num_vars_found += 1

    if num_vars_found == 0:
        msg = f"No variable found with standard name {standard_name}!"
        raise ValueError(msg)

    if num_vars_found > 1:
        msg = f"Found {num_vars_found} variables with standard name {standard_name}! Please specify the variable yourself."  # noqa: E501
        raise ValueError(msg)

    return var_found

def get_variable_by_standard_type(standard_type: str, variables: dict[str, Variable]):
    num_vars_found = 0
    var_found = None

    for var in variables.values():
        if var.standard and var.standard.variable_type == standard_type:
            var_found = var
            num_vars_found += 1

    if num_vars_found == 0:
        msg = f"No variable found with variable type {standard_type}!"
        raise ValueError(msg)

    if num_vars_found > 1:
        msg = f"Found {num_vars_found} variables with variable type {standard_type}! Please specify the variable yourself."  # noqa: E501
        raise ValueError(msg)

    return var_found

def validate_standard(func):
    def validate_units(variables: dict[str, Variable]):
        for key, var in variables.items():
            if var.standard:
                assert (
                    var.metadata.unit == var.standard.standard_unit
                ), f"Variable {key} has wrong units! Actual unit: {var.metadata.unit}. Should be: {var.standard.standard_unit}."

    def validate_dimensions(variables: dict[str, Variable]):
        for key, var in variables.items():
            if var.standard:
                match var.standard.variable_type:
                    case "Epoch":
                        assert (
                            var.data.ndim == 1
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Flux":
                        if "DO" in var.standard_name:
                            expected_ndim = 2
                        elif "DU" in var.standard_name:
                            expected_ndim = 3
                        assert (
                            var.data.ndim == expected_ndim
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Energy":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "PitchAngle":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                    case "Position":
                        assert (
                            var.data.ndim == 2
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"
                        assert (
                            var.data.shape[1] == 3
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong shape: {var.data.shape}!"

                    case "MLT":
                        assert (
                            var.data.ndim == 1
                        ), f"Variable {key} with type {var.standard.variable_type} has wrong dimensions: {var.data.ndim}!"


    @wraps(func)
    def with_validated_standard(*args:tuple, **kwargs:dict):
        variables = args[0]
        if not isinstance(variables, dict):
            msg = "First argument of the function has to be a dictionary holding instances of Variable!"
            raise TypeError(msg)

        validate_units(variables)
        validate_dimensions(variables)

        return func(*args, **kwargs)

    return with_validated_standard
