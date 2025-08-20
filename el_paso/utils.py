from __future__ import annotations

import logging
import re
import time
import timeit
import typing
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from functools import wraps
from multiprocessing.pool import MapResult
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import pandas as pd
import tqdm
from packaging import version as version_pkg

import el_paso as ep

logger = logging.getLogger(__name__)

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

def extract_version(file_name:str|Path) -> tuple[str, version_pkg.Version]:
    """Extract the version string from the file name.

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

T = TypeVar("T", bound=Path|str)
def get_file_by_version(file_paths:Iterable[T], version:str) -> T|None:
    """Filter the list of files to keep only the one with the highest version or matching version.

    Args:
        file_names (list): List of file paths.
        version (str, optional): Specific version to match. If provided, only files with this version are returned.

    Returns:
        list: List of file paths with the highest version or matching version.

    """
    latest_file = None

    if version != "latest":
        normalized_version = re.sub(r"[_-]", ".", version.replace("v", ""))
        target_version = version_pkg.parse(normalized_version)
    else:
        target_version = None

    for file in file_paths:
        _, ver_obj = extract_version(file)

        # Check if the current file matches the target version if specified
        if target_version and ver_obj == target_version:
            return file

        # If no specific version is targeted, find the highest version
        if latest_file is None or ver_obj > extract_version(latest_file)[1]:
            latest_file = file

    # Extract the file names from the dictionary
    return latest_file

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

P = ParamSpec("P")
R = TypeVar("R")

def timed_function(func_name:str|None=None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def timed_function_(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrap(*args: P.args, **kwargs: P.kwargs) -> R:
            tic = timeit.default_timer()
            result = f(*args, **kwargs)
            toc = timeit.default_timer()
            if func_name:
                logger.info(f"\t\t{func_name} finished in {toc-tic:0.3f} seconds")
            else:
                logger.info(f"\t\tFinished in {toc-tic:0.3f} seconds")

            return result
        return wrap
    return timed_function_

def enforce_utc_timezone(time:datetime) -> datetime:
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time

def datenum_to_datetime(datenum_val: float) -> datetime:
    return pd.to_datetime(datenum_val-719529, unit="D", origin=pd.Timestamp("1970-01-01")).to_pydatetime().replace(tzinfo=timezone.utc)

def datetime_to_datenum(datetime_val: datetime) -> float:
    mdn = datetime_val + timedelta(days = 366)
    frac = (datetime_val - datetime(datetime_val.year, datetime_val.month, datetime_val.day, 0, 0, 0, tzinfo=timezone.utc)).seconds / (24.0 * 60.0 * 60.0)

    return mdn.toordinal() + round(frac, 6)

def assert_n_dim(var: ep.Variable, n_dims:int, name_in_file:str) -> None:

    provided = var.get_data().ndim

    if provided != n_dims:
        msg = (f"Encountered dimension missmatch for variable with name {name_in_file}:"
               "should be {n_dims}, got: {provided}!")
        raise ValueError(msg)

def show_process_bar_for_map_async(map_result:MapResult[Any], chunksize:int) -> None:
    init = typing.cast("int", map_result._number_left) * chunksize  # type: ignore[reportUnknownMemberType] # noqa: SLF001
    with tqdm.tqdm(total=init) as t:
        while (True):
            if map_result.ready():
                break
            t.n = (init-map_result._number_left*chunksize)  # type: ignore[reportUnknownMemberType] # noqa: SLF001
            t.refresh()
            time.sleep(1)

class Hashabledict(dict[Any,Any]):
    def __hash__(self):
        return hash((frozenset(self), frozenset(self.itervalues())))

def make_dict_hashable(dict_input:dict[Any,Any]|None) -> Hashabledict|None:
    if dict_input is None:
        return dict_input

    return Hashabledict(dict_input)
