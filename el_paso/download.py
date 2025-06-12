from __future__ import annotations

import logging
import os
import re
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import requests

from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version


def download(start_time: datetime,
             end_time: datetime,
             save_path: str|Path,
             file_cadence: Literal["daily", "monthly", "single_file"],
             download_url: str,
             file_name_stem: str,
             download_arguments_prefixes: str = "",
             download_arguments_suffixes: str = "",
             method:Literal["request", "wget"]="request",
             authentification_info:tuple[str,str]=("",""),
             rename_file_name_stem: str|None = None,
             *,
             skip_existing: bool = True) -> None:
    """Download the product data according to the specified standard."""

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    save_path = Path(save_path)

    curr_time = start_time

    match file_cadence:
        case "daily":
            while curr_time <= end_time:
                match method:
                    case "request":
                        _requests_download(curr_time, save_path, download_url, file_name_stem, authentification_info, rename_file_name_stem, skip_existing=skip_existing)
                    case "wget":
                        _wget_download(curr_time, save_path, download_url, download_arguments_prefixes, download_arguments_suffixes)
                curr_time += timedelta(days=1)

        case "monthly":
            raise NotImplementedError

        case "single_file":
            match method:
                case "request":
                    _requests_download(curr_time, save_path, download_url, file_name_stem, authentification_info, rename_file_name_stem, skip_existing=skip_existing)
                case "wget":
                    _wget_download(curr_time, save_path, download_url, download_arguments_prefixes, download_arguments_suffixes)
        case _:
            raise NotImplementedError

def _requests_download(current_time:datetime,
                       save_path:Path,
                       download_url:str,
                       file_name_stem:str,
                       authentification_info:tuple[str,str],
                       rename_file_name_stem: str|None,
                       *,
                       skip_existing:bool) -> None:
    """Download a file using the requests library."""
    save_path = Path(fill_str_template_with_time(str(save_path), current_time))
    save_path.mkdir(exist_ok=True, parents=True)

    url = fill_str_template_with_time(download_url, current_time)
    file_name_stem = fill_str_template_with_time(file_name_stem, current_time)

    try:
        response = requests.get(url, stream=True, auth=requests.auth.HTTPDigestAuth(*authentification_info))

        if response.status_code == 404:
            msg = f"File not found on server: {url}"
            raise FileNotFoundError(msg)

        response.raise_for_status()

        found_files = typing.cast("list[str]", re.findall(rf"{file_name_stem}", response.text))
        latest_file_name = get_file_by_version(found_files, version="latest")

        if latest_file_name is None:
            msg = f"No file found matching the pattern {file_name_stem} in the response from {url}"
            raise FileNotFoundError(msg)

        if rename_file_name_stem is None:
            save_file_name = latest_file_name
        else:
            save_file_name = fill_str_template_with_time(rename_file_name_stem, current_time)

        if skip_existing and (save_path / save_file_name).exists():
            print(f"File already exists, skipping download: {save_path / save_file_name}")
            return

        response = requests.get(f"{url}/{latest_file_name}", stream=True, auth=requests.auth.HTTPDigestAuth(*authentification_info))

        if response.status_code == 404:
            msg = f"File not found on server: {url}"
            raise FileNotFoundError(msg)

        response.raise_for_status()

        with (save_path / save_file_name).open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded successfully: {save_path / latest_file_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")


def _wget_download(current_time:datetime,
                   save_path:Path,
                   download_url:str,
                   download_arguments_prefixes:str,
                   download_arguments_suffixes:str) -> None:

    Path(fill_str_template_with_time(str(save_path), current_time)).parents[0].mkdir(exist_ok=True, parents=True)

    # Replace "yyyymmdd" or "YYYYMMDD" in url, prefix, and suffix with the parsed string
    url = fill_str_template_with_time(download_url, current_time)
    prefix = fill_str_template_with_time(download_arguments_prefixes, current_time)
    suffix = fill_str_template_with_time(download_arguments_suffixes, current_time)
    download_command = f"wget {prefix} {url} {suffix}"
    logging.info(download_command)

    # Execute the download command
    try:
        os.system(download_command)
    except Exception as e:
        logging.info(f"Error downloading file using command {download_command}: {e}")
