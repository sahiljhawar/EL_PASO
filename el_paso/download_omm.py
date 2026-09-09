# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from el_paso.download import _get_next_time
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, timed_function

if TYPE_CHECKING:
    from el_paso.typing import FileCadence

logger = logging.getLogger(__name__)

_SPACETRACK_BASE_URL = "https://www.space-track.org"
_SPACETRACK_LOGIN_URL = f"{_SPACETRACK_BASE_URL}/ajaxauth/login"
_SPACETRACK_QUERY_URL = f"{_SPACETRACK_BASE_URL}/basicspacedata/query"
_SPACETRACK_TIMEOUT_SECONDS = 60
_SPACETRACK_RATE_LIMIT_STATUS = 429
_SPACETRACK_DEFAULT_RETRY_AFTER_SECONDS = 60.0
_SPACETRACK_MAX_RETRIES = 3

_LESS_THAN = "%3C"

_QUERY_DATETIME_FORMAT = "%Y-%m-%d"

_OMM_CSV_FIELDS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)


def _spacetrack_credentials(username: str | None, password: str | None) -> tuple[str, str]:
    if username is None:
        username = os.environ.get("SPACETRACK_USER")
    if password is None:
        password = os.environ.get("SPACETRACK_PASS")

    if username is None:
        msg = "Space-Track username not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if password is None:
        msg = "Space-Track password not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    return username, password


@cache
def _login_spacetrack(username: str, password: str) -> requests.Session:
    """Authenticate with Space-Track and return a session carrying the login cookie.

    Cached per (username, password) so a single process reuses one session (and thus one login
    request) across multiple `download_omm` calls.
    """
    session = requests.Session()
    response = session.post(
        _SPACETRACK_LOGIN_URL,
        data={"identity": username, "password": password},
        timeout=_SPACETRACK_TIMEOUT_SECONDS,
    )

    if response.status_code != requests.codes.ok:
        msg = f"Space-Track login failed: {response.text}"
        raise ValueError(msg)

    return session


def _get_with_retry(session: requests.Session, url: str) -> requests.Response:
    """GET `url`, backing off on Space-Track's 429 (rate limit: 30/min, 300/hr per its docs)."""
    response = session.get(url, timeout=_SPACETRACK_TIMEOUT_SECONDS)

    for _ in range(_SPACETRACK_MAX_RETRIES):
        if response.status_code != _SPACETRACK_RATE_LIMIT_STATUS:
            return response

        wait_seconds = float(response.headers.get("Retry-After", _SPACETRACK_DEFAULT_RETRY_AFTER_SECONDS))
        logger.warning(f"Space-Track rate limit hit, waiting {wait_seconds:.0f}s before retrying: {url}")
        time.sleep(wait_seconds)
        response = session.get(url, timeout=_SPACETRACK_TIMEOUT_SECONDS)

    return response


def _build_gp_history_url(norad_id: int, query_suffix: str) -> str:
    return f"{_SPACETRACK_QUERY_URL}/class/gp_history/NORAD_CAT_ID/{norad_id}/{query_suffix}"


def _parse_omm_xml(xml_content: str) -> list[dict[str, str]]:
    """Parse a Space-Track Orbit Mean-Elements Message (OMM) XML response.

    Args:
        xml_content (str): The raw XML text of a Space-Track `gp`/`gp_history` query response.

    Returns:
        list[dict[str, str]]: One dict per element set, in document order. Includes every field
        Space-Track's XML carries (metadata, mean elements, TLE parameters, and any
        userDefinedParameters extras); `download_omm` itself only persists the canonical
        OMM/TLE subset (see `_OMM_CSV_FIELDS`).
    """
    root = ET.fromstring(xml_content)  # noqa: S314

    records: list[dict[str, str]] = []

    for omm_elem in root.findall("omm"):
        for segment in omm_elem.findall("./body/segment"):
            record: dict[str, str] = {}

            metadata = segment.find("metadata")
            if metadata is not None:
                record.update({child.tag: child.text or "" for child in metadata})

            data = segment.find("data")
            if data is not None:
                mean_elements = data.find("meanElements")
                if mean_elements is not None:
                    record.update({child.tag: child.text or "" for child in mean_elements})

                tle_parameters = data.find("tleParameters")
                if tle_parameters is not None:
                    record.update({child.tag: child.text or "" for child in tle_parameters})

                user_defined_parameters = data.find("userDefinedParameters")
                if user_defined_parameters is not None:
                    for user_defined in user_defined_parameters.findall("USER_DEFINED"):
                        parameter_name = user_defined.get("parameter")
                        if parameter_name:
                            record[parameter_name] = user_defined.text or ""

            if record:
                records.append(record)

    return records


def _parse_epoch(epoch: str) -> datetime:
    return enforce_utc_timezone(datetime.fromisoformat(epoch))


def _write_omm_csv(records: list[dict[str, str]], file_path: Path) -> None:
    """Write parsed OMM records to disk as CSV, restricted to the canonical OMM/TLE field set."""
    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OMM_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in _OMM_CSV_FIELDS})


def _group_consecutive(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = [[indices[0]]]

    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    return groups


def _target_path(save_path: Path, file_name_stem: str, time: datetime, *, sort_raw_files_by_time: bool) -> Path:
    """Resolve one chunk's output path, matching `el_paso.download`'s own file layout.

    Only `file_name_stem` (and the generated 'YYYY/MM/' suffix) is time-templated, never
    `save_path` itself: `fill_str_template_with_time` does a plain substring replace, so
    templating the whole path would corrupt any literal 'MM'/'YYYY'/etc. in a user-chosen
    `save_path` (e.g. a directory literally named "OMM").
    """
    if sort_raw_files_by_time:
        save_path = save_path / fill_str_template_with_time("YYYY/MM/", time)

    return save_path / fill_str_template_with_time(file_name_stem, time)


def _download_omm_instant(
    norad_id: int,
    target_time: datetime,
    target_path: Path,
    username: str,
    password: str,
    *,
    skip_existing: bool,
) -> None:
    if skip_existing and target_path.exists():
        logger.info(f"File already exists, skipping download: {target_path}")
        return

    session = _login_spacetrack(username, password)

    epoch_predicate = f"{_LESS_THAN}{target_time.strftime(_QUERY_DATETIME_FORMAT)}"
    url = _build_gp_history_url(norad_id, f"EPOCH/{epoch_predicate}/orderby/EPOCH desc/format/xml/limit/1")

    response = _get_with_retry(session, url)
    response.raise_for_status()
    records = _parse_omm_xml(response.text)

    if not records:
        logger.warning(f"No OMM elset found at or before {target_time.isoformat()} for NORAD ID {norad_id}.")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _write_omm_csv(records, target_path)
    logger.info(f"Downloaded successfully: {target_path}")


@timed_function("download_omm")
def download_omm(
    norad_id: int,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    save_path: str | Path = "./OMM",
    file_name_stem: str = "omm_YYYYMMDD.csv",
    username: str | None = None,
    password: str | None = None,
    *,
    sort_raw_files_by_time: bool = False,
    skip_existing: bool = True,
) -> None:
    """Download Orbit Mean-Elements Message (OMM) data from Space-Track for one NORAD ID.

    Args:
        norad_id (int): The NORAD catalog ID to download OMM data for.
        start_time (datetime | None, optional): Start of the time range to download. If `end_time` is None, this is
            instead treated as the single instant to fetch the nearest preceding element set for. If not given,
            today's date at 00:00:00 UTC is used.
        end_time (datetime | None, optional): End of the time range to download. If None,
            `start_time` is treated as an individual instant: the most recent element set with an
            EPOCH strictly before `start_time` is downloaded. Defaults to None.
        save_path (str | Path, optional): Base directory the parsed CSV files are written under.
            Defaults to "./OMM".
        file_name_stem (str, optional): Time-templated file name (see `fill_str_template_with_time`)
            for each chunk's output file, joined onto `save_path`. Defaults to "omm_YYYYMMDD.csv".
        username (str | None, optional): Space-Track username. If None, read from the
            `SPACETRACK_USER` environment variable. Defaults to None.
        password (str | None, optional): Space-Track password. If None, read from the
            `SPACETRACK_PASS` environment variable. Defaults to None.
        sort_raw_files_by_time (bool, optional): If True, creates subdirectories for each year and
            month under `save_path` (e.g. 'YYYY/MM/'). If not given, files are written directly under `save_path`.
        skip_existing (bool, optional): If True, skip downloading (and querying Space-Track for)
            chunks whose output file already exists. Defaults to True.

    Raises:
        ValueError: If `username`/`password` is not provided and not available via the
            `SPACETRACK_USER`/`SPACETRACK_PASS` environment variables, if Space-Track login fails,
            or if `end_time` is not None and not after `start_time`.
    """
    username, password = _spacetrack_credentials(username, password)

    if not start_time:
        start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    start_time = enforce_utc_timezone(start_time)
    save_path = Path(save_path)

    if end_time is None:
        target_path = _target_path(save_path, file_name_stem, start_time, sort_raw_files_by_time=sort_raw_files_by_time)
        _download_omm_instant(norad_id, start_time, target_path, username, password, skip_existing=skip_existing)
        return

    end_time = enforce_utc_timezone(end_time)

    if end_time <= start_time:
        msg = "'end_time' must be after 'start_time' (omit 'end_time' to query a single instant)."
        raise ValueError(msg)

    chunks: list[tuple[datetime, datetime]] = []
    curr_time = start_time
    while curr_time < end_time:
        next_time = _get_next_time(curr_time, "daily")
        next_time = end_time if next_time is None else min(next_time, end_time)
        chunks.append((curr_time, next_time))
        curr_time = next_time

    target_paths = [
        _target_path(save_path, file_name_stem, chunk_start, sort_raw_files_by_time=sort_raw_files_by_time)
        for chunk_start, _ in chunks
    ]

    missing_indices = [i for i, path in enumerate(target_paths) if not (skip_existing and path.exists())]

    if not missing_indices:
        logger.info("All OMM files already exist, skipping Space-Track query.")
        return

    session = _login_spacetrack(username, password)

    for group in _group_consecutive(missing_indices):
        group_start = chunks[group[0]][0]
        group_end = chunks[group[-1]][1]

        start_str = group_start.strftime(_QUERY_DATETIME_FORMAT)
        end_str = group_end.strftime(_QUERY_DATETIME_FORMAT)
        url = _build_gp_history_url(norad_id, f"EPOCH/{start_str}--{end_str}/orderby/EPOCH asc/format/xml")

        response = _get_with_retry(session, url)
        response.raise_for_status()
        records = _parse_omm_xml(response.text)

        for idx in group:
            chunk_start, chunk_end = chunks[idx]
            chunk_records = [
                record
                for record in records
                if "EPOCH" in record and chunk_start <= _parse_epoch(record["EPOCH"]) < chunk_end
            ]

            if not chunk_records:
                logger.warning(
                    f"No OMM records found between {chunk_start.isoformat()} and {chunk_end.isoformat()} "
                    f"for NORAD ID {norad_id}."
                )
                continue

            target_paths[idx].parent.mkdir(parents=True, exist_ok=True)
            _write_omm_csv(chunk_records, target_paths[idx])
            logger.info(f"Downloaded successfully: {target_paths[idx]}")
