# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import el_paso as ep

_SAMPLE_OMM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ndm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<omm id="CCSDS_OMM_VERS" version="3.0">
<header>
<COMMENT>GENERATED VIA SPACE-TRACK.ORG API</COMMENT>
<CREATION_DATE>2024-01-01T01:16:16</CREATION_DATE>
<ORIGINATOR>18 SPCS</ORIGINATOR>
</header>
<body><segment>
<metadata>
<OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata>
<data>
<meanElements>
<EPOCH>2024-01-01T00:18:14.850432</EPOCH>
<MEAN_MOTION>15.49961425</MEAN_MOTION>
<ECCENTRICITY>0.00033470</ECCENTRICITY>
<INCLINATION>51.6422</INCLINATION>
<RA_OF_ASC_NODE>68.6294</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>343.4617</ARG_OF_PERICENTER>
<MEAN_ANOMALY>78.0593</MEAN_ANOMALY>
</meanElements>
<tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>43247</REV_AT_EPOCH>
<BSTAR>0.00029758000000</BSTAR>
<MEAN_MOTION_DOT>0.00016541</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0000000000000</MEAN_MOTION_DDOT>
</tleParameters>
<userDefinedParameters>
<USER_DEFINED parameter="SEMIMAJOR_AXIS">6794.976</USER_DEFINED>
<USER_DEFINED parameter="OBJECT_TYPE">PAYLOAD</USER_DEFINED>
<USER_DEFINED parameter="DECAY_DATE"></USER_DEFINED>
</userDefinedParameters>
</data>
</segment></body>
</omm>
<omm id="CCSDS_OMM_VERS" version="3.0">
<header>
<COMMENT>GENERATED VIA SPACE-TRACK.ORG API</COMMENT>
<CREATION_DATE>2024-01-01T06:46:17</CREATION_DATE>
<ORIGINATOR>18 SPCS</ORIGINATOR>
</header>
<body><segment>
<metadata>
<OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata>
<data>
<meanElements>
<EPOCH>2024-01-01T04:40:56.161344</EPOCH>
<MEAN_MOTION>15.49966131</MEAN_MOTION>
<ECCENTRICITY>0.00034720</ECCENTRICITY>
<INCLINATION>51.6400</INCLINATION>
<RA_OF_ASC_NODE>67.7274</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>339.1814</ARG_OF_PERICENTER>
<MEAN_ANOMALY>20.9033</MEAN_ANOMALY>
</meanElements>
<tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>43250</REV_AT_EPOCH>
<BSTAR>0.00029045000000</BSTAR>
<MEAN_MOTION_DOT>0.00014230</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0000000000000</MEAN_MOTION_DDOT>
</tleParameters>
</data>
</segment></body>
</omm>
</ndm>
"""

_EMPTY_OMM_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<ndm></ndm>\n'


def test_parse_omm_xml_extracts_all_fields() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    records = download_omm_mod._parse_omm_xml(_SAMPLE_OMM_XML)

    assert len(records) == 2

    first = records[0]
    assert first["OBJECT_NAME"] == "ISS (ZARYA)"
    assert first["EPOCH"] == "2024-01-01T00:18:14.850432"
    assert first["MEAN_MOTION"] == "15.49961425"
    assert first["NORAD_CAT_ID"] == "25544"
    assert first["BSTAR"] == "0.00029758000000"
    # userDefinedParameters extension fields are flattened in by their 'parameter' attribute
    assert first["SEMIMAJOR_AXIS"] == "6794.976"
    assert first["OBJECT_TYPE"] == "PAYLOAD"
    assert first["DECAY_DATE"] == ""

    second = records[1]
    assert second["EPOCH"] == "2024-01-01T04:40:56.161344"
    assert second["REV_AT_EPOCH"] == "43250"
    # second record has no userDefinedParameters block
    assert "SEMIMAJOR_AXIS" not in second


def test_parse_omm_xml_empty_response() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    assert download_omm_mod._parse_omm_xml(_EMPTY_OMM_XML) == []


def test_group_consecutive() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")

    assert download_omm_mod._group_consecutive([0, 1, 2, 5, 6, 9]) == [[0, 1, 2], [5, 6], [9]]
    assert download_omm_mod._group_consecutive([3]) == [[3]]


def test_write_omm_csv_matches_canonical_field_set_and_order(tmp_path: Path) -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")

    records = download_omm_mod._parse_omm_xml(_SAMPLE_OMM_XML)
    file_path = tmp_path / "sample.csv"
    download_omm_mod._write_omm_csv(records, file_path)

    lines = file_path.read_text().splitlines()
    assert lines[0] == (
        "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
        "ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,"
        "ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT"
    )

    with file_path.open(newline="") as f:
        reread_records = list(csv.DictReader(f))

    assert len(reread_records) == 2
    assert reread_records[0]["OBJECT_NAME"] == "ISS (ZARYA)"
    assert reread_records[0]["EPOCH"] == "2024-01-01T00:18:14.850432"
    assert reread_records[0]["NORAD_CAT_ID"] == "25544"
    # userDefinedParameters extras (e.g. SEMIMAJOR_AXIS) are not part of the canonical field set
    assert "SEMIMAJOR_AXIS" not in reread_records[0]


def test_download_omm_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASS", raising=False)

    with pytest.raises(ValueError, match="Space-Track username not found"):
        ep.download_omm(
            norad_id=25544,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            save_path=tmp_path,
        )


def test_download_omm_end_time_before_start_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    with pytest.raises(ValueError, match="'end_time' must be after 'start_time'"):
        ep.download_omm(
            norad_id=25544,
            start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            save_path=tmp_path,
        )


@pytest.mark.basic
def test_download_omm_skip_existing_makes_no_network_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When every target chunk file already exists, no login or query should happen."""
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    download_omm_mod = importlib.import_module("el_paso.download_omm")

    for day in ("20240101", "20240102", "20240103"):
        (tmp_path / f"omm_{day}.csv").write_text("OBJECT_NAME\nplaceholder\n")

    login_calls = []
    monkeypatch.setattr(
        download_omm_mod, "_login_spacetrack", lambda *args: login_calls.append(args) or MagicMock()
    )

    ep.download_omm(
        norad_id=25544,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 4, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=True,
    )

    assert login_calls == []


@pytest.mark.basic
def test_download_omm_range_mode_writes_parsed_files_and_discards_xml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    download_omm_mod = importlib.import_module("el_paso.download_omm")

    requested_urls: list[str] = []

    def fake_login(_username: str, _password: str) -> MagicMock:
        session = MagicMock()

        def fake_get(url: str, **_kwargs: object) -> MagicMock:
            requested_urls.append(url)
            response = MagicMock()
            response.status_code = 200
            response.text = _SAMPLE_OMM_XML
            response.raise_for_status = lambda: None
            return response

        session.get.side_effect = fake_get
        return session

    monkeypatch.setattr(download_omm_mod, "_login_spacetrack", fake_login)

    ep.download_omm(
        norad_id=25544,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    # Exactly one Space-Track request for the single missing (contiguous) chunk.
    assert len(requested_urls) == 1
    assert "NORAD_CAT_ID/25544/" in requested_urls[0]
    assert "format/xml" in requested_urls[0]

    omm_files = list(tmp_path.glob("*.csv"))
    assert len(omm_files) == 1
    assert omm_files[0].name == "omm_20240101.csv"

    assert list(tmp_path.glob("*.xml")) == []

    content = omm_files[0].read_text()
    assert content.splitlines()[0].startswith("OBJECT_NAME,OBJECT_ID,EPOCH,")
    assert "ISS (ZARYA)" in content
    assert "<omm" not in content


def test_download_omm_live(tmp_path: Path, skip_if_unreachable: Callable[..., None]) -> None:
    skip_if_unreachable("https://www.space-track.org")

    username = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")

    if username is None or password is None:
        pytest.skip("SPACETRACK_USER/SPACETRACK_PASS not set; skipping live Space-Track test.")

    ep.download_omm(
        norad_id=25544,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    omm_files = list(tmp_path.glob("*.csv"))
    assert len(omm_files) == 1

    with omm_files[0].open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1
    assert rows[0]["NORAD_CAT_ID"] == "25544"
    assert list(tmp_path.glob("*.xml")) == []
