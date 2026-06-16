# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from astropy import units as u

import el_paso as ep


def test_esa_api(tmp_path: Path, skip_if_unreachable: Callable[..., None]):

    skip_if_unreachable("https://swe.ssa.esa.int", "https://sso.s2p.esa.int")

    client_id = os.environ.get("ESA_CLIENT_ID")
    client_secret = os.environ.get("ESA_CLIENT_SECRET")

    if client_id is None:
        msg = "Client ID not found!"
        raise ValueError(msg)

    if client_secret is None:
        msg = "Client secret not found!."
        raise ValueError(msg)

    start_time = datetime(2025, 3, 17, tzinfo=timezone.utc)
    end_time = datetime(2025, 3, 17, 1, tzinfo=timezone.utc)

    url = "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_edrsc_ngrm_spid204030252_science_ep_l1_gc_v3"
    file_name_stem = "EDRS-C_ngrm_YYYYMMDD_L1d.csv"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        file_cadence="daily",
        download_url=url,
        file_name_stem="",
        rename_file_name_stem=file_name_stem,
        method="esa_swe",
        authentication_info=(client_id, client_secret),
        skip_existing=False,
    )

    assert len(list(tmp_path.glob("*"))) == 1


@pytest.mark.basic
def test_request(tmp_path: Path, skip_if_unreachable: Callable[..., None]):

    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = datetime(2013, 3, 17, 1, tzinfo=timezone.utc)

    url = "https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/hope/pitchangle/rel04/YYYY/"
    file_name_stem = "rbspa_rel04_ect-hope-pa-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    files = list(tmp_path.glob("*"))
    assert len(files) == 1

    data_path = tmp_path / "2013" / "03"

    assert not data_path.exists()

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert data_path.exists()
    assert len(list(data_path.glob("*"))) == 1


def test_exit_after_download(caplog: pytest.LogCaptureFixture):

    # test if the programs exits; it should not
    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    ep.exit_after_download = True

    with pytest.raises(SystemExit) as sample_exception:
        ep.download(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(1999, 1, 1, tzinfo=timezone.utc),
            save_path="",
            download_url="",
            file_name_stem="",
            file_cadence="daily",
            method="request",
            skip_existing=True,
            sort_raw_files_by_time=True,
        )

    assert sample_exception.value.code == 1
    assert "Exiting after ep.download is completed!" in caplog.text

    ep.exit_after_download = False
    os.environ["EL_PASO_EXIT_AFTER_DOWNLOAD"] = "True"

    with pytest.raises(SystemExit) as sample_exception:
        ep.download(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(1999, 1, 1, tzinfo=timezone.utc),
            save_path="",
            download_url="",
            file_name_stem="",
            file_cadence="daily",
            method="request",
            skip_existing=True,
            sort_raw_files_by_time=True,
        )

    assert sample_exception.value.code == 1
    assert "Exiting after ep.download is completed!" in caplog.text

def test_skip_download_via_ep_flag(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    ep.skip_download = True

    was_called = False

    def spy_function() -> None:
        nonlocal was_called
        was_called = True

    monkeypatch.setattr(ep.utils, "enforce_utc_timezone", spy_function)

    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert not was_called
    assert "Skipping ep.download" in caplog.text


def test_skip_download_via_env_var(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    os.environ["EL_PASO_SKIP_DOWNLOAD"] = "True"

    was_called = False

    def spy_function() -> None:
        nonlocal was_called
        was_called = True

    monkeypatch.setattr(ep.utils, "enforce_utc_timezone", spy_function)

    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert not was_called
    assert "Skipping ep.download" in caplog.text
