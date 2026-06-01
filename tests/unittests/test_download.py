import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from astropy import units as u

import el_paso as ep


@pytest.mark.basic
def test_esa_api(tmp_path: Path):

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
