# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

import el_paso as ep
from el_paso.dataset import GFZDataSet
from el_paso.recipes.probav import process_ept_electron_fluxes


@pytest.mark.basic
def test_probav_ept(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:

    skip_if_unreachable(
        "https://sso.s2p.esa.int", "https://sso-csr-ucl-ac-be.content.swe.s2p.esa.int", "https://spdf.gsfc.nasa.gov"
    )

    client_id = os.environ.get("ESA_CLIENT_ID")
    client_secret = os.environ.get("ESA_CLIENT_SECRET")

    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_ept_electron_fluxes(
        start_time=start_time,
        end_time=end_time,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
        client_id=client_id,
        client_secret=client_secret,
    )

    out_path = (
        processed_data_path
        / "PROBAV"
        / "probav"
        / f"probav_ept_{start_time:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()
