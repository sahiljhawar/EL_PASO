# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

#ruff: noqa: D103, INP001

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from swvo.io.RBMDataSet import RBMNcDataSet

from examples.Arase.arase_mepe import process_mepe_level_3
from examples.VanAllenProbes.process_ect_combined import process_ect_combined


@pytest.mark.parametrize("mag_field", ["T89", "TS04", "OP77Q"])
@pytest.mark.parametrize("save_strategy", ["netcdf"])
def test_arase_mepe_snapshot(mag_field: Literal["T89", "TS04", "OP77Q"],
                             save_strategy: Literal["DataOrg", "h5", "netcdf"],
                             tmpdir: Path = Path(".")) -> None:

    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.4, seconds=-1)

    irbem_lib_path = Path(__file__).parent / "../../IRBEM/libirbem.so"
    processed_data_path = tmpdir / "ARASE" / "arase"

    process_mepe_level_3(start_time, end_time, irbem_lib_path, mag_field,
                         raw_data_path = Path(__file__).parent / "data" / "raw",
                         processed_data_path = processed_data_path,
                         num_cores = 32,
                         cadence = timedelta(hours=1),
                         save_strategy = save_strategy,
                         use_level_3_orbit_data = False)

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    match save_strategy:
        case "DataOrg":
            out_path = processed_data_path / "arase_mepe" / "level_3" / f"{start_date:%Y%m%d}to{end_date:%Y%m%d}" / mag_field
            assert out_path.exists()
        case "h5":
            out_path = processed_data_path / f"arase_mepe-l3_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.h5"
            assert out_path.exists()
        case "netcdf":
            out_path = processed_data_path / f"arase_mepe-l3_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.nc"
            assert out_path.exists()

    arase_proc = RBMNcDataSet(start_time, end_time, tmpdir, "ARASE", "mepe", mag_field)
    arase_true = RBMNcDataSet(start_time, end_time, Path(__file__).parent / "data" / "processed",
                              "ARASE", "mepe", mag_field)

    assert arase_proc == arase_true
