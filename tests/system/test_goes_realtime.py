# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swvo.io.RBMDataSet import InstrumentEnum, RBMNcDataSet

from examples.GOES.process_goes_realtime import process_goes_real_time


def test_goes_realtime_snapshot(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2025, 12, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.1)

    irbem_lib_path = Path(__file__).parent / "../../IRBEM/libirbem.so"

    processed_data_path = tmpdir

    process_goes_real_time(
        start_time=start_time,
        end_time=end_time,
        sat_str="primary",
        irbem_lib_path=irbem_lib_path,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
    )

    out_path = processed_data_path / "primary" / "goes_primary_20251201to20251231_T89.nc"
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "GOES" / "primary")

    goes_proc = RBMNcDataSet(start_time, end_time, tmpdir, "GOESPrimary", InstrumentEnum.MAGEDandEPEAD, "T89")

    goes_true = RBMNcDataSet(
        start_time,
        end_time,
        Path(__file__).parent / "data" / "processed",
        "GOESPrimary",
        InstrumentEnum.MAGEDandEPEAD,
        "T89",
    )

    assert goes_proc == goes_true, f"Different variables: {goes_proc.get_different_variables(goes_true)}"
