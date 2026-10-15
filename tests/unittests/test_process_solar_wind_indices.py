# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u
from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputResult
from el_paso.recipes.model.process_solar_wind_indices import (
    process_solar_wind_indices,
    solar_wind_indices_strategy,
)


@pytest.mark.basic
def test_strategy_saves_only_epoch_and_the_indices_the_model_requires(tmp_path: Path) -> None:
    t89 = solar_wind_indices_strategy(tmp_path, "T89")
    ts04 = solar_wind_indices_strategy(tmp_path, "TS04")

    assert [f.names_to_save for f in t89.output_files] == [["Epoch", "Kp"]]
    assert len(ts04.output_files) == 1
    assert {"Epoch", "Kp", "Dst", "Pdyn", "ByIMF", "BzIMF"} <= set(ts04.output_files[0].names_to_save)


@pytest.mark.basic
def test_recipe_writes_a_monthly_solar_wind_indices_file(tmp_path: Path) -> None:
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)
    num_samples = 24
    kp_var = ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full(num_samples, 3.0))

    with patch(
        "el_paso.save.construct_maginput",
        return_value=MagInputResult(maginput={}, indices_solar_wind={"Kp": kp_var}),
    ) as mock_construct:
        process_solar_wind_indices(
            start_time, end_time, mag_field="T89", processed_data_path=tmp_path, bin_cadence=timedelta(hours=1)
        )

    # the time grid handed to the index loader covers [start_time, end_time) at the requested cadence
    time_var = mock_construct.call_args.args[0]
    assert time_var.get_data(ep.units.posixtime).shape == (num_samples,)

    file_path = tmp_path / "SolarWindIndices_20130101to20130131_T89.nc"
    assert file_path.exists()
