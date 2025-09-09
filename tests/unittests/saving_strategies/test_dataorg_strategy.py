# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache 2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep

rng = np.random.default_rng(1337)

def test_dataorg_strategy_withtout_time_error(tmp_path: Path):

    variables_to_save = {
        "time": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((20,21))),
        "Flux": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((10,11))),
        "Lstar": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((51,))),
    }

    strategy = ep.saving_strategies.DataOrgStrategy(base_data_path=tmp_path,
                                                    mission="mission",
                                                    satellite="satellite",
                                                    instrument="instrument",
                                                    kext="T89",
                                                    file_format=".mat")

    with pytest.raises(ValueError, match="start_time"):
        ep.save(variables_to_save, strategy)

@pytest.mark.parametrize("file_format", [".mat", ".pickle", ".h5"])
def test_basic_dataorg_strategy(tmp_path: Path, file_format:str):

    start_time = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time = [start_time]
    for _ in range(100):
        time.append(time[-1] + timedelta(hours=1))
    end_time = time[-1]

    time = [t.timestamp() for t in time]
    time_var = ep.Variable(original_unit=ep.units.posixtime, data=np.asarray(time))

    variables_to_save = {
        "time": time_var,
        "Flux": ep.Variable(original_unit=(u.cm**2 * u.s * u.sr * u.keV)**(-1), data=rng.normal(size=(len(time),11,5))),
        "Lstar": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal(size=(len(time),5))),
    }

    strategy = ep.saving_strategies.DataOrgStrategy(base_data_path=tmp_path,
                                                    mission="mission",
                                                    satellite="satellite",
                                                    instrument="instrument",
                                                    kext="T89",
                                                    file_format=file_format)

    ep.save(variables_to_save, strategy, start_time=start_time, end_time=end_time, time_var=time_var)

    save_path = tmp_path / "MISSION" / "satellite" / "Processed_Mat_Files" / ("satellite_instrument_20150101to20150131_flux_ver4" + file_format)
    assert save_path.exists()