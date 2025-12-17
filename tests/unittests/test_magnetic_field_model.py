from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import el_paso as ep

mag_field_list = ["OP77", "T89", "T01s", "TS04"]
@pytest.mark.parametrize("mag_field", mag_field_list)
def test_magnetic_field(mag_field: Literal["T89", "OP77", "TS04", "T01s"]):

    true_data = {
        "OP77": (92.31, 97.28, 106.8),
        "T89": (82.09, 90.64, 95.8),
        "T01s": (39.35, 159.48, 335.31),
        "TS04": (27.82, 91.7, 155.64),
    }

    start_time = datetime(2024, 5, 10, 16, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 11, 0, tzinfo=timezone.utc)

    time_list:list[float] = []
    curr_time = start_time

    while curr_time <= end_time:
        time_list.append(curr_time.timestamp())
        curr_time += timedelta(minutes=30)

    time_var = ep.Variable(data=np.asarray(time_list), original_unit=ep.units.posixtime)

    xgeo_data = np.tile(np.array([0, 6.6, 0]), (len(time_var.get_data()), 1))
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_local", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=time_var,
        xgeo_var=xgeo_var,
        variables_to_compute=variables_to_compute,
        irbem_lib_path=Path(__file__).parent.parent.parent / "IRBEM" / "libirbem.so",
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=12,
    )

    mag_field_data = magnetic_field_variables["B_local_" + mag_field].get_data("nT")
    min_value = np.round(mag_field_data.min(), 2)
    mean_value = np.round(mag_field_data.mean(), 2)
    max_value = np.round(mag_field_data.max(), 2)

    assert min_value == true_data[mag_field][0]
    assert mean_value == true_data[mag_field][1]
    assert max_value == true_data[mag_field][2]

