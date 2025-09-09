# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache 2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from astropy import units as u
from matplotlib import pyplot as plt
from swvo.io import RBMDataSet

import el_paso as ep
from examples.VanAllenProbes.process_hope_electrons import process_hope_electrons

# ruff: noqa: D103, S101, PLR2004

sat_str_list = ["a", "b"]
mag_field_list = ["TS04", "T89"]

@pytest.mark.parametrize("sat_str", sat_str_list)
@pytest.mark.parametrize("mag_field", mag_field_list)
@pytest.mark.visual
def test_mageph_rbsp(sat_str:Literal["a", "b"], mag_field:Literal["T89", "TS04"]):

    # process Lstar using el paso
    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0, hours=23, minutes=59)

    Path("tests/comparisons/raw_data").mkdir(exist_ok=True)
    Path("tests/comparisons/processed_data").mkdir(exist_ok=True)

    # process_hope_electrons(start_time, end_time, sat_str, "IRBEM/libirbem.so", mag_field,
    #                        raw_data_path="tests/comparisons/raw_data", processed_data_path="tests/comparisons/processed_data", num_cores=12)

    match mag_field:
        case "T89":
            mfm_enum = RBMDataSet.MfmEnum.T89
        case "TS04":
            mfm_enum = RBMDataSet.MfmEnum.T04s

    rbsp_data = RBMDataSet.RBMDataSet(start_time,
                                      end_time,
                                      Path("tests/comparisons/processed_data/"),
                                      "RBSPA",
                                      RBMDataSet.InstrumentEnum.HOPE,
                                      mfm_enum,
                                      verbose=True)

    rbsp_data_server = RBMDataSet.RBMDataSet(start_time,
                                      end_time,
                                      Path("/export/rbm6/data/data-dev/"),
                                      "RBSPA",
                                      RBMDataSet.InstrumentEnum.HOPE,
                                      mfm_enum,
                                      verbose=True)

    # load from mageph data
    match mag_field:
        case "T89":
            mag_field_str_data = "T89D"
        case "TS04":
             mag_field_str_data = "TS04D"
    file_name_stem = "rbsp" + sat_str + "_def_MagEphem_" + mag_field_str_data + "_YYYYMMDD_.{6}.h5"

    ep.download(start_time, end_time,
                save_path="tests/comparisons/raw_data",
                download_url=f"https://rbsp-ect.newmexicoconsortium.org/data_pub/rbsp{sat_str}/MagEphem/definitive/YYYY/",
                file_name_stem=file_name_stem,
                file_cadence="daily",
                method="request",
                skip_existing=True)

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="IsoTime",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Lstar",
            name_or_column="Lstar",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Alpha_eq",
            name_or_column="Alpha",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Kp",
            name_or_column="Kp",
            unit=u.dimensionless_unscaled,
        ),
    ]

    variables = ep.extract_variables_from_files(start_time,
                                                end_time,
                                                "daily",
                                                "tests/comparisons/raw_data",
                                                file_name_stem, extraction_infos)

    timestamps = [datetime.fromisoformat(str(t)[2:-2]).replace(tzinfo=timezone.utc).timestamp() for t in variables["Epoch"].get_data()]

    el_paso_timestamps = [t.timestamp() for t in rbsp_data.datetime]

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(19, 10))

    server_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        server_Lstar.append(np.interp(20, np.rad2deg(rbsp_data_server.alpha_local[it,:]), rbsp_data_server.Lstar[it,:]))

    server_timestamps = [t.timestamp() for t in rbsp_data_server.datetime]

    el_paso_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(20, np.rad2deg(rbsp_data.alpha_local[it,:]), rbsp_data.Lstar[it,:]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 20)

    ax1.plot(timestamps, variables["Lstar"].get_data()[:,deg_idx[0]], "k")
    ax1.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax1.plot(server_timestamps, server_Lstar, "b:")
    ax1.legend(["MagEph", "EL PASO", "Data server"])
    ax1.set_title("20 deg")

    server_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        server_Lstar.append(np.interp(50, np.rad2deg(rbsp_data_server.alpha_local[it,:]), rbsp_data_server.Lstar[it,:]))

    server_timestamps = [t.timestamp() for t in rbsp_data_server.datetime]

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    el_paso_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(50, np.rad2deg(rbsp_data.alpha_local[it,:]), rbsp_data.Lstar[it,:]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 50)

    ax2.plot(timestamps, variables["Lstar"].get_data()[:,deg_idx[0]], "k")
    ax2.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax2.plot(server_timestamps, server_Lstar, "b:")
    ax2.legend(["MagEph", "EL PASO", "Data server"])
    ax2.set_title("50 deg")

    server_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        server_Lstar.append(np.interp(70, np.rad2deg(rbsp_data_server.alpha_local[it,:]), rbsp_data_server.Lstar[it,:]))

    server_timestamps = [t.timestamp() for t in rbsp_data_server.datetime]

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    el_paso_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(70, np.rad2deg(rbsp_data.alpha_local[it,:]), rbsp_data.Lstar[it,:]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 70)

    ax3.plot(timestamps, variables["Lstar"].get_data()[:,deg_idx[0]], "k")
    ax3.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax3.plot(server_timestamps, server_Lstar, "b:")
    ax3.legend(["MagEph", "EL PASO", "Data server"])
    ax3.set_title("70 deg")

    plt.savefig(f"{Path(__file__).parent / f'mag_eph_test_{mag_field}.png'}")
