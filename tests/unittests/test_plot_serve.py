# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``el-paso plot --serve``: `el_paso.cli.plot_serve` and the marimo app `el_paso.cli.plot_app`."""

from __future__ import annotations

import dataclasses
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u
from el_paso.cli import plot_serve
from el_paso.cli.plot_cli import PlotRequest
from el_paso.dataset.utils import python2matlab
from el_paso.recipes.rbsp import rbsp_mageis_electron_strategy
from el_paso.recipes.rbsp.process_rbsp_mageis_electrons import process_rbsp_mageis_electrons

if TYPE_CHECKING:
    from el_paso.typing import InternalName

START = datetime(2017, 10, 15, tzinfo=timezone.utc)
END = datetime(2017, 10, 16, tzinfo=timezone.utc)
ENERGIES_KEV = [100.0, 400.0, 1600.0]
ALPHAS_DEG = [10.0, 30.0, 60.0, 90.0]


def save_mageis_shaped_data(path: Path) -> None:
    """Write one day of small synthetic data for rbspa with the MagEIS strategy."""
    time_size = 48
    datetimes = [START + i * timedelta(minutes=30) for i in range(time_size)]
    flux_unit = (u.cm**2 * u.s * u.sr * u.keV) ** (-1)
    variables = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=np.array([python2matlab(t) for t in datetimes])),
        "FEDU": ep.Variable(original_unit=flux_unit, data=np.random.default_rng(0).uniform(1, 100, (time_size, 3, 4))),
        "Energy_FEDU": ep.Variable(original_unit=u.keV, data=np.tile(ENERGIES_KEV, (time_size, 1))),
        "Alpha": ep.Variable(original_unit=u.deg, data=np.tile(ALPHAS_DEG, (time_size, 1))),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.linspace(40.0, 60.0, time_size)),
        "L_star": ep.Variable(
            original_unit=u.dimensionless_unscaled, data=np.tile(np.linspace(2.0, 6.0, time_size)[:, None], (1, 4))
        ),
        "MLT": ep.Variable(original_unit=u.hour, data=np.linspace(0.0, 23.5, time_size)),
    }
    strategy = rbsp_mageis_electron_strategy(
        processed_data_path=path, mag_field="TS04", satellite="a", save_strategy="netcdf"
    )
    ep.save(variables, strategy, start_time=START, end_time=END, time_var=variables["Epoch"])  # ty:ignore[invalid-argument-type]


def make_request(path: Path, **fields: Any) -> PlotRequest:  # noqa: ANN401
    """A plot request for the data of `save_mageis_shaped_data`, as the CLI resolves it."""
    defaults: dict[str, Any] = {
        "recipe": process_rbsp_mageis_electrons,
        "factory": rbsp_mageis_electron_strategy,
        "start_time": START,
        "end_time": END,
        "strategy_kwargs": [
            {"processed_data_path": path, "mag_field": "TS04", "satellite": "a", "save_strategy": "netcdf"}
        ],
        "variables": None,
        "output_dir": None,
        "filename": None,
        "show": False,
        "dry_run": False,
        "verbose": False,
        "plot_defaults": {},
        "serve": True,
    }
    return PlotRequest(**{**defaults, **fields})


@pytest.fixture
def mageis_dataset(tmp_path: Path) -> ep.DataSet:
    save_mageis_shaped_data(tmp_path)
    strategy = rbsp_mageis_electron_strategy(
        processed_data_path=tmp_path, mag_field="TS04", satellite="a", save_strategy="netcdf"
    )
    return ep.DataSet(strategy, START, END, verbose=False)


@pytest.mark.minimal
@pytest.mark.basic
def test_request_survives_the_round_trip_to_the_app(tmp_path: Path) -> None:
    plot_defaults = {"panels": ["Kp", "FEDU"], "FEDU": {"energies": [108, 470] * u.keV, "vmin": 1.0, "along": "L_star"}}
    request = make_request(
        tmp_path,
        variables=["FEDU", "L_star"],
        plot_defaults=plot_defaults,
        energies=[470.0] * u.keV,
        pitch_angle=[30.0, 90.0] * u.deg,
        along="L_star",
        colormap="fire",
        host="0.0.0.0",  # noqa: S104
        port=9000,
    )

    text = plot_serve.encode_request(dataclasses.replace(request, password="s3cret"))
    assert "s3cret" not in text  # the password never reaches the request file
    decoded = plot_serve.decode_request(text)
    assert decoded.password is None

    assert decoded.recipe is process_rbsp_mageis_electrons
    assert decoded.factory is rbsp_mageis_electron_strategy
    assert (decoded.start_time, decoded.end_time) == (START, END)
    assert decoded.strategy_kwargs == request.strategy_kwargs  # the path stays a Path
    assert decoded.variables == ["FEDU", "L_star"]
    assert decoded.energies is not None
    assert decoded.energies.to_value(u.keV).tolist() == [470.0]
    assert decoded.pitch_angle is not None
    assert decoded.pitch_angle.to_value(u.deg).tolist() == [30.0, 90.0]
    fedu = decoded.plot_defaults["FEDU"]
    assert fedu["energies"].to_value(u.keV).tolist() == [108.0, 470.0]
    assert (fedu["vmin"], fedu["along"]) == (1.0, "L_star")
    assert decoded.plot_defaults["panels"] == ["Kp", "FEDU"]
    assert (decoded.along, decoded.colormap, decoded.serve, decoded.host, decoded.port) == (
        "L_star",
        "fire",
        True,
        "0.0.0.0",  # noqa: S104
        9000,
    )


@pytest.mark.minimal
@pytest.mark.basic
def test_request_needs_importable_functions(tmp_path: Path) -> None:
    def local_factory(**kwargs: object) -> ep.SavingStrategy:
        raise NotImplementedError

    with pytest.raises(TypeError, match="not a module-level function"):
        plot_serve.encode_request(make_request(tmp_path, factory=local_factory))


@pytest.mark.minimal
@pytest.mark.basic
def test_snap_picks_the_nearest_channels_without_duplicates() -> None:
    channels = [10.0, 30.0, 60.0, 90.0]
    assert plot_serve.snap(channels, [43.0]) == [30.0]
    assert plot_serve.snap(channels, [50.0, 89.0, 61.0]) == [60.0, 90.0]
    assert plot_serve.snap([], [45.0]) == []


@pytest.mark.minimal
@pytest.mark.basic
def test_spectral_variables_and_their_channels(mageis_dataset: ep.DataSet) -> None:
    variables: list[InternalName] = ["L_star", "FEDU", "B_Eq", "MLT"]
    assert plot_serve.spectral_variables(mageis_dataset, variables) == ["FEDU"]
    assert plot_serve.energy_channels(mageis_dataset, "FEDU").tolist() == ENERGIES_KEV
    assert plot_serve.energy_channels(mageis_dataset, "L_star").size == 0


@pytest.mark.minimal
@pytest.mark.basic
def test_start_values_come_from_the_recipe_and_the_command_line() -> None:
    recipe = {"FEDU": {"energies": [108, 470] * u.keV, "pitch_angle": 90 * u.deg, "vmin": 1.0, "along": "L_star"}}
    overrides = {"energies": None, "pitch_angle": [30.0, 60.0] * u.deg, "vmin": None, "vmax": 1e6}

    starts = plot_serve.start_values(["FEDU"], recipe, overrides, None)

    assert starts == {
        "energies_kev": [108.0, 470.0],
        "pitch_angles_deg": [30.0, 60.0],  # the command line wins
        "vmin": 1.0,
        "vmax": 1e6,
        "along": "L_star",
    }
    assert plot_serve.start_values([], recipe, overrides, "L_m")["along"] == "L_m"


@pytest.mark.minimal
@pytest.mark.basic
def test_widgets_set_the_spectral_variables_and_leave_the_others() -> None:
    recipe = {"FEDU": {"vmin": 1.0, "along": "L_star"}, "L_star": {"vmax": 7.0}}
    overrides = {"energies": None, "pitch_angle": None, "vmin": None, "vmax": 8.0}

    settings = plot_serve.widget_panel_settings(
        ["FEDU", "L_star"],
        ["FEDU"],
        recipe,
        overrides,
        energies_kev=[470.0, 108.0],
        pitch_angles_deg=[30.0, 90.0],
        color_limits=None,
        along="L_m",
    )

    fedu = settings["FEDU"]
    assert fedu["energies"].to_value(u.keV).tolist() == [108.0, 470.0]
    assert fedu["pitch_angle"].to_value(u.deg).tolist() == [30.0, 90.0]
    assert fedu["along"] == "L_m"
    assert "vmin" not in fedu  # auto colour limits drop the recipe's
    assert settings["L_star"] == {"vmax": 8.0}  # the recipe's, with the command line on top

    one = plot_serve.widget_panel_settings(
        ["FEDU"],
        ["FEDU"],
        recipe,
        overrides,
        energies_kev=[],
        pitch_angles_deg=[45.0],
        color_limits=(1.0, 100.0),
        along=None,
    )
    assert one["FEDU"]["pitch_angle"] == 45 * u.deg  # one angle stays a scalar
    assert (one["FEDU"]["vmin"], one["FEDU"]["vmax"]) == (1.0, 100.0)
    assert "energies" not in one["FEDU"]
    assert "along" not in one["FEDU"]


@pytest.mark.minimal
@pytest.mark.basic
def test_app_command(tmp_path: Path) -> None:
    command = plot_serve.app_command(tmp_path / "request.json", host="127.0.0.1", port=2800, open_browser=False)

    assert command[:4] == [sys.executable, "-m", "marimo", "run"]
    assert command[4] == str(plot_serve.APP_PATH)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "2800"
    assert "--headless" in command
    assert "--no-token" in command  # no login without a password
    # the app's own arguments come after `--`
    assert command[command.index("--") :] == ["--", "--request", str(tmp_path / "request.json")]
    assert "--headless" not in plot_serve.app_command(tmp_path / "r.json", host="127.0.0.1", port=1)

    protected = plot_serve.app_command(
        tmp_path / "r.json", host="127.0.0.1", port=1, password_file=tmp_path / "password"
    )
    assert protected[protected.index("--token") : protected.index("--token") + 3] == [
        "--token",
        "--token-password-file",
        str(tmp_path / "password"),
    ]
    assert "--no-token" not in protected


@pytest.mark.minimal
@pytest.mark.basic
def test_port_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert plot_serve.port_in_use("127.0.0.1", port)
    assert not plot_serve.port_in_use("127.0.0.1", port)
    assert plot_serve.is_local_host("127.0.0.1")
    assert not plot_serve.is_local_host("0.0.0.0")  # noqa: S104


@pytest.mark.minimal
@pytest.mark.basic
def test_app_is_a_valid_marimo_notebook() -> None:
    pytest.importorskip("marimo")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "marimo", "check", str(plot_serve.APP_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.minimal
@pytest.mark.basic
def test_app_draws_the_request_with_pitch_angles_snapped_to_the_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("marimo")
    save_mageis_shaped_data(tmp_path)
    request = make_request(
        tmp_path,
        variables=["FEDU", "B_Eq"],
        along="L_star",
        energies=[400.0] * u.keV,
        pitch_angle=[25.0, 90.0] * u.deg,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(plot_serve.encode_request(request), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["plot_app.py", "--", "--request", str(request_path)])
    monkeypatch.setattr(ep, "setup_logging", lambda *args, **kwargs: None)

    from el_paso.cli import plot_app  # noqa: PLC0415

    outputs, defs = plot_app.app.run()

    # the slider offers exactly the pitch-angle channels of the files; 25° snaps to 30°
    assert defs["angle_options"].tolist() == ALPHAS_DEG
    assert defs["pitch_angles"].value == [30.0, 90.0]
    assert defs["pitch_angle"].value == 30.0
    assert sorted(defs["energies"].value) == [400.0]
    assert defs["along"].value == "L_star"
    figure_output = outputs[-1].text
    assert "drawn in" in figure_output
    assert "Could not draw" not in figure_output


@pytest.mark.minimal
@pytest.mark.basic
def test_serve_hands_marimo_a_private_password_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], *, check: bool) -> None:
        password_file = Path(command[command.index("--token-password-file") + 1])
        seen["password"] = password_file.read_text(encoding="utf-8")
        seen["mode"] = password_file.stat().st_mode & 0o777
        seen["request"] = Path(command[-1]).read_text(encoding="utf-8")
        seen["command"] = command
        seen["file"] = password_file

    monkeypatch.setattr(plot_serve, "port_in_use", lambda host, port: False)
    monkeypatch.setattr(plot_serve.subprocess, "run", fake_run)

    plot_serve.serve(make_request(tmp_path), password="s3cret")

    assert seen["password"] == "s3cret"
    assert seen["mode"] == 0o600
    assert "s3cret" not in seen["request"]
    assert "s3cret" not in " ".join(seen["command"])  # not visible in the process list
    assert not seen["file"].exists()  # removed when the app stops
