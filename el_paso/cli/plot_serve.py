# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Serve ``el-paso plot`` as an interactive web app (``--serve``), built with marimo.

The app (``plot_app.py`` next to this module) draws the same matplotlib figure as ``el-paso plot``,
through :func:`el_paso.cli.plot_figure.build_figure`, and shows it with ``marimo.mpl.interactive`` for
zoom and pan. Widgets change the time range, the panels and the settings of the variables with an
energy dimension (energies, pitch angles, colour limits, ``along``); the pitch angles snap to the
channels in the files.

``--serve`` writes the resolved `PlotRequest` to a JSON file and runs ``marimo run`` on the app in a
subprocess, which reads the request back. marimo is an optional dependency (``el_paso[serve]``) and
is only imported by the app.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from astropy import units as u
    from numpy.typing import NDArray

    from el_paso.cli.plot_cli import PlotRequest
    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName

logger = logging.getLogger("el_paso.cli")

DEFAULT_HOST = "127.0.0.1"
"""Host the app binds to by default. The app has no authentication, so it is only reachable locally."""

DEFAULT_PORT = 2718
"""Port the app listens on by default (marimo's own default)."""

APP_PATH = Path(__file__).with_name("plot_app.py")
"""The marimo app that ``--serve`` runs."""

INSTALL_HINT = "pip install 'el_paso[serve]'"
"""How to install the optional dependencies of ``--serve``."""

PASSWORD_ENV_VAR = "EL_PASO_SERVE_PASSWORD"  # noqa: S105 - the name of the variable, not a password
"""Environment variable ``--password`` is read from, which keeps the password out of the shell history."""

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_CALLABLE_FIELDS = ("recipe", "factory")
"""`PlotRequest` fields holding functions, encoded as ``module:qualname``."""

_DROPPED_FIELDS = ("availability", "password")
"""`PlotRequest` fields the app does not need (the password goes to marimo, never into the request file)."""


def marimo_available() -> bool:
    """Whether marimo, the optional dependency of ``--serve``, is installed."""
    return importlib.util.find_spec("marimo") is not None


def is_local_host(host: str) -> bool:
    """Whether `host` is only reachable from this machine."""
    return host in _LOCAL_HOSTS


def _callable_name(function: Callable[..., Any]) -> str:
    """``module:qualname`` of a module-level function, for `_import_callable`.

    Raises:
        TypeError: If `function` has no qualified name, e.g. a `functools.partial`.
    """
    qualname = getattr(function, "__qualname__", "")
    if not qualname or "<locals>" in qualname:
        msg = f"{function!r} is not a module-level function, so the app cannot import it."
        raise TypeError(msg)
    return f"{function.__module__}:{qualname}"


def _import_callable(name: str) -> Callable[..., Any]:
    module_name, _, qualname = name.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _encode(value: Any) -> Any:  # noqa: ANN401
    """`value` as JSON-compatible data, with quantities, paths and times tagged so that `_decode` restores them."""
    if hasattr(value, "unit") and hasattr(value, "value"):  # an astropy quantity
        magnitude = getattr(value.value, "tolist", lambda: value.value)()
        return {"__quantity__": magnitude, "unit": value.unit.to_string()}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:  # noqa: ANN401
    """Invert `_encode`."""
    if isinstance(value, dict):
        if "__quantity__" in value:
            from astropy import units as u  # noqa: PLC0415

            return u.Quantity(value["__quantity__"], value["unit"])
        if "__path__" in value:
            return Path(value["__path__"])
        if "__datetime__" in value:
            return datetime.fromisoformat(value["__datetime__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def encode_request(request: PlotRequest) -> str:
    """Write a plot request as JSON for the app; see `decode_request`.

    The recipe and the strategy factory are stored by import path, so they must be importable
    module-level functions, as those of registered recipes are.

    Args:
        request (PlotRequest): The resolved plot request.

    Returns:
        str: The JSON text.
    """
    fields: dict[str, Any] = {}
    for field in dataclasses.fields(request):
        if field.name in _DROPPED_FIELDS:
            continue
        value = getattr(request, field.name)
        fields[field.name] = _callable_name(value) if field.name in _CALLABLE_FIELDS else _encode(value)
    return json.dumps(fields, indent=1)


def decode_request(text: str) -> PlotRequest:
    """Read a plot request written by `encode_request`.

    Args:
        text (str): The JSON text.

    Returns:
        PlotRequest: The request, with the recipe and the factory imported again.
    """
    from el_paso.cli.plot_cli import PlotRequest  # noqa: PLC0415

    fields = {name: _decode(value) for name, value in json.loads(text).items()}
    for name in _CALLABLE_FIELDS:
        fields[name] = _import_callable(fields[name])
    return PlotRequest(**fields)


def app_command(
    request_path: Path,
    *,
    host: str,
    port: int,
    open_browser: bool = True,
    password_file: Path | None = None,
) -> list[str]:
    """The command line that serves the app for a request written to `request_path`.

    With `password_file`, marimo asks for the password in it before showing the app; the password
    itself is not on the command line, where other users of the machine could read it.
    """
    auth = ["--token", "--token-password-file", str(password_file)] if password_file else ["--no-token"]
    return [
        sys.executable,
        "-m",
        "marimo",
        "run",
        str(APP_PATH),
        "--host",
        host,
        "--port",
        str(port),
        *auth,
        *([] if open_browser else ["--headless"]),
        "--",
        "--request",
        str(request_path),
    ]


def port_in_use(host: str, port: int) -> bool:
    """Whether something already listens on `host`:`port`, so that the app could not bind to it."""
    import socket  # noqa: PLC0415

    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


class ServeError(RuntimeError):
    """The app could not be started or stopped with an error; the message says why."""


def serve(
    request: PlotRequest, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, password: str | None = None
) -> None:
    """Run the app for `request` until it is stopped (Ctrl+C), opening it in the browser.

    Args:
        request (PlotRequest): The resolved plot request.
        host (str): Host to bind to. Anything but a local host makes the app reachable from other
            machines, which is warned about without a `password`.
        port (int): Port to listen on.
        password (str | None): Password the app asks for, or ``None`` for no login. It is passed to
            marimo in a file only the current user can read, which is deleted when the app stops.

    Raises:
        ServeError: If the port is taken, or marimo exits with an error.
    """
    if port_in_use(host, port):
        msg = f"Port {port} on {host} is already in use; stop what runs there or pass another --port."
        raise ServeError(msg)
    if not is_local_host(host) and not password:
        logger.warning(
            f"Serving on {host} without --password: anyone who can reach {host}:{port} "
            "can see the data and use the app."
        )
    # TemporaryDirectory is created with mode 0o700, so only the current user can read the files in it
    with tempfile.TemporaryDirectory(prefix="el_paso_plot_") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_text(encode_request(request), encoding="utf-8")
        password_file = None
        if password:
            password_file = Path(directory) / "password"
            password_file.touch(mode=0o600)
            password_file.write_text(password, encoding="utf-8")
        login = "with a password" if password else "without a login"
        logger.info(f"Serving the plot app on http://{host}:{port} {login} (Ctrl+C to stop).")
        command = app_command(request_path, host=host, port=port, password_file=password_file)
        try:
            subprocess.run(command, check=True)  # noqa: S603
        except KeyboardInterrupt:
            logger.info("Stopped the plot app.")
        except subprocess.CalledProcessError as error:
            msg = f"The plot app stopped with exit code {error.returncode}; its output above says why."
            raise ServeError(msg) from error


# Helpers of the app, kept here so that they can be tested without marimo.


def spectral_variables(dataset: DataSet, variables: Sequence[InternalName]) -> list[InternalName]:
    """The variables the app's energy and pitch-angle widgets apply to: those with an energy dimension.

    Args:
        dataset (DataSet): A dataset of the request, for its data standard.
        variables (Sequence[InternalName]): The plotted variables.

    Returns:
        list[InternalName]: The variables whose first non-time dimension is energy, in the given order.
    """
    from el_paso.dataset.plot import _is_dimension  # noqa: PLC0415

    standard = dataset.saving_strategy.data_standard
    selected: list[InternalName] = []
    for name in variables:
        if name not in standard.variable_infos:
            continue
        dependencies = standard.get_dependencies(name)[1:]
        if dependencies and _is_dimension(dependencies[0], "Energy"):
            selected.append(name)
    return selected


def energy_channels(dataset: DataSet, name: InternalName) -> NDArray[np.float64]:
    """The energy channels of a variable in keV (median over time, to three significant digits), sorted.

    Args:
        dataset (DataSet): The dataset that holds the variable.
        name (InternalName): A variable with an energy dimension, e.g. ``"FEDU"``.

    Returns:
        NDArray[np.float64]: The channels, empty if the variable has no energy dimension or no data.
    """
    import numpy as np  # noqa: PLC0415
    from astropy import units as u  # noqa: PLC0415

    from el_paso.dataset.plot import _get_median_coordinate, _is_dimension  # noqa: PLC0415

    standard = dataset.saving_strategy.data_standard
    dependencies = standard.get_dependencies(name)[1:]
    if not dependencies or not _is_dimension(dependencies[0], "Energy"):
        return np.asarray([], dtype=float)
    coordinate_name, channels = _get_median_coordinate(dataset, dependencies[0])
    unit = standard.variable_infos[coordinate_name].unit  # ty:ignore[invalid-argument-type]
    kev = (np.asarray(channels, dtype=float).ravel() * unit).to_value(u.keV)
    return np.unique([float(f"{value:.3g}") for value in kev if np.isfinite(value)])


def snap(channels: Sequence[float] | NDArray[np.float64], values: Sequence[float]) -> list[float]:
    """Each of `values` replaced by the nearest of `channels`, without duplicates, in the order of `values`.

    This is the channel `DataSet.plot` shows for a requested value, so a widget that offers only
    channels shows what is drawn.
    """
    snapped: list[float] = []
    for value in values:
        if len(channels) == 0:
            break
        nearest = float(min(channels, key=lambda channel: abs(float(channel) - value)))
        if nearest not in snapped:
            snapped.append(nearest)
    return snapped


def widget_panel_settings(
    variables: Sequence[InternalName],
    spectral: Sequence[InternalName],
    recipe_settings: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any],
    *,
    energies_kev: Sequence[float],
    pitch_angles_deg: Sequence[float],
    color_limits: tuple[float, float] | None,
    along: InternalName | None,
) -> dict[str, dict[str, Any]]:
    """The per-variable settings of the app: the request's, with the widget values for the `spectral` variables.

    The other variables keep the recipe's settings with the command-line overrides applied, so the
    widgets only change the panels they are about.

    Args:
        variables (Sequence[InternalName]): The plotted variables.
        spectral (Sequence[InternalName]): The variables the widgets apply to (see `spectral_variables`).
        recipe_settings (Mapping[str, Mapping[str, Any]]): The recipe's settings per variable.
        overrides (Mapping[str, Any]): The command-line settings for every panel; ``None`` values are ignored.
        energies_kev (Sequence[float]): Selected energies; empty for all channels (a spectrogram without `along`).
        pitch_angles_deg (Sequence[float]): Selected pitch angles: one without `along`, one per column with it.
        color_limits (tuple[float, float] | None): ``(vmin, vmax)``, or ``None`` for the 5th and 95th percentiles.
        along (InternalName | None): Coordinate on the y axis of the spectral variables, or ``None`` for time.

    Returns:
        dict[str, dict[str, Any]]: Settings for `build_figure`'s `panel_settings`, with nothing left to override.
    """
    from astropy import units as u  # noqa: PLC0415

    given = {key: value for key, value in overrides.items() if value is not None}
    settings: dict[str, dict[str, Any]] = {}
    for name in variables:
        if name not in spectral:
            settings[name] = {**recipe_settings.get(name, {}), **given}
            continue
        widget: dict[str, Any] = {}
        if energies_kev:
            widget["energies"] = u.Quantity(sorted(energies_kev), u.keV)
        if pitch_angles_deg:
            angles = list(pitch_angles_deg)
            widget["pitch_angle"] = u.Quantity(angles[0] if len(angles) == 1 else angles, u.deg)
        if color_limits is not None:
            widget["vmin"], widget["vmax"] = color_limits
        if along is not None:
            widget["along"] = along
        settings[name] = widget
    return settings


def start_values(
    spectral: Sequence[InternalName],
    recipe_settings: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any],
    along: InternalName | None,
) -> dict[str, Any]:
    """The start values of the app's widgets: the resolved settings of the first spectral variable.

    Args:
        spectral (Sequence[InternalName]): The variables the widgets apply to.
        recipe_settings (Mapping[str, Mapping[str, Any]]): The recipe's settings per variable.
        overrides (Mapping[str, Any]): The command-line settings for every panel.
        along (InternalName | None): The command-line ``--along-*`` coordinate.

    Returns:
        dict[str, Any]: ``energies_kev`` and ``pitch_angles_deg`` (lists, empty for the defaults),
        ``vmin``, ``vmax`` and ``along`` (``None`` for the defaults).
    """
    from astropy import units as u  # noqa: PLC0415

    from el_paso.dataset.plot import resolve_panel_settings  # noqa: PLC0415

    if not spectral:
        return {"energies_kev": [], "pitch_angles_deg": [], "vmin": None, "vmax": None, "along": along}
    name = spectral[0]
    settings = resolve_panel_settings([name], recipe_settings, {**overrides, "along": along})[name]
    energies: u.Quantity | None = settings.energies
    pitch_angle: u.Quantity | None = settings.pitch_angle
    return {
        "energies_kev": [] if energies is None else [float(e) for e in energies.to_value(u.keV)],
        "pitch_angles_deg": [] if pitch_angle is None else [float(a) for a in pitch_angle.to_value(u.deg).ravel()],
        "vmin": settings.vmin,
        "vmax": settings.vmax,
        "along": settings.along,
    }
