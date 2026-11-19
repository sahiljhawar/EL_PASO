# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Quick-look plotting for `DataSet`, attached as `DataSet.plot`.

The plot type follows from the dimensions of a variable in the data standard (layer 1 of the plot
defaults): a line for time series, one line per component for vectors, and a spectrogram for
variables that depend on energy and/or pitch angle. Panel settings (energies, pitch angle and colour
limits) are layered on top: per-variable settings such as a recipe's ``PLOT_DEFAULTS`` (layer 2),
then the keyword arguments of `plot`, which the CLI fills from ``--energy``, ``--pitch-angle``,
``--vmin`` and ``--vmax`` (layer 3) and which apply to every panel.

With ``along`` (one of `ALONG_COORDINATES`, from ``--along-L-star``, ``--along-L-m`` or
``--along-R-eq``), every panel is instead a scatter plot over time with that coordinate on the y axis
and the variable's value as colour. A variable is then drawn as a grid of such panels: one row per
selected energy (or per channel), and one column per selected pitch angle, since the variable and
L* or Lm both depend on pitch angle. matplotlib is imported when `plot` is called.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

from el_paso.one_of import alternatives_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from el_paso.data_standard import Dependency
    from el_paso.dataset import DataSet
    from el_paso.typing import FixedDimensionName, InternalName

logger = logging.getLogger(__name__)

_FIELD_MODEL_VARIABLES = frozenset({"L_star", "L_m", "MLT", "MLT_Eq", "B_Eq", "R_Eq", "InvMu", "InvK", "MLat", "PSD"})
"""Variables computed with a magnetic field model, whose labels name the strategy's ``mag_field``."""

_LOG_COLOR_VARIABLES = frozenset(
    {"FEDU", "FEDU_SA", "FEIU", "FEIU_SA", "FEDO", "FPDU", "FPDU_SA", "FIDU", "FIDU_SA", "PSD"}
)
"""Flux-like variables that get a logarithmic colour scale."""

_DEFAULT_PITCH_ANGLE = 90.0 * u.deg
"""Pitch angle nearest to which spectrograms with an energy and a pitch-angle dimension are cut by default."""

_ENERGY_TOLERANCE = 0.1
"""Largest distance of a matched energy channel from the requested energy, as a fraction of the latter."""

_COLOR_LIMIT_PERCENTILES = (5.0, 95.0)
"""Percentiles of a spectrogram's finite values that set its colour limits."""

_PANEL_HEIGHT_INCHES = 2.5

DEFAULT_COLORMAP = "viridis"
"""Colormap of spectrograms and scatter panels: a colorcet name, or else a matplotlib colormap name."""

PANEL_SETTING_NAMES = ("energies", "pitch_angle", "vmin", "vmax", "along")
"""Settings a panel accepts, per variable (``panel_settings``) or for all panels (keyword arguments)."""

ALONG_COORDINATES = ("L_star", "L_m", "R_Eq")
"""Coordinates that the ``along`` setting puts on the y axis of scatter panels."""


@dataclasses.dataclass(frozen=True)
class PanelSettings:
    """The settings of one panel after layering; ``None`` means the default of the variable's plot type.

    Attributes:
        energies: Energies to show, for variables with an energy dimension.
        pitch_angle: Pitch-angle cut, for variables with a pitch-angle dimension, each angle matched to the
            nearest channel. Without it, variables with an energy and a pitch-angle dimension use 90°, and
            those with only a pitch-angle dimension (e.g. ``L_star``) show all pitch angles. Several
            angles (a 1-d quantity) give one column of panels each with `along`, and one line each for
            variables with only a pitch-angle dimension; a spectrogram can only be cut at one.
        vmin: Lower colour limit of spectrograms and scatter panels (y-limit of energy line panels),
            in the unit of the plotted variable.
        vmax: Upper colour limit of spectrograms and scatter panels (y-limit of energy line panels),
            in the unit of the plotted variable.
        along: One of `ALONG_COORDINATES` to draw the variable as scatter panels with this coordinate on
            the y axis, instead of the usual plot type of its dimensions.
    """

    energies: u.Quantity | None = None
    pitch_angle: u.Quantity | None = None
    vmin: float | None = None
    vmax: float | None = None
    along: InternalName | None = None


def _check_setting(key: str, value: object) -> Any:  # noqa: ANN401
    """Validate one panel setting and return it in canonical form (energies as a 1-d quantity, limits as floats).

    Raises:
        TypeError: If the value has the wrong type, e.g. an energy without a unit.
        ValueError: If the value is out of range.
    """
    if key == "energies":
        energies = np.atleast_1d(_as_quantity(value, u.keV, key))
        if energies.ndim != 1 or energies.size == 0:
            msg = f"energies must be a scalar or 1-d quantity with at least one value, got shape {energies.shape}."
            raise ValueError(msg)
        if not np.all(energies.to_value(u.keV) > 0):
            msg = f"energies must be positive, got {energies}."
            raise ValueError(msg)
        return energies
    if key == "pitch_angle":
        pitch_angle = _as_quantity(value, u.deg, key)
        degrees = np.atleast_1d(pitch_angle.to_value(u.deg))
        if pitch_angle.ndim > 1 or degrees.size == 0 or not np.all((degrees >= 0) & (degrees <= 180)):
            msg = f"pitch_angle must be one angle or a 1-d list of angles between 0 and 180 degrees, got {pitch_angle}."
            raise ValueError(msg)
        return pitch_angle
    if key == "along":
        if value not in ALONG_COORDINATES:
            msg = f"along must be one of {', '.join(ALONG_COORDINATES)}, got {value!r}."
            raise ValueError(msg)
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{key} must be a plain number in the unit of the plotted variable, got {value!r}."
        raise TypeError(msg)
    return float(value)


def _check_settings(settings: Mapping[str, object]) -> dict[str, Any]:
    """Validate the settings of one layer, dropping those that are ``None``.

    Raises:
        TypeError: If a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range, or `vmin` is not below `vmax`.
    """
    unknown = sorted(set(settings) - set(PANEL_SETTING_NAMES))
    if unknown:
        msg = f"unknown setting(s) {', '.join(map(repr, unknown))}; known are {', '.join(PANEL_SETTING_NAMES)}."
        raise TypeError(msg)
    checked = {key: _check_setting(key, value) for key, value in settings.items() if value is not None}
    _check_limits(checked)
    return checked


def _check_limits(settings: Mapping[str, Any]) -> None:
    """Raise `ValueError` if both colour limits are set and `vmin` is not below `vmax`."""
    vmin, vmax = settings.get("vmin"), settings.get("vmax")
    if vmin is not None and vmax is not None and not vmin < vmax:
        msg = f"vmin ({vmin:g}) must be below vmax ({vmax:g})."
        raise ValueError(msg)


def check_panel_settings(panel_settings: Mapping[str, Mapping[str, object]]) -> None:
    """Validate per-variable panel settings, e.g. those of a recipe's ``PLOT_DEFAULTS``.

    Variable names are not checked against a data standard here; settings for variables that are
    not plotted are ignored by `plot`.

    Args:
        panel_settings (Mapping[str, Mapping[str, object]]): Settings per internal variable name,
            each with keys from :data:`PANEL_SETTING_NAMES`.

    Raises:
        TypeError: If an entry is not a mapping, or a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range, or `vmin` is not below `vmax`.
    """
    from collections.abc import Mapping  # noqa: PLC0415

    for name, settings in panel_settings.items():
        if not isinstance(settings, Mapping):
            msg = f"{name}: settings must be a mapping of {', '.join(PANEL_SETTING_NAMES)}, got {settings!r}."
            raise TypeError(msg)
        try:
            _check_settings(settings)
        except (TypeError, ValueError) as error:
            msg = f"{name}: {error}"
            raise type(error)(msg) from error


def resolve_panel_settings(
    variables: Sequence[str],
    panel_settings: Mapping[str, Mapping[str, object]] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, PanelSettings]:
    """Layer the settings of each panel: per-variable `panel_settings`, then `overrides` for every panel.

    Args:
        variables (Sequence[str]): Internal names of the plotted variables.
        panel_settings (Mapping[str, Mapping[str, object]] | None): Settings per variable (layer 2),
            e.g. ``{"FEDU": {"pitch_angle": 90 * u.deg, "vmin": 1.0}}``. Entries for variables that are
            not plotted are ignored.
        overrides (Mapping[str, object] | None): Settings for every panel (layer 3); ``None`` values
            leave the lower layers in place.

    Returns:
        dict[str, PanelSettings]: The resolved settings of each variable.

    Raises:
        TypeError: If a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range, or a panel's resolved `vmin` is not below its `vmax`
            (e.g. an override of `vmin` above a per-variable `vmax`).
    """
    check_panel_settings(panel_settings or {})
    checked_overrides = _check_settings(overrides or {})

    resolved: dict[str, PanelSettings] = {}
    for name in variables:
        settings = {**_check_settings((panel_settings or {}).get(name, {})), **checked_overrides}
        try:
            _check_limits(settings)
        except ValueError as error:
            msg = f"{name}: {error}"
            raise ValueError(msg) from error
        resolved[name] = PanelSettings(**settings)
    return resolved


def resolve_colormap(name: str) -> Colormap:
    """Look up a colormap by name, first among the colorcet colormaps (e.g. ``rainbow4``), then in matplotlib.

    Raises:
        ValueError: If neither colorcet nor matplotlib knows the name.
    """
    import colorcet  # noqa: PLC0415
    import matplotlib as mpl  # noqa: PLC0415

    if name in colorcet.cm:
        return colorcet.cm[name]
    try:
        return mpl.colormaps[name]
    except KeyError:
        msg = f"unknown colormap '{name}'; use a colorcet name (e.g. rainbow4, fire) or a matplotlib colormap name."
        raise ValueError(msg) from None


def _label(dataset: DataSet, name: InternalName) -> str:
    """Axis label ``name [unit]``, naming the field model for model-dependent variables."""
    info = dataset.saving_strategy.data_standard.variable_infos[name]
    label = name
    mag_field = getattr(dataset.saving_strategy, "mag_field", None)
    if mag_field and name in _FIELD_MODEL_VARIABLES:
        label += f" ({mag_field})"
    unit = "deg" if info.unit == u.rad else info.unit.to_string()
    return f"{label} [{unit}]" if unit else label


def _resolve_dependency(
    dataset: DataSet, dependency: Dependency
) -> tuple[InternalName | FixedDimensionName, NDArray[np.float64]]:
    """Name and values of the coordinate variable of a dimension, with a time dimension if it has one.

    A dimension with alternatives (e.g. electron or proton energy) uses the first that has data; if
    none has, the last alternative and its empty values are returned. Angles are converted from radians
    to degrees.

    Args:
        dataset (DataSet): The dataset to load the coordinate from.
        dependency (Dependency): One entry of ``data_standard.get_dependencies(...)[1:]``: an internal
            name, a `OneOf` of alternative internal names, or a fixed dimension name (such as
            ``"Position_components"``, which has no coordinate variable and gives empty values).

    Returns:
        tuple[InternalName | FixedDimensionName, NDArray[np.float64]]: The name of the variable the
        values were taken from, and the values with shape ``(n_channels,)`` or ``(n_times, n_channels)``.
    """
    alternatives = alternatives_of(dependency)
    values = np.asarray([])
    name = alternatives[0]
    for name in alternatives:
        values = dataset.get_by_internal_name(name)  # ty:ignore[invalid-argument-type]
        if values.size > 0:
            break
    if dataset.saving_strategy.data_standard.variable_infos[name].unit == u.rad:  # ty:ignore[invalid-argument-type]
        values = np.degrees(values)  # angles are stored in radians, but shown in degrees
    return name, values


def _get_median_coordinate(
    dataset: DataSet, dependency: Dependency
) -> tuple[InternalName | FixedDimensionName, NDArray[np.float64]]:
    """Like `_resolve_dependency`, but time-dependent coordinates are reduced to their median over time.

    The result always has shape ``(n_channels,)``, as needed for the axis of a spectrogram.
    """
    name, values = _resolve_dependency(dataset, dependency)
    if values.ndim > 1:
        values = np.nanmedian(values, axis=0)
    return name, values


def match_nearest(
    coordinate: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    n_times: int,
    tolerance: float | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Match each target to the nearest channel of `coordinate`, separately at every time step.

    Args:
        coordinate (NDArray[np.float64]): Channel values, either fixed with shape ``(n_channels,)``
            or time-dependent with shape ``(n_times, n_channels)`` (e.g. MagEIS ``Energy_FEDU``,
            whose channels shift between files).
        targets (NDArray[np.float64]): Requested values, in the unit of `coordinate`.
        n_times (int): Number of time steps, used to broadcast a fixed `coordinate`.
        tolerance (float | None): Largest accepted distance to the target as a fraction of the target.
            Time steps whose nearest channel is further away are not matched. ``None`` accepts any distance.

    Returns:
        tuple[NDArray[np.intp], NDArray[np.float64]]: Channel indices and matched channel values,
        both of shape ``(n_times, n_targets)``. Not matched time steps (also those where the
        coordinate is NaN) have index 0 and value NaN, so the values can be used as a mask.
    """
    coordinate = np.asarray(coordinate, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    channels = np.broadcast_to(coordinate, (n_times, coordinate.shape[-1])) if coordinate.ndim == 1 else coordinate
    if channels.shape[0] != n_times:
        msg = f"coordinate has {channels.shape[0]} time steps, expected {n_times}."
        raise ValueError(msg)

    distance = np.abs(channels[:, :, None] - targets[None, None, :])  # (time, channel, target)
    distance = np.where(np.isnan(distance), np.inf, distance)
    indices = np.argmin(distance, axis=1)  # (time, target)
    values = np.take_along_axis(channels, indices, axis=1)
    nearest = np.take_along_axis(distance, indices[:, None, :], axis=1)[:, 0, :]
    matched = np.isfinite(nearest)
    if tolerance is not None:
        matched &= nearest <= tolerance * np.abs(targets)[None, :]
    return np.where(matched, indices, 0), np.where(matched, values, np.nan)


def _is_dimension(dependency: Dependency, prefix: str) -> bool:
    """Whether a dimension's coordinate variable (every alternative of it) starts with `prefix`, e.g. ``"Energy"``."""
    alternatives = alternatives_of(dependency)
    return all(name.startswith(prefix) for name in alternatives)


def _cut_pitch_angle(
    dataset: DataSet, data: NDArray[np.float64], dependency: Dependency, pitch_angle: u.Quantity
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cut the last (pitch-angle) dimension of `data` at the channel nearest `pitch_angle`, at each time step.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The cut data, with one dimension less, and the
        matched pitch angle in degrees per time step. Time steps without any pitch angle are NaN in both.
    """
    _, pitch_angles = _resolve_dependency(dataset, dependency)
    indices, matched = match_nearest(pitch_angles, pitch_angle.to_value(u.deg), n_times=data.shape[0])
    shape = (data.shape[0],) + (1,) * (data.ndim - 1)
    cut = np.take_along_axis(data, indices[:, 0].reshape(shape), axis=-1)[..., 0]
    missing = np.isnan(matched[:, 0]).reshape(shape[:-1])
    return np.where(missing, np.nan, cut), matched[:, 0]


def _pitch_angles(settings: PanelSettings) -> list[u.Quantity]:
    """The pitch angles of a panel's settings as a list, defaulting to `_DEFAULT_PITCH_ANGLE`."""
    if settings.pitch_angle is None:
        return [_DEFAULT_PITCH_ANGLE]
    return list(np.atleast_1d(settings.pitch_angle))


def pitch_angle_channels(dataset: DataSet, name: InternalName) -> NDArray[np.float64]:
    """The pitch-angle channels of a variable in degrees, as `plot` matches them: the median over time.

    A requested pitch angle is cut at the nearest of these channels (per time step, for channels that
    shift with time), so they are the values a pitch-angle selection can usefully take, e.g. the steps
    of a slider.

    Args:
        dataset (DataSet): The dataset that holds the variable.
        name (InternalName): Internal name of a variable, e.g. ``"FEDU"`` or ``"L_star"``.

    Returns:
        NDArray[np.float64]: The sorted finite channels, rounded to 1e-6 degrees, empty if the variable has
        no pitch-angle dimension.
    """
    dependencies = dataset.saving_strategy.data_standard.get_dependencies(name)[1:]
    if not dependencies or not _is_dimension(dependencies[-1], "Alpha"):
        return np.asarray([], dtype=float)
    _, channels = _get_median_coordinate(dataset, dependencies[-1])
    channels = np.asarray(channels, dtype=float).ravel()
    # angles are stored in radians, so e.g. 30° comes back as 29.999999999999996°
    return np.unique(np.round(channels[np.isfinite(channels)], 6))


def _pitch_angle_note(matched: NDArray[np.float64]) -> str:
    """Panel note naming the median matched pitch angle, e.g. ``pitch angle 90°``."""
    return f"pitch angle {np.nanmedian(matched):.0f}°"


def _select_energies(
    dataset: DataSet,
    name: InternalName,
    data: NDArray[np.float64],
    dependency: Dependency,
    energies: u.Quantity,
    *,
    warn: bool = True,
) -> tuple[NDArray[np.float64], list[str]]:
    """The values of `data` (shape ``(time, energy)``) at each of `energies`, using the nearest channel per time step.

    Time steps whose nearest channel is more than `_ENERGY_TOLERANCE` away from the requested energy
    are NaN, and an energy without any match is reported with a warning (unless `warn` is false, for
    repeated selections of the same channels).

    Returns:
        tuple[NDArray[np.float64], list[str]]: Values of shape ``(time, len(energies))`` and one label
        per energy with the median matched energy, e.g. ``470 keV``.
    """
    coordinate_name, raw_energy = _resolve_dependency(dataset, dependency)
    energy_unit = dataset.saving_strategy.data_standard.variable_infos[coordinate_name].unit  # ty:ignore[invalid-argument-type]
    targets = energies.to_value(energy_unit)
    indices, matched = match_nearest(raw_energy, targets, n_times=data.shape[0], tolerance=_ENERGY_TOLERANCE)
    values = np.where(np.isnan(matched), np.nan, np.take_along_axis(data, indices, axis=1))

    labels: list[str] = []
    for k, target in enumerate(targets):
        if np.all(np.isnan(matched[:, k])):
            if warn:
                logger.warning(f"{name}: no channel within {_ENERGY_TOLERANCE:.0%} of {target:g} {energy_unit}.")
            labels.append(f"{target:g} {energy_unit}: no match")
        else:
            labels.append(f"{np.nanmedian(matched[:, k]):.4g} {energy_unit}")
    return values, labels


def _plot_variable(
    dataset: DataSet,
    ax: Axes,
    name: InternalName,
    time: NDArray[Any],
    settings: PanelSettings,
    cmap: Colormap,
) -> None:
    """Draw one variable into `ax`, choosing the plot type from its dimensions.

    Of `settings`, the pitch angle applies to variables with an energy and a pitch-angle dimension,
    and the colour limits to spectrograms only.
    """
    standard = dataset.saving_strategy.data_standard
    data = dataset.get_by_internal_name(name)
    dependencies = standard.get_dependencies(name)[1:]
    mag_field = getattr(dataset.saving_strategy, "mag_field", None)
    title = f"{name} ({mag_field})" if mag_field and name in _FIELD_MODEL_VARIABLES else name

    if data.size == 0 or time.size == 0:
        ax.text(0.5, 0.5, f"{title}: no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return

    ax.xaxis_date()
    if not dependencies:
        ax.plot(time, data, ".-", markersize=2, linewidth=0.8)
        ax.set_ylabel(_label(dataset, name))
        return

    coordinates = [_get_median_coordinate(dataset, dependency) for dependency in dependencies]

    if data.ndim - 1 != len(dependencies):
        msg = f"{name} has {data.ndim - 1} non-time dimensions, but its data standard lists {len(dependencies)}."
        raise ValueError(msg)

    # (time, energy, pitch angle) -> (time, energy), cut at the pitch angle nearest the requested one at each time step
    selection_note = ""
    if data.ndim == 3 and data.shape[-1] == 1:
        # a single pitch-angle bin (e.g. omnidirectional DMSP FEDO, whose coordinate is a min/max range): nothing to cut
        data = data[..., 0]
        coordinates = coordinates[:1]
    elif data.ndim == 3:
        pitch_angle = _DEFAULT_PITCH_ANGLE if settings.pitch_angle is None else settings.pitch_angle
        data, matched = _cut_pitch_angle(dataset, data, dependencies[1], pitch_angle)
        selection_note = _pitch_angle_note(matched)
        coordinates = coordinates[:1]

    # (time, pitch angle) -> (time), if pitch angles are requested explicitly, e.g. L_star: one line each
    if data.ndim == 2 and settings.pitch_angle is not None and coordinates[0][0].startswith("Alpha"):
        pitch_angles = _pitch_angles(settings)
        for pitch_angle in pitch_angles:
            line, matched = _cut_pitch_angle(dataset, data, dependencies[0], pitch_angle)
            ax.plot(time, line, ".-", markersize=2, linewidth=0.8, label=_pitch_angle_note(matched))
        ax.set_ylabel(_label(dataset, name))
        if len(pitch_angles) == 1:
            _add_note(ax, _pitch_angle_note(matched))
        else:
            ax.legend(loc="upper right", fontsize="x-small", ncols=min(len(pitch_angles), 4))
        return

    first_dependency = dependencies[0]
    is_component = isinstance(first_dependency, str) and first_dependency not in standard.variable_infos

    if is_component:
        for component in range(data.shape[1]):
            ax.plot(time, data[:, component], linewidth=0.8, label=f"{component}")
        ax.legend(loc="upper right", fontsize="x-small", ncols=data.shape[1])
        ax.set_ylabel(_label(dataset, name))
        return

    coordinate_name, coordinate = coordinates[0]
    if settings.energies is not None and coordinate_name.startswith("Energy"):
        # an energy coordinate is a variable, never a fixed dimension name
        _plot_energy_lines(dataset, ax, name, time, data, coordinate_name, settings, selection_note)  # ty:ignore[invalid-argument-type]
        return
    if coordinate.size != data.shape[1]:
        msg = f"{name} has {data.shape[1]} {coordinate_name} channels, but the coordinate has {coordinate.size} values."
        raise ValueError(msg)

    _plot_spectrogram(dataset, ax, name, time, data, coordinate_name, coordinate, settings, selection_note, cmap)


def _plot_spectrogram(
    dataset: DataSet,
    ax: Axes,
    name: InternalName,
    time: NDArray[Any],
    data: NDArray[np.float64],
    coordinate_name: InternalName | FixedDimensionName,
    coordinate: NDArray[np.float64],
    settings: PanelSettings,
    selection_note: str,
    cmap: Colormap,
) -> None:
    """Draw `data` of shape ``(time, channel)`` as a spectrogram with `coordinate` on the y axis and a colour bar.

    Flux-like variables get a log colour scale, and energy coordinates a log y axis. `settings.vmin`
    and `settings.vmax` set the colour limits. `selection_note` is written into the panel.
    """
    import matplotlib.dates as mdates  # noqa: PLC0415

    use_log = name in _LOG_COLOR_VARIABLES
    if use_log:
        data = np.where(data > 0, data, np.nan)

    mesh = ax.pcolormesh(
        mdates.date2num(time),
        coordinate,
        data.T,
        shading="nearest",
        cmap=cmap,
        norm=_color_norm(data, log=use_log, vmin=settings.vmin, vmax=settings.vmax),
        rasterized=True,
    )
    if coordinate_name.startswith("Energy"):
        ax.set_yscale("log")
    ax.set_ylabel(_label(dataset, coordinate_name))  # ty:ignore[invalid-argument-type]
    # name and unit on separate lines, so that the label fits the height of one panel
    ax.figure.colorbar(mesh, ax=ax, pad=0.01, extend="both", label=_label(dataset, name).replace(" [", "\n[", 1))
    _add_note(ax, selection_note)


def _add_note(ax: Axes, note: str) -> None:
    """Write `note` (e.g. the matched pitch angle) into the upper left corner of the panel."""
    if note:
        ax.text(
            0.005,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )


def _plot_energy_lines(
    dataset: DataSet,
    ax: Axes,
    name: InternalName,
    time: NDArray[Any],
    data: NDArray[np.float64],
    coordinate_name: InternalName,
    settings: PanelSettings,
    selection_note: str,
) -> None:
    """Draw one line per selected energy (log y), matched to the nearest channel at each time step.

    `data` has shape ``(time, energy)``. Time steps whose nearest channel is more than
    `_ENERGY_TOLERANCE` away from the requested energy are left blank. The legend gives the
    median matched energy. `settings.vmin` and `settings.vmax` set the y-limits.
    """
    assert settings.energies is not None
    lines, labels = _select_energies(dataset, name, data, coordinate_name, settings.energies)
    lines = np.where(lines > 0, lines, np.nan)  # log axis

    for k, label in enumerate(labels):
        ax.plot(time, lines[:, k], linewidth=0.8, label=label)

    ax.set_yscale("log")
    if settings.vmin is not None or settings.vmax is not None:
        ax.set_ylim(settings.vmin, settings.vmax)
    ax.set_ylabel(_label(dataset, name))
    ax.legend(loc="upper right", fontsize="x-small", ncols=min(len(labels), 4))
    _add_note(ax, selection_note)


def _has_pitch_angle_dimension(dataset: DataSet, name: InternalName) -> bool:
    """Whether a variable's last dimension is pitch angle (e.g. FEDU, L_star), in its data standard."""
    standard = dataset.saving_strategy.data_standard
    if name not in standard.variable_infos:
        return False
    dependencies = standard.get_dependencies(name)[1:]
    return bool(dependencies) and _is_dimension(dependencies[-1], "Alpha")


def _is_spectral(dataset: DataSet, name: InternalName) -> bool:
    """Whether a variable has an energy and a pitch-angle dimension (e.g. FEDU): a spectrogram cuts it at one angle."""
    standard = dataset.saving_strategy.data_standard
    if name not in standard.variable_infos:
        return False
    dependencies = standard.get_dependencies(name)[1:]
    return len(dependencies) == 2 and _has_pitch_angle_dimension(dataset, name)


def _panel_shape(dataset: DataSet, name: InternalName, settings: PanelSettings) -> tuple[int, int]:
    """Rows and columns of the panels of one variable.

    Without ``along`` a variable has one panel. With it, a variable with an energy dimension has one row
    per selected energy, or per channel without a selection; and one column per selected pitch angle if
    the variable or the ``along`` coordinate depends on pitch angle.
    """
    if settings.along is None:
        return 1, 1
    columns = 1
    if _has_pitch_angle_dimension(dataset, name) or _has_pitch_angle_dimension(dataset, settings.along):
        columns = len(_pitch_angles(settings))

    standard = dataset.saving_strategy.data_standard
    if name not in standard.variable_infos:
        return 1, columns
    dependencies = standard.get_dependencies(name)[1:]
    if not dependencies or not _is_dimension(dependencies[0], "Energy"):
        return 1, columns
    if settings.energies is not None:
        return settings.energies.size, columns
    data = dataset.get_by_internal_name(name)
    return (data.shape[1] if data.ndim >= 2 and data.shape[1] > 0 else 1), columns


def panel_shapes(
    dataset: DataSet,
    variables: Sequence[InternalName],
    *,
    along: InternalName | None = None,
    panel_settings: Mapping[str, Mapping[str, object]] | None = None,
    energies: u.Quantity | None = None,
    pitch_angle: u.Quantity | None = None,
) -> list[tuple[int, int]]:
    """Rows and columns of the panels `plot` draws for each variable, e.g. to lay out ``plot(axes=...)``.

    A variable has one panel, unless it is drawn along a coordinate (`along`, or the ``along`` setting in
    its `panel_settings`): then it is a grid with one row per selected energy (from `energies`, else from
    its `panel_settings`), or per energy channel if none are selected, and one column per selected pitch
    angle (from `pitch_angle`, else from its `panel_settings`; one column at 90° by default). Variables
    without an energy dimension have one row, and those for which neither the variable nor the coordinate
    depends on pitch angle (e.g. ``B_Eq`` along ``R_Eq``) one column.

    Args:
        dataset (DataSet): The dataset to plot.
        variables (Sequence[InternalName]): Internal names of the variables to plot.
        along (InternalName | None): The `along` argument of `plot`; overrides the ``along`` setting of every variable.
        panel_settings (Mapping[str, Mapping[str, object]] | None): The `panel_settings` argument of `plot`.
        energies (u.Quantity | None): The `energies` argument of `plot`.
        pitch_angle (u.Quantity | None): The `pitch_angle` argument of `plot`.

    Returns:
        list[tuple[int, int]]: The rows and columns of each variable, in the order of `variables`. `plot`
        fills the panels row by row.

    Raises:
        TypeError: If a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range.
    """
    settings = resolve_panel_settings(
        variables, panel_settings, {"energies": energies, "pitch_angle": pitch_angle, "along": along}
    )
    return [_panel_shape(dataset, name, settings[name]) for name in variables]


def count_panels(
    dataset: DataSet,
    variables: Sequence[InternalName],
    *,
    along: InternalName | None = None,
    panel_settings: Mapping[str, Mapping[str, object]] | None = None,
    energies: u.Quantity | None = None,
    pitch_angle: u.Quantity | None = None,
) -> list[int]:
    """Number of panels `plot` draws for each variable: the rows times the columns of `panel_shapes`.

    Args:
        dataset (DataSet): The dataset to plot.
        variables (Sequence[InternalName]): Internal names of the variables to plot.
        along (InternalName | None): The `along` argument of `plot`; overrides the ``along`` setting of every variable.
        panel_settings (Mapping[str, Mapping[str, object]] | None): The `panel_settings` argument of `plot`.
        energies (u.Quantity | None): The `energies` argument of `plot`.
        pitch_angle (u.Quantity | None): The `pitch_angle` argument of `plot`.

    Returns:
        list[int]: The number of panels of each variable, in the order of `variables`.

    Raises:
        TypeError: If a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range.
    """
    shapes = panel_shapes(
        dataset, variables, along=along, panel_settings=panel_settings, energies=energies, pitch_angle=pitch_angle
    )
    return [rows * columns for rows, columns in shapes]


def figure_size(shapes: Sequence[tuple[int, int]], panel_height: float = _PANEL_HEIGHT_INCHES) -> tuple[float, float]:
    """Size in inches of a figure laid out by `create_panel_axes`: 10 inches wide, wider for grids.

    Args:
        shapes (Sequence[tuple[int, int]]): Rows and columns of each block of panels, top to bottom.
        panel_height (float): Height of one row of panels in inches.

    Returns:
        tuple[float, float]: Width and height.
    """
    rows = sum(r for r, _ in shapes)
    columns = max((c for _, c in shapes), default=1)
    return min(10.0 + 4.0 * (columns - 1), 30.0), panel_height * rows + 0.8


def create_panel_axes(figure: Figure, shapes: Sequence[tuple[int, int]]) -> list[list[Axes]]:
    """Lay out blocks of panels in one column with a shared time axis; a block can be a grid.

    Every block spans the full width of the figure and is split evenly into its columns (the grid
    has the least common multiple of all column counts as its columns). All panels share the x axis
    with the first one, and only the bottom row shows tick labels.

    Args:
        figure (Figure): The empty figure to add the panels to.
        shapes (Sequence[tuple[int, int]]): Rows and columns of each block, top to bottom, e.g. from
            `panel_shapes`; a single panel is ``(1, 1)``.

    Returns:
        list[list[Axes]]: The panels of each block, row by row, as `plot` takes them for one variable.
    """
    import math  # noqa: PLC0415

    total_rows = sum(r for r, _ in shapes)
    total_columns = math.lcm(*(c for _, c in shapes)) if shapes else 1
    grid = figure.add_gridspec(max(total_rows, 1), total_columns)
    first: Axes | None = None
    blocks: list[list[Axes]] = []
    row = 0
    for rows, columns in shapes:
        span = total_columns // columns
        block: list[Axes] = []
        for r in range(rows):
            for c in range(columns):
                ax = figure.add_subplot(grid[row + r, c * span : (c + 1) * span], sharex=first)
                first = first or ax
                if row + r < total_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                block.append(ax)
        row += rows
        blocks.append(block)
    return blocks


def _along_coordinate(
    dataset: DataSet, along: InternalName, pitch_angle: u.Quantity, n_times: int
) -> tuple[NDArray[np.float64], str]:
    """Values of the y coordinate of scatter panels, cut at `pitch_angle` if it depends on pitch angle (L*, Lm).

    Returns:
        tuple[NDArray[np.float64], str]: One value per time step, and a note naming the matched pitch
        angle (empty for coordinates without a pitch-angle dimension, such as ``R_Eq``).

    Raises:
        ValueError: If the coordinate does not have one value per time step.
    """
    coordinate = np.asarray(dataset.get_by_internal_name(along), dtype=float)
    dependencies = dataset.saving_strategy.data_standard.get_dependencies(along)[1:]
    note = ""
    if dependencies and _is_dimension(dependencies[-1], "Alpha") and coordinate.ndim == len(dependencies) + 1:
        coordinate, matched = _cut_pitch_angle(dataset, coordinate, dependencies[-1], pitch_angle)
        note = f"{along} at {_pitch_angle_note(matched)}"
    if coordinate.shape != (n_times,):
        msg = f"{along} must have one value per time step ({n_times}), got shape {coordinate.shape}."
        raise ValueError(msg)
    return coordinate, note


def _along_rows(
    dataset: DataSet,
    name: InternalName,
    data: NDArray[np.float64],
    dependencies: Sequence[Dependency],
    settings: PanelSettings,
    along: InternalName,
    *,
    warn: bool,
) -> list[tuple[NDArray[np.float64], str]]:
    """Split `data` (already cut in pitch angle) into the rows of an along grid: one per energy, or a single row.

    Raises:
        ValueError: If the variable has a dimension other than energy (e.g. a vector component).
    """
    if not dependencies:
        return [(data, "")]
    if len(dependencies) == 1 and _is_dimension(dependencies[0], "Energy"):
        if settings.energies is not None:
            values, labels = _select_energies(dataset, name, data, dependencies[0], settings.energies, warn=warn)
        else:
            coordinate_name, channel_energies = _get_median_coordinate(dataset, dependencies[0])
            if channel_energies.size != data.shape[1]:
                msg = (
                    f"{name} has {data.shape[1]} {coordinate_name} channels, "
                    f"but the coordinate has {channel_energies.size} values."
                )
                raise ValueError(msg)
            unit = dataset.saving_strategy.data_standard.variable_infos[coordinate_name].unit  # ty:ignore[invalid-argument-type]
            values, labels = data, [f"{energy:.4g} {unit}" for energy in channel_energies]
        return [(values[:, k], label) for k, label in enumerate(labels)]
    msg = (
        f"{name} depends on {', '.join(map(str, dependencies))}; only variables of time, energy and "
        f"pitch angle can be drawn along {along}."
    )
    raise ValueError(msg)


def _plot_along(
    dataset: DataSet,
    axes: Sequence[Axes],
    name: InternalName,
    time: NDArray[Any],
    settings: PanelSettings,
    along: InternalName,
    cmap: Colormap,
) -> None:
    """Draw one variable as a grid of scatter panels over time, with `along` on the y axis and the value as colour.

    The grid has one column per pitch angle of `settings.pitch_angle` (default 90°), at which both the
    variable and the coordinate are cut if they depend on pitch angle, and one row per selected energy,
    or per channel without a selection. `axes` holds the panels row by row (see `panel_shapes`). The
    panels of a row share their colour limits and one colour bar.

    Raises:
        ValueError: If the variable has a dimension other than energy and pitch angle (e.g. a vector),
            or `axes` does not match the number of panels.
    """
    standard = dataset.saving_strategy.data_standard
    data = np.asarray(dataset.get_by_internal_name(name), dtype=float)
    dependencies = list(standard.get_dependencies(name)[1:])
    if data.ndim - 1 != len(dependencies):
        msg = f"{name} has {data.ndim - 1} non-time dimensions, but its data standard lists {len(dependencies)}."
        raise ValueError(msg)

    has_pitch_angle = bool(dependencies) and _is_dimension(dependencies[-1], "Alpha")
    remaining = dependencies[:-1] if has_pitch_angle else dependencies
    rows, n_columns = _panel_shape(dataset, name, settings)
    pitch_angles = _pitch_angles(settings)[:n_columns]

    # columns[c][r] = (coordinate, values, note) of the panel in row r and column c
    columns: list[list[tuple[NDArray[np.float64], NDArray[np.float64], str]]] = []
    for column, pitch_angle in enumerate(pitch_angles):
        coordinate, note = _along_coordinate(dataset, along, pitch_angle, data.shape[0])
        cut = data
        if has_pitch_angle:
            # the variable and the coordinate share the pitch-angle channels, so one note covers both cuts
            cut, matched = _cut_pitch_angle(dataset, data, dependencies[-1], pitch_angle)
            note = _pitch_angle_note(matched)
        cells = _along_rows(dataset, name, cut, remaining, settings, along, warn=column == 0)
        columns.append([(coordinate, values, ", ".join(filter(None, [label, note]))) for values, label in cells])

    if len(axes) != rows * len(columns) or any(len(cells) != rows for cells in columns):
        msg = f"Got {len(axes)} axes for the {rows * len(columns)} panels of {name}."
        raise ValueError(msg)

    use_log = name in _LOG_COLOR_VARIABLES
    colorbar_label = _label(dataset, name).replace(" [", "\n[", 1)
    for row in range(rows):
        row_axes = [axes[row * len(columns) + c] for c in range(len(columns))]
        cells = [columns[c][row] for c in range(len(columns))]
        if use_log:
            cells = [(coordinate, np.where(values > 0, values, np.nan), note) for coordinate, values, note in cells]
        shown = [values[np.isfinite(values) & np.isfinite(coordinate)] for coordinate, values, _ in cells]
        norm = _color_norm(np.concatenate(shown), log=use_log, vmin=settings.vmin, vmax=settings.vmax)

        mappable = None
        for ax, (coordinate, values, note) in zip(row_axes, cells, strict=True):
            mappable = _plot_scatter(dataset, ax, name, time, coordinate, values, along, note, cmap, norm) or mappable
        if mappable is not None:
            target = row_axes[0] if len(row_axes) == 1 else row_axes
            row_axes[0].figure.colorbar(mappable, ax=target, pad=0.01, extend="both", label=colorbar_label)


def _plot_scatter(
    dataset: DataSet,
    ax: Axes,
    name: InternalName,
    time: NDArray[Any],
    coordinate: NDArray[np.float64],
    values: NDArray[np.float64],
    along: InternalName,
    note: str,
    cmap: Colormap,
    norm: Normalize | None,
) -> PathCollection | None:
    """Draw `values` as coloured points at (time, `coordinate`) with colour scale `norm`, as in a spectrogram.

    Points with a NaN value or coordinate are left out; a panel without any says so.

    Returns:
        PathCollection | None: The points, for a colour bar, or ``None`` if there are none.
    """
    import matplotlib.dates as mdates  # noqa: PLC0415

    shown = np.isfinite(values) & np.isfinite(coordinate)
    ax.xaxis_date()
    ax.set_ylabel(_label(dataset, along))
    _add_note(ax, note)
    if not shown.any():
        ax.text(0.5, 0.5, f"{name}: no data", ha="center", va="center", transform=ax.transAxes)
        return None

    return ax.scatter(
        mdates.date2num(time[shown]),
        coordinate[shown],
        c=values[shown],
        s=6,
        linewidths=0,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )


def _color_norm(
    data: NDArray[np.float64], *, log: bool, vmin: float | None = None, vmax: float | None = None
) -> Normalize | None:
    """Colour limits: `vmin` and `vmax` where given, otherwise the 5th and 95th percentiles of the finite data.

    The percentiles keep outliers from dominating. Returns ``None`` (matplotlib's own autoscaling)
    if no limit is given and the data has no spread.

    Raises:
        ValueError: If a log scale gets a limit that is not positive, or if the resulting lower limit
            is not below the upper one (e.g. only `vmin` given, above the data's 95th percentile).
    """
    from matplotlib.colors import LogNorm, Normalize  # noqa: PLC0415

    if log and any(limit is not None and limit <= 0 for limit in (vmin, vmax)):
        msg = f"colour limits of a log scale must be positive, got vmin={vmin}, vmax={vmax}."
        raise ValueError(msg)

    finite = data[np.isfinite(data)]
    if finite.size == 0 and (vmin is None or vmax is None):
        return None
    if vmin is None or vmax is None:
        low, high = np.percentile(finite, _COLOR_LIMIT_PERCENTILES)
        if vmin is None and vmax is None and not low < high:
            return None
        vmin = float(low) if vmin is None else vmin
        vmax = float(high) if vmax is None else vmax
    if not vmin < vmax:
        msg = f"the lower colour limit ({vmin:g}) must be below the upper one ({vmax:g})."
        raise ValueError(msg)
    return LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)


def _as_quantity(value: object, unit: u.UnitBase, argument: str) -> u.Quantity:
    """`value` as an astropy quantity convertible to `unit`.

    Raises:
        TypeError: If `value` is not a quantity.
        ValueError: If its unit does not convert to `unit`.
    """
    if not isinstance(value, u.Quantity):
        msg = f"{argument} needs a unit, e.g. {argument}=90 * u.{unit}; got {value!r}."
        raise TypeError(msg)
    if not value.unit.is_equivalent(unit):
        msg = f"{argument} must be convertible to {unit}, got {value.unit}."
        raise ValueError(msg)
    return value


def plot(
    self: DataSet,
    variables: Sequence[InternalName],
    *,
    axes: Sequence[Axes] | None = None,
    panel_settings: Mapping[str, Mapping[str, object]] | None = None,
    energies: u.Quantity | None = None,
    pitch_angle: u.Quantity | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    along: InternalName | None = None,
    colormap: str = DEFAULT_COLORMAP,
) -> Figure:
    """Draw a quick-look plot of the given variables, one panel per variable.

    The plot type follows from the variable's dimensions in the data standard: a line for time
    series, one line per component for vectors, and a spectrogram (energy on a log y axis, flux on
    a log colour scale) for variables with an energy or pitch-angle dimension. Colour limits are
    the 5th and 95th percentiles of the data unless `vmin`/`vmax` are given. For variables with
    both an energy and a pitch-angle dimension, the pitch angle nearest `pitch_angle` (default 90°)
    is shown and noted in the panel. Labels name the strategy's ``mag_field`` for variables that
    depend on a magnetic field model. A requested variable that has no data in the dataset is an
    error, and a variable that cannot be drawn is reported with a warning and leaves its panel empty.

    Settings come in layers: the defaults of each plot type, then `panel_settings` per variable
    (e.g. a recipe's ``PLOT_DEFAULTS``), then the keyword arguments `energies`, `pitch_angle`, `vmin`,
    `vmax` and `along`, which override the settings of every panel. Each setting applies to the panels it
    fits and is ignored by the others, so that one set of arguments can serve a mixed list such as
    ``["FEDU", "L_star"]``.

    With `along` (or the ``along`` setting of a variable, e.g. in a recipe's ``PLOT_DEFAULTS``), each panel
    is a scatter plot over time with that coordinate (``L_star``, ``L_m`` or ``R_Eq``) on the y axis and
    the variable's value as colour, with a colour scale and limits as in spectrograms. A variable is then
    a grid of panels (see `panel_shapes`): one row per selected energy, or per channel if no energies are
    selected, and one column per pitch angle of `pitch_angle` (default 90°), at which the pitch-angle
    dimensions of the variable and of the coordinate are cut. The panels of a row share their colour
    limits and colour bar. Variables with other dimensions, such as vectors, cannot be drawn this way.

    Args:
        self (DataSet): The dataset to plot.
        variables (Sequence[InternalName]): Internal names of the variables to plot, in panel order.
        axes (Sequence[Axes] | None): Axes to draw into, one per panel (e.g. one satellite's block of
            a multi-satellite figure): one per variable, or as many as `count_panels` gives with
            `along`, each variable's grid row by row (`create_panel_axes` lays them out). If ``None``,
            a new figure is created.
        panel_settings (Mapping[str, Mapping[str, object]] | None): Settings per variable, with keys
            from :data:`PANEL_SETTING_NAMES`, e.g. ``{"FEDU": {"pitch_angle": 90 * u.deg, "vmax": 1e6}}``.
            Entries for variables that are not plotted are ignored.
        energies (u.Quantity | None): Energies to show for variables with an energy dimension.
            Such a variable is drawn as one line per energy (log y) instead of a spectrogram, using
            the nearest channel at each time step; time steps more than 10% away are left blank.
        pitch_angle (u.Quantity | None): Pitch angle at which variables with an energy and a
            pitch-angle dimension are cut, at the nearest channel (see `pitch_angle_channels`),
            defaulting to 90°. If given, variables with only a pitch-angle dimension (e.g. ``L_star``)
            are cut as well and drawn as a line, one per angle if several are given (a 1-d quantity).
            With `along`, each of several angles gets one column of panels.
        vmin (float | None): Lower colour limit of spectrograms and scatter panels, in the unit of
            the plotted variable. Defaults to the 5th percentile of the data.
        vmax (float | None): Upper colour limit of spectrograms and scatter panels, in the unit of
            the plotted variable. Defaults to the 95th percentile of the data.
        along (InternalName | None): One of :data:`ALONG_COORDINATES` to draw every variable as
            scatter panels with this coordinate on the y axis, or ``None`` to leave it to the ``along``
            settings of the variables (by default the plot types above).
        colormap (str): Colormap of spectrograms and scatter panels: a colorcet name or else a matplotlib
            colormap name. Defaults to ``viridis``.

    Returns:
        Figure: The figure that contains the panels.

    Raises:
        ValueError: If no variables are given, if the number of axes does not match the number of
            panels, if a variable (or the `along` coordinate) is not part of the data standard or has no
            data in the dataset, if a setting is out
            of range or has a unit of the wrong kind, if `colormap` is unknown, if a panel's resolved
            `vmin` is not below its `vmax`, or if a variable with an energy and a pitch-angle dimension
            that is not drawn along a coordinate gets several pitch angles.
        TypeError: If a setting is unknown or has the wrong type, e.g. an energy without a unit.
    """
    from matplotlib.figure import Figure  # noqa: PLC0415

    if len(variables) == 0:
        msg = "At least one variable is required to plot."
        raise ValueError(msg)
    cmap = resolve_colormap(colormap)

    settings = resolve_panel_settings(
        variables,
        panel_settings,
        {"energies": energies, "pitch_angle": pitch_angle, "vmin": vmin, "vmax": vmax, "along": along},
    )
    coordinates = sorted({s.along for s in settings.values() if s.along is not None})

    standard = self.saving_strategy.data_standard
    for name in [*variables, *coordinates]:
        if name not in standard.variable_infos:
            msg = f"'{name}' is not a variable of {type(standard).__name__}."
            raise ValueError(msg)
        if self.get_by_internal_name(name).size == 0:
            msg = (
                f"'{name}' has no data in the {self.get_print_name()} files; was it saved by the processing run "
                "(e.g. an optional variable that needs an extra processing flag)?"
            )
            raise ValueError(msg)

    for name in variables:
        if settings[name].along is None and len(_pitch_angles(settings[name])) > 1 and _is_spectral(self, name):
            msg = (
                f"{name}: several pitch angles ({settings[name].pitch_angle}) need along (e.g. along='L_star'), "
                "which draws one column of panels per pitch angle; otherwise give one pitch angle."
            )
            raise ValueError(msg)

    shapes = [_panel_shape(self, n, settings[n]) for n in variables]
    counts = [rows * columns for rows, columns in shapes]
    num_panels = sum(counts)
    if axes is None:
        figure = Figure(figsize=figure_size(shapes), layout="constrained")
        axes = [ax for block in create_panel_axes(figure, shapes) for ax in block]
    else:
        axes = list(axes)
        if len(axes) != num_panels:
            msg = f"Got {len(axes)} axes for {num_panels} panels of {len(variables)} variables."
            raise ValueError(msg)
        figure = axes[0].get_figure(root=True)
        assert figure is not None

    time = np.asarray(self.datetime)
    first = 0
    for name, count in zip(variables, counts, strict=True):
        panel_axes = axes[first : first + count]
        first += count
        try:
            if (coordinate := settings[name].along) is None:
                _plot_variable(self, panel_axes[0], name, time, settings[name], cmap)
            else:
                _plot_along(self, panel_axes, name, time, settings[name], coordinate, cmap)
        except ValueError as error:
            logger.warning(f"Could not plot {name}: {error}")
            for ax in panel_axes:
                ax.text(0.5, 0.5, f"{name}: could not be plotted", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_title(self.get_print_name(), fontsize="medium")
    return figure
