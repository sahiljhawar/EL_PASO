# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Draw the figure of ``el-paso plot``: one column of panels with a shared time axis."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import el_paso as ep

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astropy import units as u
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName

logger = logging.getLogger("el_paso.cli")

# Kept in sync with `el_paso.dataset.plot.DEFAULT_COLORMAP`; that module imports numpy and astropy, which
# the CLI must not load just to render `--help`.
DEFAULT_COLORMAP = "viridis"


_PANEL_HEIGHT_INCHES = 2.0
"""Height of one panel in the multi-satellite figure."""

INDEX_PANELS = ("Kp", "Dst", "Vsw", "Nsw", "Pdyn", "BzIMF", "ByIMF", "G1", "G2", "G3", "W_params")
"""Space-weather indices and solar wind parameters that can be requested like variables.

These are the values of ``el_paso.load_indices_solar_wind_parameters.SW_Index`` (not imported here,
since that module imports swvo, pandas and scipy). They are not part of any dataset: they are loaded,
and downloaded if missing, for the plotted time range and drawn once at the top of the figure."""

_STEP_INDICES = frozenset({"Kp", "Dst"})
"""Indices that are averages over fixed intervals (3 h, 1 h) and are drawn as steps."""


def _new_figure(size: tuple[float, float], *, show: bool) -> Figure:
    """Create a figure of `size` inches.

    Only a figure that is shown is created through pyplot (and so with its backend); otherwise
    the figure is not tied to any GUI backend, and saving renders it with the backend that
    matches the file format (Agg for PNG), which also works without a display.
    """
    if show:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt.figure(figsize=size, layout="constrained")

    from matplotlib.figure import Figure  # noqa: PLC0415

    return Figure(figsize=size, layout="constrained")


def _plot_index(ax: Axes, name: str, start_time: datetime, end_time: datetime) -> None:
    """Draw a space-weather index or solar wind parameter into `ax`, downloading it first if it is not cached.

    Kp and Dst are drawn as steps, vector parameters (``W_params``) as one line per component. A
    failure to load the data is reported with a warning and leaves the panel empty.
    """
    import numpy as np  # noqa: PLC0415

    try:
        data, time = ep.load_indices_solar_wind_parameters(start_time, end_time, [name])[name]  # ty:ignore[invalid-argument-type]
        times = np.asarray([datetime.fromtimestamp(t, tz=timezone.utc) for t in time.get_data(ep.units.posixtime)])
        values = np.asarray(data.get_data(), dtype=float)
        unit = data.metadata.unit
    except Exception as error:  # noqa: BLE001 - swvo raises whatever its download or parsing raises
        logger.warning(f"Could not load {name}: {error}")
        times, values, unit = np.empty(0), np.empty(0), None

    ax.set_ylabel(f"{name} [{unit}]" if unit is not None and str(unit) else name)
    if times.size == 0 or not np.any(np.isfinite(values)):
        ax.text(0.5, 0.5, f"{name}: no data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.xaxis_date()
    if name in _STEP_INDICES:
        ax.step(times, values, where="post", linewidth=0.8)
    elif values.ndim == 2:
        n_components = values.shape[-1]
        for component in range(n_components):
            ax.plot(times, values[:, component], linewidth=0.8, label=f"W{component + 1}")
        ax.legend(loc="upper right", fontsize="x-small", ncols=n_components)
    else:
        ax.plot(times, values, linewidth=0.8)


def build_figure(
    datasets: Sequence[DataSet],
    variables: Sequence[InternalName],
    start_time: datetime,
    end_time: datetime,
    *,
    indices: Sequence[str] = (),
    show: bool = False,
    panel_settings: Mapping[str, Mapping[str, Any]] | None = None,
    energies: u.Quantity | None = None,
    pitch_angle: u.Quantity | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    along: InternalName | None = None,
    colormap: str = DEFAULT_COLORMAP,
) -> Figure:
    """Draw the variables of several datasets into one column of panels with a shared time axis.

    Panels are stacked in one block of rows per dataset (satellite), one panel per variable, in
    the given order; nothing is overlaid. With `along`, a variable is a grid instead, with one row per
    energy and one column per pitch angle (see :func:`el_paso.dataset.plot.panel_shapes`); the figure
    gets wider with the number of columns. Each block is drawn by `DataSet.plot` and titled with its
    dataset's name. Only the bottom row shows time tick labels, covering ``[start_time, end_time]``.

    Args:
        datasets (Sequence[DataSet]): The datasets to plot, one block of panels each.
        variables (Sequence[InternalName]): Internal names of the variables to plot per dataset.
        start_time (datetime): Start of the time axis.
        end_time (datetime): End of the time axis.
        indices (Sequence[str]): Space-weather indices and solar wind parameters (from `INDEX_PANELS`)
            to draw as panels at the top of the figure, once for all datasets. Missing data is downloaded.
        show (bool): Create the figure through pyplot, so that it can be shown in a window.
        panel_settings (Mapping[str, Mapping[str, Any]] | None): Settings per variable, e.g. from the
            recipe's ``PLOT_DEFAULTS``, passed to `DataSet.plot`; the arguments below override them.
        energies (u.Quantity | None): Energies to show, passed to `DataSet.plot`.
        pitch_angle (u.Quantity | None): Pitch-angle cut, or several (one column each) with `along`,
            passed to `DataSet.plot`.
        vmin (float | None): Lower colour limit of spectrograms, passed to `DataSet.plot`.
        vmax (float | None): Upper colour limit of spectrograms, passed to `DataSet.plot`.
        along (InternalName | None): Coordinate on the y axis of scatter panels, passed to `DataSet.plot`.
        colormap (str): Colormap of spectrograms and scatter panels, passed to `DataSet.plot`.

    Returns:
        Figure: The figure.
    """
    import matplotlib.dates as mdates  # noqa: PLC0415

    from el_paso.dataset.plot import create_panel_axes, figure_size, panel_shapes  # noqa: PLC0415

    dataset_shapes = [
        panel_shapes(
            dataset,
            variables,
            along=along,
            panel_settings=panel_settings,
            energies=energies,
            pitch_angle=pitch_angle,
        )
        for dataset in datasets
    ]
    shapes = [(1, 1)] * len(indices) + [shape for block in dataset_shapes for shape in block]
    figure = _new_figure(figure_size(shapes, _PANEL_HEIGHT_INCHES), show=show)
    panel_blocks = create_panel_axes(figure, shapes)

    for (ax,), name in zip(panel_blocks, indices, strict=False):
        _plot_index(ax, name, start_time, end_time)
    if indices:
        panel_blocks[0][0].set_title("Space weather", fontsize="medium")

    first = len(indices)
    for dataset, block_shapes in zip(datasets, dataset_shapes, strict=True):
        block = [ax for axes in panel_blocks[first : first + len(block_shapes)] for ax in axes]
        first += len(block_shapes)
        dataset.plot(
            variables,
            axes=block,
            panel_settings=panel_settings,
            energies=energies,
            pitch_angle=pitch_angle,
            vmin=vmin,
            vmax=vmax,
            along=along,
            colormap=colormap,
        )

    # all panels share the time axis (and so its locator and formatter); the bottom row is labelled
    bottom_rows, bottom_columns = shapes[-1]
    bottom_row = panel_blocks[-1][(bottom_rows - 1) * bottom_columns :]
    bottom = bottom_row[0]
    bottom.xaxis_date()
    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    bottom.set_xlim(float(mdates.date2num(start_time)), float(mdates.date2num(end_time)))
    for ax in bottom_row:
        ax.set_xlabel("Time (UTC)")
    return figure
