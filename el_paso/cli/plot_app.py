# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""The marimo app of ``el-paso plot --serve``; see `el_paso.cli.plot_serve`.

Run by ``marimo run plot_app.py -- --request <file>``, with a request written by
`el_paso.cli.plot_serve.encode_request`. It draws the figure of ``el-paso plot`` with
`build_figure` and shows it with ``marimo.mpl.interactive``.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="el-paso plot")


@app.cell
def _():
    import dataclasses
    import functools
    import logging
    import time as clock
    from datetime import datetime, time, timezone
    from pathlib import Path

    import marimo as mo
    import numpy as np

    import el_paso as ep
    from el_paso.cli import plot_serve
    from el_paso.cli.plot_cli import (
        _load_datasets,
        _recipe_settings,
        _setting_overrides,
        resolve_panels,
        split_panels,
    )
    from el_paso.cli.plot_figure import INDEX_PANELS, build_figure
    from el_paso.dataset.plot import ALONG_COORDINATES, pitch_angle_channels

    _args = mo.cli_args()
    if not _args.get("request"):
        _msg = "Start the app with `el-paso plot <mission> <recipe> ... --serve`."
        raise SystemExit(_msg)
    request = plot_serve.decode_request(Path(str(_args["request"])).read_text(encoding="utf-8"))
    ep.setup_logging(logging.DEBUG if request.verbose else logging.INFO)

    start_indices, start_variables = split_panels(resolve_panels(request))
    recipe_settings = _recipe_settings(request)
    overrides = _setting_overrides(request)

    @functools.lru_cache(maxsize=6)
    def load(start, end):
        """The datasets of the request for a time range, one per satellite that has processed files."""
        return tuple(_load_datasets(dataclasses.replace(request, start_time=start, end_time=end)))

    first_datasets = load(request.start_time, request.end_time) if start_variables else ()
    return (
        ALONG_COORDINATES,
        INDEX_PANELS,
        build_figure,
        clock,
        datetime,
        first_datasets,
        load,
        mo,
        np,
        overrides,
        pitch_angle_channels,
        plot_serve,
        recipe_settings,
        request,
        start_indices,
        start_variables,
        time,
        timezone,
    )


@app.cell
def _(first_datasets, np, pitch_angle_channels, plot_serve, recipe_settings, request, overrides, start_variables):
    # The widgets of energy and pitch angle offer the channels of the first spectral variable (e.g. FEDU).
    spectral = plot_serve.spectral_variables(first_datasets[0], start_variables) if first_datasets else []
    starts = plot_serve.start_values(spectral, recipe_settings, overrides, request.along)
    if spectral:
        energy_options = plot_serve.energy_channels(first_datasets[0], spectral[0])
        angle_options = pitch_angle_channels(first_datasets[0], spectral[0])
    else:
        energy_options, angle_options = np.asarray([]), np.asarray([])
    return angle_options, energy_options, spectral, starts


@app.cell
def _(
    ALONG_COORDINATES,
    INDEX_PANELS,
    angle_options,
    datetime,
    energy_options,
    first_datasets,
    mo,
    np,
    plot_serve,
    request,
    spectral,
    start_indices,
    start_variables,
    starts,
    timezone,
):
    from datetime import timedelta

    _first_day, _last_day = request.start_time.date(), request.end_time.date()
    time_range = mo.ui.date_range(
        start=_first_day - timedelta(days=366),
        stop=max(_last_day, datetime.now(timezone.utc).date()),
        value=(_first_day, _last_day),
        label="Time range",
    )

    _known = set(first_datasets[0].saving_strategy.data_standard.variable_infos) if first_datasets else set()
    _candidates = [*start_variables, *spectral, "L_star", "L_m", "MLT", "B_Eq", "R_Eq"]
    _options = list(dict.fromkeys([*start_indices, *start_variables, *(n for n in _candidates if n in _known)]))
    panels = mo.ui.multiselect(
        options=[*_options, *(n for n in INDEX_PANELS if n not in _options)],
        value=[*start_indices, *start_variables],
        label="Panels (indices such as Kp are downloaded)",
    )

    _energy_labels = {f"{e:g} keV": float(e) for e in energy_options}
    energies = mo.ui.multiselect(
        options=_energy_labels,
        value=[f"{e:g} keV" for e in plot_serve.snap(energy_options, starts["energies_kev"])],
        label=f"{', '.join(spectral) or 'Flux'} energies (none: all channels)",
    )

    # Pitch angles snap to the channels in the files, the ones the figure shows.
    _start_angles = plot_serve.snap(angle_options, starts["pitch_angles_deg"] or [90.0])
    pitch_angle = mo.ui.slider(
        steps=[float(a) for a in angle_options] or [90.0],
        value=_start_angles[0] if _start_angles else 90.0,
        label="Pitch angle [deg]",
        show_value=True,
        debounce=True,
        disabled=angle_options.size == 0,
    )
    pitch_angles = mo.ui.multiselect(
        options={f"{a:g}°": float(a) for a in angle_options},
        value=[f"{a:g}°" for a in _start_angles],
        label="Pitch angles [deg] (one column each)",
    )

    _along_options = {"time": "time", **{name: name for name in ALONG_COORDINATES}}
    along = mo.ui.dropdown(options=_along_options, value=starts["along"] or "time", label="Y axis")

    _vmin, _vmax = starts["vmin"], starts["vmax"]
    auto_limits = mo.ui.switch(value=_vmin is None or _vmax is None, label="Auto colour limits (5th/95th pct)")
    color_exponents = mo.ui.range_slider(
        -3,
        9,
        step=0.5,
        value=(float(np.log10(_vmin)) if _vmin else 0.0, float(np.log10(_vmax)) if _vmax else 6.0),
        label="Colour limits [log10]",
        show_value=True,
        debounce=True,
    )
    colormap = mo.ui.dropdown(
        options=list(dict.fromkeys([request.colormap, "viridis", "rainbow4", "turbo", "plasma", "fire"])),
        value=request.colormap,
        label="Colormap",
    )
    interactive = mo.ui.switch(value=True, label="Zoom / pan")
    return (
        along,
        auto_limits,
        color_exponents,
        colormap,
        energies,
        interactive,
        panels,
        pitch_angle,
        pitch_angles,
        time_range,
    )


@app.cell
def _(
    along,
    auto_limits,
    color_exponents,
    colormap,
    energies,
    interactive,
    mo,
    panels,
    pitch_angle,
    pitch_angles,
    time_range,
):
    # Along a coordinate each pitch angle is a column of panels, so several can be picked.
    _angle_widget = pitch_angle if along.value == "time" else pitch_angles
    mo.vstack(
        [
            mo.hstack([time_range, panels, interactive], justify="start", gap=2),
            mo.hstack([along, energies, _angle_widget], justify="start", gap=2),
            mo.hstack([auto_limits, color_exponents, colormap], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(
    INDEX_PANELS,
    along,
    auto_limits,
    build_figure,
    clock,
    color_exponents,
    colormap,
    datetime,
    energies,
    first_datasets,
    interactive,
    load,
    mo,
    overrides,
    panels,
    pitch_angle,
    pitch_angles,
    plot_serve,
    recipe_settings,
    request,
    time,
    time_range,
    timezone,
):
    _started = clock.perf_counter()
    if tuple(time_range.value) == (request.start_time.date(), request.end_time.date()):
        _start, _end = request.start_time, request.end_time  # keep the times of the command line
    else:
        _start = datetime.combine(time_range.value[0], time(0), tzinfo=timezone.utc)
        _end = datetime.combine(time_range.value[1], time(23, 59, 59), tzinfo=timezone.utc)

    _indices = [name for name in panels.value if name in INDEX_PANELS]
    _names = [name for name in panels.value if name not in INDEX_PANELS]
    _along = None if along.value == "time" else along.value
    _angles = [pitch_angle.value] if _along is None else sorted(pitch_angles.value) or [pitch_angle.value]

    _datasets = list(load(_start, _end)) if _names else []
    _reference = _datasets[0] if _datasets else (first_datasets[0] if first_datasets else None)
    _spectral = plot_serve.spectral_variables(_reference, _names) if _reference is not None else []
    _settings = plot_serve.widget_panel_settings(
        _names,
        _spectral,
        recipe_settings,
        overrides,
        energies_kev=list(energies.value),
        pitch_angles_deg=_angles,
        color_limits=None if auto_limits.value else tuple(10**e for e in color_exponents.value),
        along=_along,
    )

    if not panels.value:
        _output = mo.callout(mo.md("Pick at least one panel."), kind="warn")
    elif _names and not _datasets:
        _output = mo.callout(
            mo.md(f"No processed files between {_start:%Y-%m-%d} and {_end:%Y-%m-%d}; see the terminal."),
            kind="warn",
        )
    else:
        try:
            _figure = build_figure(
                _datasets, _names, _start, _end, indices=_indices, panel_settings=_settings, colormap=colormap.value
            )
        except (TypeError, ValueError) as _error:
            _output = mo.callout(mo.md(f"Could not draw the figure: {_error}"), kind="danger")
        else:
            _elapsed = clock.perf_counter() - _started
            _view = mo.mpl.interactive(_figure) if interactive.value else _figure
            _caption = f"*{_start:%Y-%m-%d %H:%M} to {_end:%Y-%m-%d %H:%M}, drawn in {_elapsed:.2f} s*"
            _output = mo.vstack([mo.md(_caption), _view])
    _output
    return


if __name__ == "__main__":
    app.run()
