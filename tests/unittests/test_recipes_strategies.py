# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the named saving-strategy factories in `el_paso.recipes.strategies`.

These functions only wire mission/satellite/instrument/data-standard literals into the
right `ep.saving_strategies.*Strategy` class, so the strategy classes themselves already
have their own read/write behavior covered under `tests/unittests/saving_strategies/`.
These tests just check each factory builds the right class with the right attributes, and
that overridable keyword-only arguments (`file_format`, `data_standard`) actually take
effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import el_paso as ep
from el_paso.recipes import strategies as rs

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from el_paso.saving_strategy import SavingStrategy


# Each case is (factory, extra_args, extra_kwargs, expected_cls, expected_attrs).
# `base_data_path` (tmp_path) is prepended by the test itself, since it isn't known at
# collection time.
CASES: list[
    tuple[Callable[..., SavingStrategy], tuple[Any, ...], dict[str, Any], type[SavingStrategy], dict[str, Any]]
] = [
    (
        rs.arase_xep_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "xep", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.arase_xep_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "Arase", "satellite": "Arase", "instrument": "XEP", "mag_field": "T89"},
    ),
    (
        rs.arase_mepe_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "ARASE", "satellite": "arase", "instrument": "mepe", "mag_field": "T89"},
    ),
    (
        rs.arase_mepe_h5_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "mepe", "mag_field": "T89", "file_format": ".h5"},
    ),
    (
        rs.arase_mepe_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "mepe", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.arase_pwe_densities_strategy,
        ("T89",),
        {},
        ep.saving_strategies.DensityNetCDFStrategy,
        {"mission": "Arase", "satellite": "Other", "instrument": "PWE", "mag_field": "T89"},
    ),
    (
        rs.dmsp_ssj_electron_strategy,
        ("T89", "f16"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "DMSP", "satellite": "f16", "instrument": "SSJ", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.esa_ngrm_strategy,
        ("T89", "S6A"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "ESA", "satellite": "s6a", "instrument": "ngrm", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.esa_ngrm_strategy,
        ("T89", "Cluster1"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "ESA", "satellite": "cluster1", "instrument": "ngrm", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.goes_r_mps_high_gfz_strategy,
        ("T89", "goes16"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "GOES", "satellite": "goes16", "instrument": "MAGED", "mag_field": "T89"},
    ),
    (
        rs.goes_r_mps_high_netcdf_strategy,
        ("T89", "goes16"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "GOES", "satellite": "goes16", "instrument": "MAGED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.goes_realtime_gfz_strategy,
        ("T89", "16"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "GOES", "satellite": "goes_16", "instrument": "mps-high", "mag_field": "T89"},
    ),
    (
        rs.goes_realtime_netcdf_strategy,
        ("T89", "16"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {
            "mission": "GOES",
            "satellite": "goes_16",
            "instrument": "mps-high",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        rs.gps_cxd_strategy,
        ("T89", "ns41"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "GPS", "satellite": "ns41", "instrument": "cxd", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.poes_ted_strategy,
        ("T89", "noaa19"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "POES", "satellite": "noaa19", "instrument": "TED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.poes_meped_strategy,
        ("T89", "noaa19"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "POES", "satellite": "noaa19", "instrument": "MEPED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.probav_ept_electron_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "ept", "mag_field": "T89"},
    ),
    (
        rs.probav_ept_electron_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "ept", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.probav_ept_proton_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "EPT-proton", "mag_field": "T89"},
    ),
    (
        rs.probav_ept_proton_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {
            "mission": "PROBAV",
            "satellite": "probav",
            "instrument": "EPT-proton",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        rs.rbsp_ect_combined_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "ect_combined", "mag_field": "T89"},
    ),
    (
        rs.rbsp_ect_combined_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {
            "mission": "RBSP",
            "satellite": "rbspa",
            "instrument": "ect_combined",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        rs.rbsp_hope_electron_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "hope", "mag_field": "T89"},
    ),
    (
        rs.rbsp_hope_electron_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "hope", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.rbsp_hope_proton_gfz_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "hope", "mag_field": "T89"},
    ),
    (
        rs.rbsp_hope_proton_netcdf_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "hope", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.rbsp_mageis_electron_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "mageis", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.rbsp_mageis_proton_gfz_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "mageis", "mag_field": "T89"},
    ),
    (
        rs.rbsp_mageis_proton_netcdf_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "mageis", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rs.rbsp_rbspice_proton_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "rbspice", "mag_field": "T89"},
    ),
    (
        rs.rbsp_rbspice_proton_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "rbspice", "mag_field": "T89", "file_format": ".nc"},
    ),
]


def _case_id(factory: Callable[..., SavingStrategy], extra_kwargs: dict[str, Any]) -> str:
    suffix = f"[{','.join(f'{k}={v}' for k, v in extra_kwargs.items())}]" if extra_kwargs else ""
    return getattr(factory, "__name__", repr(factory)) + suffix


CASE_IDS = [_case_id(factory, kwargs) for factory, _args, kwargs, _cls, _attrs in CASES]


@pytest.mark.basic
@pytest.mark.parametrize(
    ("factory", "extra_args", "extra_kwargs", "expected_cls", "expected_attrs"), CASES, ids=CASE_IDS
)
def test_strategy_factory_builds_expected_strategy(
    tmp_path: Path,
    factory: Callable[..., SavingStrategy],
    extra_args: tuple[Any, ...],
    extra_kwargs: dict[str, Any],
    expected_cls: type[SavingStrategy],
    expected_attrs: dict[str, Any],
) -> None:
    strategy = factory(tmp_path, *extra_args, **extra_kwargs)

    assert type(strategy) is expected_cls
    assert strategy.base_data_path == tmp_path

    for attr_name, expected_value in expected_attrs.items():
        assert getattr(strategy, attr_name) == expected_value, attr_name

    if factory is not rs.arase_pwe_densities_strategy:
        # arase_pwe_densities_strategy is the one documented exception, see
        # test_arase_pwe_densities_strategy_data_standard_is_none below.
        assert isinstance(strategy.data_standard, ep.data_standards.GFZStandard)


@pytest.mark.basic
def test_arase_pwe_densities_strategy_data_standard_is_none(tmp_path: Path) -> None:
    """Documents a pre-existing quirk.

    `DensityNetCDFStrategy(data_standard=None)` ends up with `self.data_standard is None`
    rather than falling back to `PRBEMStandard()`, because
    `MonthlyRBStrategy.__init__` re-assigns `self.data_standard` from the raw (unfallen-back)
    argument it's called with. This isn't something the recipe wrapper can fix on its own; it
    just documents the actual observed behavior so a future strategy-class fix doesn't silently
    change what this wrapper returns without anyone noticing.
    """
    strategy = rs.arase_pwe_densities_strategy(tmp_path, "T89")

    assert strategy.data_standard is None


@pytest.mark.basic
@pytest.mark.parametrize("file_format", ["h5", "cdf", "mat"])
def test_file_format_override_takes_effect(tmp_path: Path, file_format: str) -> None:
    strategy = rs.arase_xep_strategy(tmp_path, "T89", file_format=file_format)  # ty:ignore[invalid-argument-type]

    assert strategy.file_format == "." + file_format  # ty:ignore[unresolved-attribute]


def test_file_format_is_keyword_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        rs.arase_xep_strategy(tmp_path, "T89", "nc")  # ty:ignore[too-many-positional-arguments]


@pytest.mark.basic
@pytest.mark.parametrize(
    "data_standard",
    [ep.data_standards.GFZStandard(), ep.data_standards.PRBEMStandard()],
    ids=lambda ds: type(ds).__name__,
)
def test_arase_mepe_data_standard_override_takes_effect(
    tmp_path: Path, data_standard: ep.typing.DataStandard[Any]
) -> None:
    strategy = rs.arase_mepe_netcdf_strategy(tmp_path, "T89", data_standard)

    assert strategy.data_standard is data_standard


@pytest.mark.basic
def test_rbsp_emfisis_waves_strategy(tmp_path: Path) -> None:
    strategy = rs.rbsp_emfisis_waves_strategy(tmp_path, "a")

    assert type(strategy) is ep.saving_strategies.DailyWaveStrategy
    assert strategy.mission == "RBSP"
    assert strategy.satellite == "rbspa"
    assert strategy.instrument == "EMFISIS"
    assert isinstance(strategy.data_standard, ep.data_standards.GFZStandard)


@pytest.mark.basic
def test_rbsp_emfisis_waves_strategy_data_standard_override(tmp_path: Path) -> None:
    prbem = ep.data_standards.PRBEMStandard()

    strategy = rs.rbsp_emfisis_waves_strategy(tmp_path, "b", prbem)

    assert strategy.data_standard is prbem


@pytest.mark.basic
def test_no_recipe_constructs_a_saving_strategy_directly() -> None:
    """Every recipe must build its saving strategy through `el_paso.recipes.strategies`.

    Guards against a new (or edited) `process_*.py` file constructing an
    `ep.saving_strategies.*Strategy(...)` inline instead of adding/reusing a named function in
    `el_paso/recipes/strategies.py`. See `hooks/check_recipe_strategies.py` for the scan
    itself and its (frozen) allowlist of pre-existing exceptions; this test is what makes that
    scan part of the enforced test suite instead of only a pre-commit hook.
    """
    from hooks.check_recipe_strategies import REPO_ROOT, find_violations  # noqa: PLC0415

    violations = find_violations()

    assert not violations, "\n".join(f"{v.path.relative_to(REPO_ROOT)}:{v.lineno}: {v.detail}" for v in violations)


@pytest.mark.basic
def test_every_public_strategy_function_is_tested() -> None:
    """Every public function in `el_paso/recipes/strategies.py` must be covered here.

    Guards against a new strategy function being added to `strategies.py` (for a new or edited
    recipe) without a matching entry in this file's `CASES` table, or a dedicated test function
    for anything that doesn't fit the common `(path, mag_field[, satellite])` shape (e.g.
    `rbsp_emfisis_waves_strategy`). See
    `hooks/check_recipe_strategies.py:find_untested_strategy_functions`.
    """
    from hooks.check_recipe_strategies import find_untested_strategy_functions  # noqa: PLC0415

    untested = find_untested_strategy_functions()

    assert not untested, f"No test coverage for: {', '.join(sorted(untested))}"
