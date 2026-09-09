# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Named saving-strategy factories for each processing recipe.

Each function here fixes the mission/satellite/instrument/data-standard details for one
recipe's output and exposes only the arguments that actually vary between calls (the
base data path and the magnetic field model, plus the satellite identifier for recipes
that process more than one spacecraft). This keeps the strategy construction in each
``process_*`` recipe down to a single call instead of repeating the full
``ep.saving_strategies.*Strategy(...)`` invocation everywhere. This can also be useful for
users who want to import a strategy and provide the minimal configuration to construct a
strategy to read and write ``ep.DataSet``s.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import el_paso as ep

if TYPE_CHECKING:
    from collections.abc import Callable

    from el_paso.saving_strategy import SavingStrategy
    from el_paso.typing import DataStandard, MagneticFieldLiteral, MFSFormats, StandardName


def _build_strategy(
    strategy_cls: Callable[..., SavingStrategy],
    base_data_path: str | Path,
    mission: str,
    satellite: str,
    instrument: str,
    mag_field: MagneticFieldLiteral,
    data_standard: DataStandard[StandardName] | None = None,
    **kwargs: object,
) -> SavingStrategy:
    """Common constructor shared by the GFZ/MonthlyRB/DailyLEORB strategy family."""
    return strategy_cls(
        Path(base_data_path),
        mission,
        satellite,
        instrument,
        mag_field,
        data_standard=data_standard or ep.data_standards.GFZStandard(),
        **kwargs,
    )


def arase_xep_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly strategy for Arase XEP."""
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "Arase",
        "arase",
        "xep",
        mag_field,
        file_format=file_format,
    )


def arase_xep_gfz_strategy(base_data_path: str | Path, mag_field: MagneticFieldLiteral) -> SavingStrategy:
    """Legacy GFZ .mat strategy for Arase XEP."""
    return _build_strategy(ep.saving_strategies.GFZStrategy, base_data_path, "Arase", "Arase", "XEP", mag_field)


def arase_mepe_gfz_strategy(
    base_data_path: str | Path,
    mag_field: MagneticFieldLiteral,
    data_standard: DataStandard[StandardName] | None = None,
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for Arase MEP-e."""
    return _build_strategy(
        ep.saving_strategies.GFZStrategy, base_data_path, "ARASE", "arase", "mepe", mag_field, data_standard
    )


def arase_mepe_h5_strategy(
    base_data_path: str | Path,
    mag_field: MagneticFieldLiteral,
    data_standard: DataStandard[StandardName] | None = None,
    *,
    file_format: MFSFormats = "h5",
) -> SavingStrategy:
    """Monthly HDF5 strategy for Arase MEP-e."""
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "Arase",
        "arase",
        "mepe",
        mag_field,
        data_standard,
        file_format=file_format,
    )


def arase_mepe_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: MagneticFieldLiteral,
    data_standard: DataStandard[StandardName] | None = None,
    *,
    file_format: MFSFormats = "nc",
) -> SavingStrategy:
    """Monthly NetCDF strategy for Arase MEP-e."""
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "Arase",
        "arase",
        "mepe",
        mag_field,
        data_standard,
        file_format=file_format,
    )


def arase_pwe_densities_strategy(base_data_path: str | Path, mag_field: MagneticFieldLiteral) -> SavingStrategy:
    """Monthly NetCDF density strategy for Arase PWE."""
    return ep.saving_strategies.DensityNetCDFStrategy(
        base_data_path=base_data_path,
        mission="Arase",
        satellite="Other",
        instrument="PWE",
        mag_field=mag_field,
    )


def dmsp_ssj_electron_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Daily LEO/RB strategy for DMSP SSJ electrons."""
    return _build_strategy(
        ep.saving_strategies.DailyLEORBStrategy,
        base_data_path,
        "DMSP",
        satellite,
        "SSJ",
        mag_field,
        file_format=file_format,
    )


def esa_ngrm_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = ".nc"
) -> SavingStrategy:
    """NetCDF strategy for ESA NGRM, daily for S6 satellites and monthly otherwise."""
    strategy_cls = (
        ep.saving_strategies.DailyLEORBStrategy
        if satellite.startswith("S6")
        else ep.saving_strategies.MonthlyRBStrategy
    )
    return _build_strategy(
        strategy_cls,
        base_data_path,
        "ESA",
        satellite.lower(),
        "ngrm",
        mag_field,
        file_format=file_format,
    )


def _goes_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, instrument: str
) -> SavingStrategy:
    return _build_strategy(ep.saving_strategies.GFZStrategy, base_data_path, "GOES", satellite, instrument, mag_field)


def _goes_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: MagneticFieldLiteral,
    satellite: str,
    instrument: str,
    file_format: MFSFormats,
) -> SavingStrategy:
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "GOES",
        satellite,
        instrument,
        mag_field,
        file_format=file_format,
    )


def goes_r_mps_high_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for GOES-R MPS-HI/MAGED."""
    return _goes_gfz_strategy(base_data_path, mag_field, satellite, "MAGED")


def goes_r_mps_high_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for GOES-R MPS-HI/MAGED."""
    return _goes_netcdf_strategy(base_data_path, mag_field, satellite, "MAGED", file_format)


def goes_realtime_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for GOES realtime mps-high."""
    return _goes_gfz_strategy(base_data_path, mag_field, "goes_" + satellite, "mps-high")


def goes_realtime_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = ".nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for GOES realtime mps-high."""
    return _goes_netcdf_strategy(base_data_path, mag_field, "goes_" + satellite, "mps-high", file_format)


def gps_cxd_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for LANL GPS CXD."""
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "GPS",
        satellite,
        "cxd",
        mag_field,
        file_format=file_format,
    )


def poes_ted_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Daily LEO/RB strategy for POES TED."""
    return _build_strategy(
        ep.saving_strategies.DailyLEORBStrategy,
        base_data_path,
        "POES",
        satellite,
        "TED",
        mag_field,
        file_format=file_format,
    )


def poes_meped_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Daily LEO/RB strategy for POES MEPED."""
    return _build_strategy(
        ep.saving_strategies.DailyLEORBStrategy,
        base_data_path,
        "POES",
        satellite,
        "MEPED",
        mag_field,
        file_format=file_format,
    )


def _probav_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, instrument: str
) -> SavingStrategy:
    return _build_strategy(ep.saving_strategies.GFZStrategy, base_data_path, "PROBAV", "probav", instrument, mag_field)


def _probav_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, instrument: str, file_format: MFSFormats
) -> SavingStrategy:
    return _build_strategy(
        ep.saving_strategies.DailyLEORBStrategy,
        base_data_path,
        "PROBAV",
        "probav",
        instrument,
        mag_field,
        file_format=file_format,
    )


def probav_ept_electron_gfz_strategy(base_data_path: str | Path, mag_field: MagneticFieldLiteral) -> SavingStrategy:
    """Legacy GFZ .mat strategy for PROBA-V EPT electrons."""
    return _probav_gfz_strategy(base_data_path, mag_field, "ept")


def probav_ept_electron_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, *, file_format: MFSFormats = ".nc"
) -> SavingStrategy:
    """Daily LEO/RB NetCDF strategy for PROBA-V EPT electrons."""
    return _probav_netcdf_strategy(base_data_path, mag_field, "ept", file_format)


def probav_ept_proton_gfz_strategy(base_data_path: str | Path, mag_field: MagneticFieldLiteral) -> SavingStrategy:
    """Legacy GFZ .mat strategy for PROBA-V EPT protons."""
    return _probav_gfz_strategy(base_data_path, mag_field, "EPT-proton")


def probav_ept_proton_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, *, file_format: MFSFormats = ".nc"
) -> SavingStrategy:
    """Daily LEO/RB NetCDF strategy for PROBA-V EPT protons."""
    return _probav_netcdf_strategy(base_data_path, mag_field, "EPT-proton", file_format)


def _rbsp_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, instrument: str
) -> SavingStrategy:
    return _build_strategy(
        ep.saving_strategies.GFZStrategy, base_data_path, "RBSP", "rbsp" + satellite, instrument, mag_field
    )


def _rbsp_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: MagneticFieldLiteral,
    satellite: str,
    instrument: str,
    file_format: MFSFormats,
) -> SavingStrategy:
    return _build_strategy(
        ep.saving_strategies.MonthlyRBStrategy,
        base_data_path,
        "RBSP",
        "rbsp" + satellite,
        instrument,
        mag_field,
        file_format=file_format,
    )


def rbsp_ect_combined_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for RBSP ECT-combined."""
    return _rbsp_gfz_strategy(base_data_path, mag_field, satellite, "ect_combined")


def rbsp_ect_combined_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP ECT-combined."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "ect_combined", file_format)


def rbsp_hope_electron_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for RBSP HOPE electrons."""
    return _rbsp_gfz_strategy(base_data_path, mag_field, satellite, "hope")


def rbsp_hope_electron_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP HOPE electrons."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "hope", file_format)


def rbsp_hope_proton_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for RBSP HOPE protons."""
    return _rbsp_gfz_strategy(base_data_path, mag_field, satellite, "hope")


def rbsp_hope_proton_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP HOPE protons."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "hope", file_format)


def rbsp_mageis_electron_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP MagEIS electrons."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "mageis", file_format)


def rbsp_mageis_proton_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for RBSP MagEIS protons."""
    return _rbsp_gfz_strategy(base_data_path, mag_field, satellite, "mageis")


def rbsp_mageis_proton_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP MagEIS protons."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "mageis", file_format)


def rbsp_rbspice_proton_gfz_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str
) -> SavingStrategy:
    """Legacy GFZ .mat strategy for RBSP RBSPICE protons."""
    return _rbsp_gfz_strategy(base_data_path, mag_field, satellite, "rbspice")


def rbsp_rbspice_proton_netcdf_strategy(
    base_data_path: str | Path, mag_field: MagneticFieldLiteral, satellite: str, *, file_format: MFSFormats = "nc"
) -> SavingStrategy:
    """Monthly NetCDF strategy for RBSP RBSPICE protons."""
    return _rbsp_netcdf_strategy(base_data_path, mag_field, satellite, "rbspice", file_format)


def rbsp_emfisis_waves_strategy(
    base_data_path: str | Path,
    satellite: str,
    data_standard: DataStandard[StandardName] | None = None,
) -> SavingStrategy:
    """Daily NetCDF wave strategy for RBSP EMFISIS."""
    return ep.saving_strategies.DailyWaveStrategy(
        Path(base_data_path),
        "RBSP",
        "rbsp" + satellite,
        "EMFISIS",
        data_standard or ep.data_standards.GFZStandard(),
    )
