# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from el_paso.processing.magnetic_field_utils.construct_maginput import get_saveable_sw_indices
from el_paso.saving_strategies.lcds_strategy import LCDSStrategy
from el_paso.saving_strategy import OutputFile

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import DataStandard, MagneticFieldLiteral, MFSFormats, StandardName


class SolarWindIndicesStrategy(LCDSStrategy):
    """Save only the solar wind/geomagnetic indices a magnetic field model needs, one file per month.

    Shares the monthly interval and file naming logic of `LCDSStrategy`. The single output file
    holds the time axis plus every index that `mag_field` requires and `data_standard` can store
    (e.g. only Kp for T89).
    """

    def __init__(
        self,
        base_data_path: str | Path,
        mag_field: MagneticFieldLiteral,
        data_standard: DataStandard[StandardName],
        file_format: MFSFormats = "nc",
    ) -> None:
        """Initialize a monthly solar wind indices saving strategy.

        Args:
            base_data_path (str | Path): Directory where monthly files are written.
            mag_field (MagneticFieldLiteral): Magnetic field model name, which decides the saved indices.
            data_standard (DataStandard): Instance of the data standard implementation.
            file_format (MFSFormats): One of ``"nc"``, ``"cdf"``, ``"h5"``, or ``"mat"``.
                A leading dot is also accepted.
        """
        super().__init__(base_data_path, mag_field, data_standard, file_format)

        self.output_files = [
            OutputFile("full", ["Epoch", *get_saveable_sw_indices(mag_field, data_standard)], save_incomplete=True),
        ]

    def get_file_name_stem(self) -> str:
        return "SolarWindIndices"
