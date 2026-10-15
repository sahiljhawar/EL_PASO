# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Recipes for products derived from a magnetic field model (or its inputs) alone."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, _lazy_dir, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "process_lcds": ["lcds_strategy", "process_lcds"],
        "process_solar_wind_indices": ["process_solar_wind_indices", "solar_wind_indices_strategy"],
    },
)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.model.process_lcds import lcds_strategy, process_lcds
    from el_paso.recipes.model.process_solar_wind_indices import (
        process_solar_wind_indices,
        solar_wind_indices_strategy,
    )

    __all__ = [
        "lcds_strategy",
        "process_lcds",
        "process_solar_wind_indices",
        "solar_wind_indices_strategy",
    ]
