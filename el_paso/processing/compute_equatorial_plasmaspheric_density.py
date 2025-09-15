# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from el_paso import Variable


def compute_equatorial_plasmaspheric_density(density:Variable,
                                             R0:Variable,
                                             R_local:Variable,
                                             ) -> None:

    inside_pp = density.data >= 10 * (6.6 / R0.data)**4

    alpha = np.full_like(density.data, 1)
    alpha[inside_pp] = 2.5

    density.data = density.data / (R0.data / R_local.data)**alpha
