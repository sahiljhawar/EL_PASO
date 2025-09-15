# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # noqa: N999
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep


def compute_invariant_K(bmirr:ep.Variable,  # noqa: N802
                        xj:ep.Variable) -> ep.Variable:
    r"""Computes the invariant K from mirror magnetic field and the second adiabatic invariant (I).

    The invariant K is calculated as the square root of the mirror magnetic
    field ($B_{mirr}$) multiplied by the second adiabatic invariant ($X_J$).
    The unit of the resulting invariant K is $R_E \cdot G^{0.5}$.

    Args:
        bmirr (ep.Variable): A Variable object containing the mirror magnetic
            field data. The data is expected to be convertible to Gauss (u.G).
        xj (ep.Variable): A Variable object containing the second adiabativ invariant (I).

    Returns:
        ep.Variable: A new Variable object containing the computed invariant K
            data with unit $R_E \cdot G^{0.5}$.
    """
    bmirr_data = bmirr.get_data(u.G).astype(np.float64)

    xj_data = xj.get_data(ep.units.RE).astype(np.float64)

    inv_K_var = ep.Variable(data = np.sqrt(bmirr_data) * xj_data,  # noqa: N806
                            original_unit=ep.units.RE * u.G**0.5)  # type: ignore[reportUnknownArgumentType]

    inv_K_var.metadata.add_processing_note("Created with compute_invariant_K from B_mirr and XJ.")

    return inv_K_var
