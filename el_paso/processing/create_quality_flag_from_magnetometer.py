# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from astropy import units as u

import el_paso as ep


def create_quality_flag_from_magnetometer(bt_var: ep.Variable, threshold: int = 500) -> ep.Variable:
    """Compute a quality mask for magnetometer data using Bt.

    The mask flags samples that are considered valid based on two criteria:
    1. Magnetic field magnitude (Bt) must be positive.
    2. Consecutive differences |ΔBt| must be below the spike threshold.

    Args:
        bt_var (ep.Variable): Variable holding the total magnetic field.
        threshold (int, optional): Spike detection threshold in nT.
            Defaults to 500.

    Returns:
        ep.Variable: A boolean mask variable where ``True`` indicates valid data and
        ``False`` indicates flagged samples.
    """
    bt = bt_var.get_data(u.nT)

    mask_positive = bt > 0

    delta_b = np.diff(bt, prepend=bt[0])
    mask_spike = np.abs(delta_b) < threshold

    mask = ep.Variable(
        data=mask_positive & mask_spike,
        original_unit=u.dimensionless_unscaled,
    )

    return mask
