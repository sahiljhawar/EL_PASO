# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache 2.0

import numpy as np

from el_paso.post_process.fold_pitch_angles_and_flux import _fold_pitch_angles_and_flux


def test_pitch_angle_folding():

    pa = np.array([[1, 10, 30, 60, 90, 120, 150, 170, 179]])
    flux = np.array([[[1,2,3,4,5,5,4,3,2]]])

    print(_fold_pitch_angles_and_flux(pa, flux))

if __name__ == '__main__':
    test_pitch_angle_folding()