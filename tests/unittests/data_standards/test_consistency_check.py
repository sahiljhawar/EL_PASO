# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from el_paso.data_standard import ConsistencyCheck

TIME_LEN = 100
ENERGY_LEN = 20
PITCH_ANGLE_LEN = 50

def test_consistency_correct():

    consistency_check = ConsistencyCheck()

    consistency_check.check_time_size(TIME_LEN, "call1")
    consistency_check.check_time_size(TIME_LEN, "call2")

    consistency_check.check_energy_size(ENERGY_LEN, "call1")
    consistency_check.check_energy_size(ENERGY_LEN, "call2")

    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call1")
    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call2")

def test_consistency_time_wrong():

    consistency_check = ConsistencyCheck()

    consistency_check.check_time_size(TIME_LEN, "call1")

    with pytest.raises(ValueError, match="Time length mismatch!"):
        consistency_check.check_time_size(TIME_LEN+1, "call2")

def test_consistency_energy_wrong():

    consistency_check = ConsistencyCheck()

    consistency_check.check_time_size(TIME_LEN, "call1")
    consistency_check.check_time_size(TIME_LEN, "call2")

    consistency_check.check_energy_size(ENERGY_LEN, "call1")
    consistency_check.check_energy_size(ENERGY_LEN, "call2")

    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call1")
    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call2")

    with pytest.raises(ValueError, match="Energy length mismatch!"):
        consistency_check.check_energy_size(ENERGY_LEN+1, "call3")

def test_consistency_pitch_angle_wrong():

    consistency_check = ConsistencyCheck()

    consistency_check.check_time_size(TIME_LEN, "call1")
    consistency_check.check_time_size(TIME_LEN, "call2")

    consistency_check.check_energy_size(ENERGY_LEN, "call1")
    consistency_check.check_energy_size(ENERGY_LEN, "call2")

    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call1")
    consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN, "call2")

    with pytest.raises(ValueError, match="Pitch angle length mismatch!"):
        consistency_check.check_pitch_angle_size(PITCH_ANGLE_LEN+1, "call3")
