<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Overview

This section provides a detailed reference for all modules, classes, and functions in `el_paso`.

## Core classes
[DataStandard](data_standard.md)

[SavingStrategy](saving_strategy.md)

[Variable](variable.md)

## Core functions
[download](download.md)

[extract_variables_from_files](extract_variables_from_files.md)

[save](save.md)

## Utilities

[General utilities](utilities/general_utilities.md)

[Load geomagnetic indices and solar wind parameters](utilities/load_indices_solar_wind_parameters.md)

[Scripts](utilities/scripts.md)

[Release mode](utilities/release_mode.md)

[Units](utilities/units.md)

## Processing functions

[bin_by_time](processing/bin_by_time.md)

[compute_invariank_K](processing/compute_invariant_K.md)

[compute_invariank_mu](processing/compute_invariant_mu.md)

[compute_magnetic_field_variables](processing/compute_magnetic_field_variables.md)

[compute_phase_space_density](processing/compute_phase_space_density.md)

[fold_pitch_angles_and_flux](processing/fold_pitch_angles_and_flux.md)

[convert_string_to_datetime](processing/convert_string_to_datetime.md)

<!-- ::: el_paso.processing.compute_equatorial_plasmaspheric_density

::: el_paso.processing.construct_pitch_angle_distribution

::: el_paso.processing.convert_string_to_datetime

::: el_paso.processing.extrapolate_leo_to_equatorial

::: el_paso.processing.get_real_time_tipsod

::: el_paso.processing.magnetic_field_functions -->

## Saving standards

[DataOrgStrategy](saving_strategies/data_org.md)

[MonthlyH5Strategy](saving_strategies/monthly_h5.md)

[MonthlyNetCDFStrategy](saving_strategies/monthly_netcdf.md)

[SingleFileStrategy](saving_strategies/single_file.md)

## Data Standards

[DataOrgStandard](data_standards/data_org.md)

[PRBEMStandard](data_standards/prbem.md)