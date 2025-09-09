<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache 2.0
-->

# Compute magnetic field variables

This function serves as a wrapper to calculate a suite of magnetic field and related invariants (like L-star, MLT, B_local, B_eq, invariant Mu,
and invariant K) based on provided time and geocentric coordinates. It leverages the IRBEM library for the underlying computations.

*var_names_to_compute* must be a list of strings containing the requested variable names followed by the magnetic field identifier string.

**Supported variable names**:

- B_local
- B_fofl
- MLT
- R_eq
- Lstar
- PA_eq
- invMu
- invK

**Supported magnetic field strings**:

- OP77Q
- T89
- T96
- T01
- T01s
- TS04|TS05|T04s

Examples of valid entries: *B_local_T89*, *Lstar_TS04*

::: el_paso.processing.compute_magnetic_field_variables
