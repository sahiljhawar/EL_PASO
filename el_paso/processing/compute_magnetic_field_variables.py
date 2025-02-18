from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso.processing as proc
from el_paso.utils import (
    get_variable_by_standard_name,
    get_variable_by_standard_type,
    timed_function,
    validate_standard,
)

if TYPE_CHECKING:
    from el_paso.classes import TimeVariable, Variable


@validate_standard
def compute_magnetic_field_variables(
    input_variables: dict[str, Variable],
    magnetic_field_variables: dict[str, Variable],
    irbem_lib_path: str,
    irbem_options: list,
    num_cores: int,
    time_key: str = None,
    pa_local_key: str = None,
    energy_key: str = None,
    xgeo_key: str = None,
) -> None:
    assert Path(
        irbem_lib_path,
    ).is_file(), f"No library object found under the provided irbem_lib_path: {irbem_lib_path}"
    assert len(irbem_options) == 5, f"irbem_options must be a list with exactly 5 entries! Got: {irbem_options}"

    time_var = (
        input_variables[time_key] if time_key else get_variable_by_standard_name("Epoch_posixtime", input_variables)
    )
    xgeo_var = input_variables[xgeo_key] if xgeo_key else get_variable_by_standard_name("xGEO", input_variables)

    requested_standard_names = [var.standard_name for var in magnetic_field_variables.values()]

    if (
        any("Lstar_" in standard_name for standard_name in requested_standard_names)
        or any("PA_eq" in standard_name for standard_name in requested_standard_names)
        or any("invMu" in standard_name for standard_name in requested_standard_names)
        or any("invK" in standard_name for standard_name in requested_standard_names)
    ):
        pa_local_var = (
            input_variables[pa_local_key]
            if pa_local_key
            else get_variable_by_standard_name("PA_local", input_variables)
        )

    if any("invMu_" in standard_name for standard_name in requested_standard_names):
        energy_var = (
            input_variables[energy_key] if energy_key else get_variable_by_standard_type("Energy", input_variables)
        )

    # collect magnetic_field results in this dictionary; holds the data array and the unit
    magnetic_field_results: dict[str, tuple[np.ndarray, u.BaseUnit]] = {}

    # construct the maginput (Kp, Dst, ...) used for all irbem calls
    maginput = proc.construct_maginput(time_var.data)
    irbem_input = proc.IrbemInput(irbem_lib_path, "", maginput, irbem_options, num_cores)

    for var in magnetic_field_variables.values():
        # check if the value has been calculated already
        if var.standard_name in magnetic_field_results:
            var.data, var.metadata.unit = magnetic_field_results[var.standard_name]
        else:
            magnetic_field_str = var.standard_name.split("_")[-1]
            irbem_input.magnetic_field_str = magnetic_field_str

            if "B_local" in var.standard_name:
                magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

            elif "B_fofl" in var.standard_name:
                magnetic_field_results |= proc.get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)

            elif "MLT" in var.standard_name:
                magnetic_field_results |= proc.get_MLT(xgeo_var, time_var, irbem_input)

            elif "R_eq" in var.standard_name or "B_eq" in var.standard_name or "xGEO" in var.standard_name:
                magnetic_field_results |= proc.get_magequator(xgeo_var, time_var, irbem_input)

            elif "Lstar_" in var.standard_name:
                magnetic_field_results |= proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)

            elif "PA_eq" in var.standard_name:
                magnetic_field_results |= _get_pa_eq(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    magnetic_field_results,
                    irbem_input,
                )

            elif "invMu" in var.standard_name:
                magnetic_field_results |= _get_invariant_mu(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    energy_var,
                    magnetic_field_results,
                    irbem_input,
                )

            elif "invK" in var.standard_name:
                magnetic_field_results |= _get_invariant_K(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    magnetic_field_results,
                    irbem_input,
                )

            else:
                continue

        var.data, var.metadata.unit = magnetic_field_results[var.standard_name]
        var.metadata.processing_notes += f"; irbem_options={irbem_options}"
        var.time_variable = time_var


@timed_function("Equatorial pitch angle calculation")
def _get_pa_eq(
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    magnetic_field_results: dict,
    irbem_input: proc.IrbemInput,
):
    logging.info("\tCalculating equatorial pitch angle ...")

    pa_local = (pa_local_var.data * pa_local_var.metadata.unit).to_value("")

    if ("B_eq_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_magequator(xgeo_var, time_var, irbem_input)
    if ("B_local_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    B_eq, _ = magnetic_field_results["B_eq_" + irbem_input.magnetic_field_str]
    B_local, _ = magnetic_field_results["B_local_" + irbem_input.magnetic_field_str]

    pa_eq_rad = np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:, np.newaxis])
    pa_eq_deg = np.rad2deg(pa_eq_rad)
    magnetic_field_results["PA_eq_" + irbem_input.magnetic_field_str] = (pa_eq_deg, u.deg)

    return magnetic_field_results


@timed_function("Invariant mu calculation")
def _get_invariant_mu(xgeo_var:Variable, time_var:TimeVariable, pa_local_var:Variable, energy_var:Variable, magnetic_field_results:dict, irbem_input: proc.IrbemInput):
    logging.info("\tCalculating invariant mu ...")

    pa_local = (pa_local_var.data * pa_local_var.metadata.unit).to_value("")
    energy = (energy_var.data * energy_var.metadata.unit).to_value(u.MeV)

    species_char = energy_var.standard.standard_name[-3]

    if ("B_local_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    # load needed data and convert to correct units
    B_local, B_local_unit = magnetic_field_results["B_local_" + irbem_input.magnetic_field_str]
    B_local = (B_local * B_local_unit).to_value(u.G)

    magnetic_field_results["invMu_" + irbem_input.magnetic_field_str] = proc.compute_invariant_mu(
        energy,
        pa_local,
        B_local,
        species_char,
    )

    return magnetic_field_results


@timed_function("Invariant K calculation")
def _get_invariant_K(  # noqa: N802
    xgeo_var: Variable,
    time_var: TimeVariable,
    pa_local_var: Variable,
    magnetic_field_results: dict,
    irbem_input: proc.IrbemInput,
) -> dict:
    logging.info("\tCalculating invariant K ...")

    if ("XJ_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)
    if ("B_mirr_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_mirror_point(xgeo_var, time_var, pa_local_var, irbem_input)

    # load needed data and convert to correct units
    B_mirr = magnetic_field_results["B_mirr_" + irbem_input.magnetic_field_str][0]  # noqa: N806
    B_mirr = (B_mirr * magnetic_field_results["B_mirr_" + irbem_input.magnetic_field_str][1]).to_value(u.G)  # noqa: N806
    XJ = magnetic_field_results["XJ_" + irbem_input.magnetic_field_str][0]  # noqa: N806

    magnetic_field_results["invK_" + irbem_input.magnetic_field_str] = (
        proc.compute_invariant_K(B_mirr, XJ),
        u.RE * u.G**0.5,
    )

    return magnetic_field_results
