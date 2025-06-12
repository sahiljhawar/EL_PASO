from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso.processing as proc
from el_paso import Variable
from el_paso.processing.magnetic_field_functions import IrbemOutput
from el_paso.utils import timed_function


def compute_magnetic_field_variables(
    time_var:Variable,
    xgeo_var:Variable,
    var_names_to_compute: list[str],
    irbem_lib_path: str,
    irbem_options: list[int],
    num_cores: int,
    pa_local_var: Variable|None = None,
    energy_var: Variable|None = None,
    particle_species:Literal["electron", "proton"]|None = None,
) -> dict[str, Variable]:

    assert Path(
        irbem_lib_path,
    ).is_file(), f"No library object found under the provided irbem_lib_path: {irbem_lib_path}"
    assert len(irbem_options) == 5, f"irbem_options must be a list with exactly 5 entries! Got: {irbem_options}"

    if (
        (any("Lstar_" in var_name for var_name in var_names_to_compute)
        or any("PA_eq" in var_name for var_name in var_names_to_compute)
        or any("invMu" in var_name for var_name in var_names_to_compute)
        or any("invK" in var_name for var_name in var_names_to_compute))
        and pa_local_var is None
    ):
        msg = "Pitch-angle dependent variable is requested but local pitch angles are not provided!"
        raise ValueError(msg)

    pa_local_var = typing.cast("Variable", pa_local_var)

    if any("invMu_" in var_name for var_name in var_names_to_compute):
        if energy_var is None:
            msg = "Energy dependent variable is requested but energies are not provided!"
            raise ValueError(msg)
        if particle_species is None:
            msg = "Particle-species dependent variable is requested but particle_species is not provided!"
            raise ValueError(msg)

    energy_var = typing.cast("Variable", energy_var)
    particle_species = typing.cast("Literal['electron', 'proton']", particle_species)

    # collect magnetic_field results in this dictionary; holds the data array and the unit
    magnetic_field_results: dict[str, proc.IrbemOutput] = {}

    # construct the maginput (Kp, Dst, ...) used for all irbem calls
    maginput = proc.construct_maginput(time_var)
    irbem_input = proc.IrbemInput(irbem_lib_path, "", maginput, irbem_options, num_cores)

    computed_variables:dict[str, Variable] = {}

    for var_name in var_names_to_compute:
        new_var = Variable(u.dimensionless_unscaled)

        # check if the value has been calculated already
        if var_name in magnetic_field_results:
            new_var.set_data(magnetic_field_results[var_name].arr, magnetic_field_results[var_name].unit)
        else:
            magnetic_field_str = var_name.split("_")[-1]
            irbem_input.magnetic_field_str = magnetic_field_str

            if "B_local" in var_name:
                magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

            elif "B_fofl" in var_name:
                magnetic_field_results |= proc.get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)

            elif "MLT" in var_name:
                magnetic_field_results |= proc.get_MLT(xgeo_var, time_var, irbem_input)

            elif "R_eq" in var_name or "B_eq" in var_name or "xGEO" in var_name:
                magnetic_field_results |= proc.get_magequator(xgeo_var, time_var, irbem_input)

            elif "Lstar_" in var_name:
                magnetic_field_results |= proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)

            elif "PA_eq" in var_name:
                magnetic_field_results |= _get_pa_eq(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    magnetic_field_results,
                    irbem_input,
                )

            elif "invMu" in var_name:
                magnetic_field_results |= _get_invariant_mu(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    energy_var,
                    magnetic_field_results,
                    irbem_input,
                    particle_species,
                )

            elif "invK" in var_name:
                magnetic_field_results |= _get_invariant_K(
                    xgeo_var,
                    time_var,
                    pa_local_var,
                    magnetic_field_results,
                    irbem_input,
                )

            else:
                continue

        new_var.set_data(magnetic_field_results[var_name][0], magnetic_field_results[var_name][1])
        new_var.metadata.add_processing_note(f"Variable generated with irbem_options={irbem_options}")

        computed_variables[var_name] = new_var

    return computed_variables

@timed_function("Equatorial pitch angle calculation")
def _get_pa_eq(
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    magnetic_field_results: dict[str, proc.IrbemOutput],
    irbem_input: proc.IrbemInput,
) -> dict[str, proc.IrbemOutput]:
    print("\tCalculating equatorial pitch angle ...")

    pa_local = (pa_local_var.get_data() * pa_local_var.metadata.unit).to_value(u.radian)

    if ("B_eq_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_magequator(xgeo_var, time_var, irbem_input)
    if ("B_local_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    B_eq, _ = magnetic_field_results["B_eq_" + irbem_input.magnetic_field_str]
    B_local, _ = magnetic_field_results["B_local_" + irbem_input.magnetic_field_str]

    pa_eq_rad = np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:, np.newaxis])
    pa_eq_deg = np.rad2deg(pa_eq_rad)
    magnetic_field_results["PA_eq_" + irbem_input.magnetic_field_str] = IrbemOutput(pa_eq_deg, u.deg)

    return magnetic_field_results


@timed_function("Invariant mu calculation")
def _get_invariant_mu(xgeo_var:Variable,
                      time_var:Variable,
                      pa_local_var:Variable,
                      energy_var:Variable,
                      magnetic_field_results:dict[str, proc.IrbemOutput],
                      irbem_input: proc.IrbemInput,
                      particle_species: Literal["electron", "proton"]) -> dict[str, proc.IrbemOutput]:
    print("\tCalculating invariant mu ...")

    pa_local = (pa_local_var.get_data() * pa_local_var.metadata.unit).to_value(u.radian)
    energy = (energy_var.get_data() * energy_var.metadata.unit).to_value(u.MeV)

    if ("B_local_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    # load needed data and convert to correct units
    B_local, B_local_unit = magnetic_field_results["B_local_" + irbem_input.magnetic_field_str]
    B_local = (B_local * B_local_unit).to_value(u.G)

    magnetic_field_results["invMu_" + irbem_input.magnetic_field_str] = proc.compute_invariant_mu(
        energy,
        pa_local,
        B_local,
        particle_species,
    )

    return magnetic_field_results


@timed_function("Invariant K calculation")
def _get_invariant_K(  # noqa: N802
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    magnetic_field_results: dict[str, proc.IrbemOutput],
    irbem_input: proc.IrbemInput,
) -> dict[str, proc.IrbemOutput]:
    print("\tCalculating invariant K ...")

    if ("XJ_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)
    if ("B_mirr_" + irbem_input.magnetic_field_str) not in magnetic_field_results:
        magnetic_field_results |= proc.get_mirror_point(xgeo_var, time_var, pa_local_var, irbem_input)

    # load needed data and convert to correct units
    B_mirr = magnetic_field_results["B_mirr_" + irbem_input.magnetic_field_str].arr  # noqa: N806
    B_mirr = (B_mirr * magnetic_field_results["B_mirr_" + irbem_input.magnetic_field_str].unit).to_value(u.G)  # noqa: N806
    XJ = magnetic_field_results["XJ_" + irbem_input.magnetic_field_str].arr  # noqa: N806

    magnetic_field_results["invK_" + irbem_input.magnetic_field_str] = IrbemOutput(
        proc.compute_invariant_K(B_mirr, XJ),
        u.RE * u.G**0.5,
    )

    return magnetic_field_results
