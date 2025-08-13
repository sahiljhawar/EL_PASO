from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso.processing as proc
from el_paso import Variable
from el_paso.utils import timed_function

logger = logging.getLogger(__name__)

def compute_magnetic_field_variables(
    time_var:Variable,
    xgeo_var:Variable,
    var_names_to_compute: list[str],
    irbem_lib_path: str,
    irbem_options: list[int],
    num_cores: int,
    indices_solar_wind: dict[str, Variable]|None = None,
    pa_local_var: Variable|None = None,
    energy_var: Variable|None = None,
    particle_species:Literal["electron", "proton"]|None = None,
) -> dict[str, Variable]:
    """Computes various magnetic field-related variables using the IRBEM library.

    This function serves as a wrapper to calculate a suite of magnetic field
    and related invariants (like L-star, MLT, B_local, B_eq, invariant Mu,
    and invariant K) based on provided time and geocentric coordinates. It
    leverages the IRBEM library for the underlying computations.

    Args:
        time_var (Variable): A Variable object containing time data.
        xgeo_var (Variable): A Variable object containing geocentric (XGEO)
            coordinates. Expected to be a 2D array (time, 3) where the last
            dimension represents X, Y, Z coordinates.
        var_names_to_compute (list[str]): A list of string identifiers for the
            magnetic field variables to be computed.
        irbem_lib_path (str): The file path to the compiled IRBEM library object
            (a `.so` or `.dll` file).
        irbem_options (list[int]): A list of 5 integer options for the IRBEM library
            calls, controlling aspects like model selection, bounce tracing, etc.
        num_cores (int): The number of CPU cores to use for parallel processing
            within IRBEM calls.
        indices_solar_wind (dict[str, Variable] | None): Optional. A dictionary
            containing solar wind indices (e.g., "Kp", "Dst") as `Variable` objects.
            Defaults to None.
        pa_local_var (Variable | None): Optional. A Variable object containing
            local pitch angle data. Required if any pitch-angle dependent variables
            (e.g., "PA_eq", "Lstar", "invMu", "invK") are requested. Defaults to None.
        energy_var (Variable | None): Optional. A Variable object containing
            particle energy data in MeV. Required if "invMu" is requested. Defaults to None.
        particle_species (Literal["electron", "proton"] | None): Optional. The
            species of particle (e.g., "electron", "proton"). Required if "invMu"
            is requested. Defaults to None.

    Returns:
        dict[str, Variable]: A dictionary where keys are the computed variable
        names (matching `var_names_to_compute`) and values are their corresponding
        `Variable` objects containing the calculated data and metadata.

    Raises:
        FileNotFoundError: If no IRBEM library object is found at the provided `irbem_lib_path`.
        ValueError:
            - If `irbem_options` does not contain exactly 5 entries.
            - If a pitch-angle dependent variable is requested but `pa_local_var` is not provided.
            - If an energy-dependent variable ("invMu") is requested but `energy_var` is not provided.
            - If a particle-species dependent variable ("invMu") is requested but `particle_species` is not provided.
        NotImplementedError: If a requested variable name in `var_names_to_compute`
            is not supported by this function.

    Notes:
        - The function internally constructs an `IrbemInput` object for each unique
          magnetic field model encountered in `var_names_to_compute`.
        - Intermediate computed variables (e.g., `B_eq`, `B_local`) are cached
          within the function to avoid redundant IRBEM calls when multiple
          dependent variables are requested.
    """
    if not Path(irbem_lib_path).is_file():
        msg = f"No library object found under the provided irbem_lib_path: {irbem_lib_path}"
        raise FileNotFoundError(msg)

    if len(irbem_options) != 5:  # noqa: PLR2004
        msg = f"irbem_options must be a list with exactly 5 entries! Got: {irbem_options}"
        raise ValueError(msg)

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

    # collect magnetic_field results in this dictionary
    computed_variables:dict[str, Variable] = {}

    for var_name in var_names_to_compute:

        # check if the value has been calculated already
        if var_name in computed_variables:
            continue

        magnetic_field_str = var_name.split("_")[-1]

        # construct the maginput (Kp, Dst, ...) used for all irbem calls
        # maginput_old = proc.construct_maginput(time_var, indices_solar_wind, magnetic_field_str)
        maginput = proc.magnetic_field_utils.construct_maginput(time_var, indices_solar_wind, magnetic_field_str)
        # maginput_old = proc.construct_maginput(time_var, indices_solar_wind, magnetic_field_str)

        # print(maginput_new["W3"])
        # print(maginput_old["W3"])

        # import matplotlib.pyplot as plt

        # keys_to_compare = [f"W{i}" for i in range(1, 7)]
        # fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        # axes = axes.flatten()

        # for idx, key in enumerate(keys_to_compare):
        #     axes[idx].plot(maginput_old[key], label=f"{key} old", linestyle='--', marker='o')
        #     axes[idx].plot(maginput_new[key], label=f"{key} new", linestyle='-', marker='x')
        #     axes[idx].set_title(f"Comparison: {key}")
        #     axes[idx].set_ylabel("Value")
        #     axes[idx].legend()
        #     axes[idx].grid(True)

        # axes[-1].set_xlabel("Index")
        # plt.tight_layout()
        # plt.savefig("test.png")
        # asdf

        irbem_input = proc.IrbemInput(irbem_lib_path, magnetic_field_str, maginput, irbem_options, num_cores)

        computed_variables |= _get_result(var_name,
                                          xgeo_var,
                                          time_var,
                                          pa_local_var,
                                          energy_var,
                                          computed_variables,
                                          irbem_input,
                                          particle_species)

    # only return the requested variables
    computed_variables = {
        var_name: computed_variables[var_name]
        for var_name in computed_variables
        if var_name in var_names_to_compute
    }

    return computed_variables  # noqa: RET504

def _get_result(var_name: str,  # noqa: PLR0911
                xgeo_var: Variable,
                time_var: Variable,
                pa_local_var: Variable,
                energy_var: Variable,
                computed_vars: dict[str, Variable],
                irbem_input: proc.IrbemInput,
                particle_species: Literal["electron", "proton"]) -> dict[str, Variable]:
    """Helper function to get the result for a specific variable name."""
    if "B_local" in var_name:
        return proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    if "B_fofl" in var_name:
        return proc.get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)

    if "MLT" in var_name:
        return proc.get_MLT(xgeo_var, time_var, irbem_input)

    if "R_eq" in var_name or "B_eq" in var_name or "xGEO" in var_name:
        return proc.get_magequator(xgeo_var, time_var, irbem_input)

    if "Lstar_" in var_name:
        return proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)

    if "PA_eq" in var_name:
        return _get_pa_eq(
            xgeo_var,
            time_var,
            pa_local_var,
            computed_vars,
            irbem_input,
        )

    if "invMu" in var_name:
        return _get_invariant_mu(
            xgeo_var,
            time_var,
            pa_local_var,
            energy_var,
            computed_vars,
            irbem_input,
            particle_species,
        )

    if "invK" in var_name:
        return _get_invariant_K(
            xgeo_var,
            time_var,
            pa_local_var,
            computed_vars,
            irbem_input,
        )

    msg = f"Variable '{var_name}' is not implemented in compute_magnetic_field_variables."
    raise NotImplementedError(msg)

@timed_function("Equatorial pitch angle calculation")
def _get_pa_eq(xgeo_var: Variable,
               time_var: Variable,
               pa_local_var: Variable,
               computed_vars: dict[str, Variable],
               irbem_input: proc.IrbemInput) -> dict[str, Variable]:

    logger.info("\tCalculating equatorial pitch angle ...")

    pa_local = pa_local_var.get_data(u.radian)

    if ("B_eq_" + irbem_input.magnetic_field_str) not in computed_vars:
        computed_vars |= proc.get_magequator(xgeo_var, time_var, irbem_input)
    if ("B_local_" + irbem_input.magnetic_field_str) not in computed_vars:
        computed_vars |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    B_eq = computed_vars["B_eq_" + irbem_input.magnetic_field_str].get_data(u.nT)  # noqa: N806
    B_local = computed_vars["B_local_" + irbem_input.magnetic_field_str].get_data(u.nT)  # noqa: N806

    pa_eq_rad = np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:, np.newaxis])

    pa_var = Variable(data=pa_eq_rad, original_unit=u.radian)
    pa_var.metadata.add_processing_note("Computed equatorial pitch angle from local pitch angle and B_eq/B_local ratio "
                                    f"using {irbem_input.magnetic_field_str} and options: {irbem_input.irbem_options}.")

    computed_vars["PA_eq_" + irbem_input.magnetic_field_str] = pa_var

    return computed_vars


@timed_function("Invariant mu calculation")
def _get_invariant_mu(xgeo_var:Variable,
                      time_var:Variable,
                      pa_local_var:Variable,
                      energy_var:Variable,
                      computed_vars:dict[str, Variable],
                      irbem_input: proc.IrbemInput,
                      particle_species: Literal["electron", "proton"]) -> dict[str, Variable]:
    logger.info("\tCalculating invariant mu ...")

    if ("B_local_" + irbem_input.magnetic_field_str) not in computed_vars:
        computed_vars |= proc.get_local_B_field(xgeo_var, time_var, irbem_input)

    # load needed data and convert to correct units
    B_local = computed_vars["B_local_" + irbem_input.magnetic_field_str]  # noqa: N806

    mu_var = proc.compute_invariant_mu(energy_var,
                                       pa_local_var,
                                       B_local,
                                       particle_species)

    computed_vars["invMu_" + irbem_input.magnetic_field_str] = mu_var

    return computed_vars


@timed_function("Invariant K calculation")
def _get_invariant_K(xgeo_var: Variable,  # noqa: N802
                     time_var: Variable,
                     pa_local_var: Variable,
                     computed_vars: dict[str, Variable],
                     irbem_input: proc.IrbemInput) -> dict[str, Variable]:
    logger.info("\tCalculating invariant K ...")

    if ("XJ_" + irbem_input.magnetic_field_str) not in computed_vars:
        computed_vars |= proc.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)
    if ("B_mirr_" + irbem_input.magnetic_field_str) not in computed_vars:
        computed_vars |= proc.get_mirror_point(xgeo_var, time_var, pa_local_var, irbem_input)

    # load needed data and convert to correct units
    B_mirr = computed_vars["B_mirr_" + irbem_input.magnetic_field_str]  # noqa: N806
    XJ = computed_vars["XJ_" + irbem_input.magnetic_field_str]  # noqa: N806

    computed_vars["invK_" + irbem_input.magnetic_field_str] = proc.compute_invariant_K(B_mirr, XJ)

    return computed_vars
