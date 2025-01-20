from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from el_paso.utils import get_variable_by_standard_name, validate_standard

if TYPE_CHECKING:
    from el_paso.classes import Variable


@validate_standard
def construct_pitch_angle_distribution(
        input_variables:dict[Variable],
        omni_flux_key:str,
        pa_eq_key:str,
        pa_local_key:str|None=None,
        method:str="sin",
    ) -> np.ndarray:  # noqa: N803
    """Calculate the relativistic energy for a given kinetic energy.

    Args:
        in_flux (np.ndarray or float): The kinetic energy in MeV.
        in_bloc (np.ndarray): Magnitude of local B field.
        in_beq (np.ndarray): Magnitude of equatorial B field.
        in_pa (np.ndarray): Local pitch angle array.

    Returns:
        Flux(np.ndarray or float): The calculated flux .

    Raises:
        ValueError: If an unknown species is provided.

    """
    omni_flux_var = input_variables[omni_flux_key]
    pa_local_var  = input_variables[pa_local_key] if pa_local_key else get_variable_by_standard_name("PA_local", input_variables)
    pa_eq_var     = input_variables[pa_eq_key]

    omni_flux = omni_flux_var.data
    pa_local = pa_local_var.data

    pa_eq_max = np.max(pa_eq_var.data, axis=1)

    match method:
        case "sin":

            # Create a sin distribution
            # Calculate the factor outside the loops
            sin_PA = np.sin(np.deg2rad(pa_local))
            mean_sin = np.mean(np.sin(np.linspace(0, np.pi, 36)))

            # Calculate the factor matrix
            fact_matrix = sin_PA / (mean_sin * np.reciprocal(np.sin(np.deg2rad(pa_eq_max[:, np.newaxis]))))

            differential_flux = omni_flux[:, :, np.newaxis] * fact_matrix[:, np.newaxis, :]

        case _:
            msg = f"Encountered invalid method to constrouct pitch angle distribution: {method}!"
            raise ValueError(msg)

    return differential_flux
