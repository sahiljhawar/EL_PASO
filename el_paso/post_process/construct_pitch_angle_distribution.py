from __future__ import annotations

import numpy as np

from el_paso.classes import Variable
from el_paso.standardization import get_standard_variable_by_name, get_standard_variable_by_type, validate_standard

@validate_standard
def construct_pitch_angle_distribution(
        input_variables:dict,
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
    pa_local_var  = input_variables[pa_local_key] if pa_local_key else get_standard_variable_by_name("PA_local", input_variables)
    pa_eq_var     = input_variables[pa_eq_key]

    omni_flux = omni_flux_var.data
    pa_local = pa_local_var.data

    # Determine the highest equatorial pitch angle that we observe
    # zz = np.where(B_loc < in_beq)
    # if zz[0].size > 0:
    #     B_loc[zz] = B_eq[zz]
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
