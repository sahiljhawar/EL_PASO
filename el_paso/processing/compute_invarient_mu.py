import numpy as np
from astropy import units as u

from el_paso.physics import en2pc, rest_energy

def compute_invariant_mu(energy, alpha, magnetic_field, species_char):
    """
    Calculate the relativistic energy for a given kinetic energy.

    Args:
        in_energy (np.ndarray or float): Energy channel definitions.
        in_alpha_eq (np.ndarray or float): Equatorial pitch angles.
        in_b_eq (np.ndarray or float): Equatorial B field magnitude.
        in_species (str): Particle species.

    Returns:
        InvMu(np.ndarray or float): The calculated adiabatic invariant mu.
    """
    mc2 = rest_energy(species_char)
    pct = en2pc(energy, species_char)  # Relativistic energy for the particles

    # Calculate InvMu and PSD using broadcasting
    #sin_alpha_eq = np.sin(np.radians(in_alpha_eq))
    sin_alpha_eq = np.sin(alpha)

    InvMu = (pct[:, :, np.newaxis] * sin_alpha_eq[:, np.newaxis, :]) ** 2 / (
            magnetic_field[:, np.newaxis, np.newaxis] * 2 * mc2)  # MeV/G

    InvMu[InvMu <= 0] = np.nan
    return InvMu, u.MeV/u.G
