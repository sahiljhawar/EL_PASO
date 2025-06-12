import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from el_paso.physics import ParticleLiteral, en2pc, rest_energy


def compute_invariant_mu(energy:NDArray[np.float64],
                         alpha:NDArray[np.float64],
                         magnetic_field:NDArray[np.float64],
                         particle_species:ParticleLiteral) -> tuple[NDArray[np.float64], u.UnitBase]:
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
    mc2 = rest_energy(particle_species)
    pct = en2pc(energy, particle_species)  # Relativistic energy for the particles

    # Calculate InvMu and PSD using broadcasting
    #sin_alpha_eq = np.sin(np.radians(in_alpha_eq))
    sin_alpha_eq = np.sin(alpha)

    InvMu = (pct[:, :, np.newaxis] * sin_alpha_eq[:, np.newaxis, :]) ** 2 / (
            magnetic_field[:, np.newaxis, np.newaxis] * 2 * mc2)  # MeV/G

    InvMu[InvMu <= 0] = np.nan
    return InvMu, u.MeV/u.G
