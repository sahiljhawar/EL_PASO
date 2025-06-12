from typing import Literal

import numpy as np
from numpy.typing import NDArray

ParticleLiteral = Literal["electron", "proton", "helium", "oxygen"]

def rest_energy(species:ParticleLiteral) -> float:
    """
    Return the rest energy for the input species.
    Args:
        species (str): The species of particle ('electron', 'proton', 'helium', 'oxygen').

    Returns:
        np.ndarray or float: The rest energy of the species in MeV.
    Raises:
        ValueError: If an unknown species is provided.
    """
    # Rest energy in MeV for different species
    rest_energies = {
        "electron": 0.511,  # MeV
        "proton": 938.272,  # MeV
        "helium": 3727.379,  # MeV (Helium-4 nucleus)
        "oxygen": 14958.9  # MeV (Oxygen-16 nucleus)
    }

    # Set mc2 based on the species
    if species.lower() in rest_energies:
        mc2 = rest_energies[species.lower()]
    else:
        msg = f"Unknown species '{species}'. Valid options are 'electron', 'proton', 'helium', 'oxygen'."
        raise ValueError(
            msg)

    return mc2


def en2pc(energy:NDArray[np.float64], species:ParticleLiteral="electron") -> NDArray[np.float64]:
    """Calculate the relativistic energy for a given kinetic energy.

    Args:
        energy (np.ndarray or float): The kinetic energy in MeV.
        species (str): The species of particle ('electron', 'proton', 'helium', 'oxygen').

    Returns:
        np.ndarray or float: The calculated relativistic energy.

    """
    mc2 = rest_energy(species)
    # Calculate the relativistic energy
    return np.sqrt((energy / mc2 + 1) ** 2 - 1) * mc2
