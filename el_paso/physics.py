import numpy as np

def rest_energy(species='e'):
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
        'e': 0.511,  # MeV
        'p': 938.272,  # MeV
        'h': 3727.379,  # MeV (Helium-4 nucleus)
        'o': 14958.9  # MeV (Oxygen-16 nucleus)
    }

    # Set mc2 based on the species
    if species.lower() in rest_energies:
        mc2 = rest_energies[species.lower()]
    else:
        raise ValueError(
            f"Unknown species '{species}'. Valid options are 'e', 'p', 'h', 'o'.")

    return mc2


def pfunc(energy, species='e'):
    """
    Calculate the relativistic energy for a given kinetic energy.

    Args:
        energy (np.ndarray or float): The kinetic energy in MeV.
        species (str): The species of particle ('electron', 'proton', 'helium', 'oxygen').

    Returns:
        np.ndarray or float: The calculated relativistic energy.
    """
    mc2 = rest_energy(species)
    # Calculate the relativistic energy
    y = np.sqrt((energy / mc2 + 1) ** 2 - 1) * mc2
    return y
