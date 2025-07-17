import logging
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep


def compute_PSD(flux_var:ep.Variable,
                energy_var:ep.Variable,
                particle_species:Literal["electron", "proton"]) -> ep.Variable:

    logger = logging.getLogger(__name__)
    logger.info("Computing PSD...")

    flux_data = flux_var.get_data((u.cm**2 * u.s * u.keV * u.sr) ** (-1))
    energies = energy_var.get_data(u.MeV)

    # Calculate pct for each energy value
    pct = ep.physics.en2pc(energies, particle_species)  # Relativistic energy for electrons
    psd_data = (1e3 / 2.997e10) * flux_data / (pct[:, :, np.newaxis] ** 2)

    var = ep.Variable(data=psd_data, original_unit=(u.m * u.kg * u.m / u.s)**(-3))

    var.metadata.add_processing_note("Created with compute_PSD")

    return var
