import numpy as np
from astropy import units as u

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from el_paso.classes import Product, DerivedVariable
from el_paso.physics import en2pc

def compute_PSD(product):

    flux_vars = product.get_variables_by_type('Flux')
    assert len(flux_vars) == 1, f'We assume that there is exactly ONE flux variable available for calculating PSD. Found: {len(flux_vars)}!'
    flux_var = flux_vars[0]

    energy_vars = product.get_variables_by_type('Energy')
    assert len(energy_vars) == 1, f'We assume that there is exactly ONE energy variable available for calculating PSD. Found: {len(energy_vars)}!'
    energy_var = energy_vars[0]

    species_char = flux_var.standard.standard_name[1]

    # Calculate pct for each energy value
    pct = en2pc(energy_var.data_content, species_char)  # Relativistic energy for electrons
    psd_data = (1e3 / 2.997e10) * flux_var.data_content / (pct[:, :, np.newaxis] ** 2)

    return psd_data, (u.m * u.kg * u.m / u.s)**(-3)