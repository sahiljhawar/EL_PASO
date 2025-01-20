import numpy as np
from astropy import units as u

from el_paso.physics import en2pc
from el_paso.utils import get_variable_by_standard_type, validate_standard


@validate_standard
def compute_PSD(input_variables, psd_variable, flux_key:str=None, energy_key:str=None):

    flux_var = input_variables[flux_key] if flux_key else get_variable_by_standard_type('Flux', input_variables)
    energy_var = input_variables[energy_key] if energy_key else get_variable_by_standard_type('Energy', input_variables)

    energy_var.convert_to_unit(u.MeV) 

    species_char = flux_var.standard.standard_name[1]

    # Calculate pct for each energy value
    pct = en2pc(energy_var.data, species_char)  # Relativistic energy for electrons
    psd_data = (1e3 / 2.997e10) * flux_var.data / (pct[:, :, np.newaxis] ** 2)

    psd_variable.data = psd_data
    psd_variable.metadata.unit = (u.m * u.kg * u.m / u.s)**(-3)
    psd_variable.time_variable = flux_var.time_variable