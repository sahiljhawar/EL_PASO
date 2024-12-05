import numpy as np
from astropy import units as u

from matplotlib import pyplot as plt

from el_paso.utils import timed_function
from el_paso.classes import Variable
from el_paso.standardization import get_standard_variable_by_type, get_standard_variable_by_name, validate_standard

def _fold_pitch_angles_and_flux(pitch_angles, flux, produce_statistic_plot):
    # Normalize pitch angles to the range [0, 180]
    pitch_angles_abs = np.abs(pitch_angles)

    # Fold around 90 degrees
    folded_pitch_angles = np.minimum(pitch_angles_abs, 180 - pitch_angles_abs)
    folded_pitch_angles = np.round(folded_pitch_angles, 1)

    # Create an array to hold the folded flux values
    nN, nE, _ = flux.shape
    unique_angles = np.unique(folded_pitch_angles[~np.isnan(folded_pitch_angles)])

    folded_flux = np.full((nN, nE, len(unique_angles)), np.nan)

    if produce_statistic_plot:
        fig, axes = plt.subplots(len(unique_angles), nE, figsize=(5*nE, 5*len(unique_angles)))

    for i, angle in enumerate(unique_angles):

        mask = folded_pitch_angles != angle

        mask = np.repeat(mask[:,np.newaxis,:], nE, axis=1)

        masked_flux = np.ma.masked_array(flux, mask=mask)

        if produce_statistic_plot:
            diff = np.log10(masked_flux) - np.log10(np.flip(masked_flux, axis=2))
            for ie in range(nE):

                diff_energy = diff[:,ie,:].flatten()
                mask_diff = diff_energy > 0

                axes[i,ie].hist(diff_energy[mask_diff].compressed(), bins=10)
                axes[i,ie].set_title(f'Energy = {ie}, alpha = {angle}')

        folded_flux[:,:,i] = np.nanmean(masked_flux, axis=2)
    
    # add time dimension
    unique_angles = np.tile(unique_angles.reshape(1, -1), (nN, 1))
    
    if produce_statistic_plot:
        fig.savefig('test.png')

    return folded_flux, unique_angles

@validate_standard
@timed_function()
def fold_pitch_angles_and_flux(input_variables:dict[str,Variable], flux_key:str=None, pa_local_key:str=None, produce_statistic_plot:bool=False):
    """
    Folds the pitch_angles array around 90 degrees and folds the flux array correspondingly,
    combining elements using nanmean.

    Args:
        flux (np.ndarray): A time x energy x pitch angle array of flux values.
        pitch_angles (np.ndarray): A time x pitch angle array of pitch angles (in degrees between -180 and 180).

    Returns:
        folded_flux (np.ndarray): The folded flux array.
        folded_pitch_angles (np.ndarray): The folded pitch angles array.
    """
    print('Folding pitch angles and flux ...')

    flux_var = input_variables[flux_key] if flux_key else get_standard_variable_by_type('Flux', input_variables)
    pa_local_var = input_variables[pa_local_key] if pa_local_key else get_standard_variable_by_name('PA_local', input_variables)

    flux     = flux_var.data
    pa_local = pa_local_var.data

    assert np.all(np.repeat(pa_local[0,:][np.newaxis,:], pa_local.shape[0], axis=0) == pa_local), 'We assume that local pitch angles do not change in time!'

    assert len(flux) == len(pa_local)
    assert flux.shape[2] == pa_local.shape[1], f'Dimension missmatch: flux: {flux.shape}, pitch angle: {pa_local.shape}'

    folded_flux, unique_angles = _fold_pitch_angles_and_flux(pa_local, flux, produce_statistic_plot)

    flux_var.data     = folded_flux
    pa_local_var.data = unique_angles