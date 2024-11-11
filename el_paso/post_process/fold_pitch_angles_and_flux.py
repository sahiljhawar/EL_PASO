import numpy as np

def fold_pitch_angles_and_flux(product):
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

    flux_vars = product.get_variables_by_type('Flux')
    assert len(flux_vars) == 1, f'We assume that there is exactly ONE flux variable available for calculating PSD. Found: {len(flux_vars)}!'
    flux = flux_vars[0].data_content

    assert product.get_standard_variable('PA_local').metadata.unit == ''
    pitch_angles = product.get_standard_variable('PA_local').data_content
    pitch_angles = np.rad2deg(pitch_angles)

    assert len(flux) == len(pitch_angles)
    assert flux.shape[2] == pitch_angles.shape[1]

    # Normalize pitch angles to the range [0, 180]
    pitch_angles_abs = np.abs(pitch_angles)

    # Fold around 90 degrees
    folded_pitch_angles = np.minimum(pitch_angles_abs, 180 - pitch_angles_abs)
    folded_pitch_angles = np.round(folded_pitch_angles, 1)

    # Create an array to hold the folded flux values
    N, E, P = flux.shape
    unique_angles = np.unique(folded_pitch_angles[~np.isnan(folded_pitch_angles)])

    folded_flux = np.full((N, E, len(unique_angles)), np.nan)

    for i, angle in enumerate(unique_angles):
        mask = (folded_pitch_angles == angle)

        for n in range(N):
            for e in range(E):
                if np.any(mask[n]) and flux[n, e, mask[n]].size > 0:
                    # Apply nanmean to the corresponding elements in the flux array
                    folded_flux[n, e, i] = np.nanmean(flux[n, e, mask[n]])
                else:
                    # Assign np.nan if the array is empty or all values are masked
                    folded_flux[n, e, i] = np.nan

    flux_vars[0].data_content = folded_flux
    product.get_standard_variable('PA_local').data_content = np.tile(unique_angles.reshape(1, -1), (N, 1))
