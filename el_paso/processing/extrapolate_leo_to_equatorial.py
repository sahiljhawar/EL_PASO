from __future__ import annotations

import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import fmin
from scipy.special import iv
from tqdm import tqdm

from el_paso import Variable


def steady_state_inside_lc(alpha:float|NDArray[np.float64], alpha_lc:float, z0:float, N:float) -> float|NDArray[np.float64]:
    return N * z0 * iv(0, z0 * alpha / alpha_lc) / iv(1, z0)

def steady_state_outside_lc(alpha:float|NDArray[np.float64], alpha_lc:float, z0:float, N:float) -> float|NDArray[np.float64]:
    return steady_state_inside_lc(alpha_lc, alpha_lc, z0, N) + N * np.log(np.sin(alpha) / np.sin(alpha_lc))

def function_to_solve(z0:float, alpha_t0:float, alpha_t30:float, alpha_lc:float, flux_ratio:float) -> float:
    value = iv(0, z0 * alpha_t0 / alpha_lc) / iv(0, z0 * alpha_t30 / alpha_lc)
    return np.abs(flux_ratio - value)

def extrapolate_leo_to_equatorial(pa_eq_obs:Variable,
                                  flux_obs:Variable,
                                  pa_eq_extrap:Variable,
                                  flux_extrap:Variable,
                                  B_fofl:Variable,
                                  B_loc:Variable,
                                  B_eq:Variable):

    UPPER_TRHESHOLD_SANITY = 1e10

    # calculate loss cone
    pa_eq_obs_rad = np.deg2rad(pa_eq_obs.data)
    pa_eq_extrap_rad = np.deg2rad(pa_eq_extrap.data)

    for it in tqdm(range(pa_eq_obs_rad.shape[0])):

        alpha_lc = np.asin(np.sqrt(B_eq.data[it] / B_fofl.data[it]))

        if pa_eq_obs_rad[it,0] > alpha_lc or pa_eq_obs_rad[it,1] > alpha_lc:
            continue

        if pa_eq_obs_rad[it,0] > pa_eq_obs_rad[it,1]:
            continue

        for ie in range(flux_obs.data.shape[1]):

            flux_t0 = flux_obs.data[it,ie,0]
            flux_t30 = flux_obs.data[it,ie,1]

            flux_ratio = flux_t0 / flux_t30

            if flux_ratio > 1 or np.isnan(flux_ratio):
                continue

            z0 = fmin(function_to_solve, 1,
                      args=(pa_eq_obs_rad[it,0], pa_eq_obs_rad[it,1], alpha_lc, flux_ratio),
                      full_output=False, xtol=1e-4, ftol=1e-4, disp=False)[0]

            flux_t0_N1 = steady_state_inside_lc(pa_eq_obs_rad[it,0], alpha_lc, z0, 1)

            N = flux_t0 / flux_t0_N1

            idx_inside_lc = pa_eq_extrap_rad[it,:] <= alpha_lc

            flux_extrap.data[it, ie, idx_inside_lc] = steady_state_inside_lc(pa_eq_extrap_rad[it,idx_inside_lc], alpha_lc, z0, N)
            flux_extrap.data[it, ie, ~idx_inside_lc] = steady_state_outside_lc(pa_eq_extrap_rad[it,~idx_inside_lc], alpha_lc, z0, N)

            if not np.all(np.isfinite(flux_extrap.data[it,ie,:])) or \
               np.any(flux_extrap.data[it,ie,:] > UPPER_TRHESHOLD_SANITY):
                flux_extrap.data[it,ie,:] = np.nan # something went wrong

            # if np.any(flux_extrap.data[it,ie,:] > 1e8):

            #     ic(flux_t0)
            #     ic(flux_t30)
            #     ic(np.rad2deg(pa_eq_obs_rad[it,:]))
            #     ic(np.rad2deg(alpha_lc))

            #     alpha_eq = np.linspace(0.001, alpha_lc, 30)
            #     alpha_eq_outside = np.linspace(alpha_lc, np.pi/2, 30)

            #     flux = steady_state_inside_lc(alpha_eq, alpha_lc, z0, N)
            #     flux_outside = steady_state_outside_lc(alpha_eq_outside, alpha_lc, z0, N)

            #     ic(flux)
            #     ic(flux_outside)

            #     plt.plot(np.rad2deg(alpha_eq), np.log10(flux), "r")
            #     plt.plot(np.rad2deg(alpha_eq_outside), np.log10(flux_outside), "r")
            #     plt.scatter(np.rad2deg(pa_eq_obs_rad[it,0]), np.log10(flux_t0), 10, c="b", marker="x")
            #     plt.scatter(np.rad2deg(pa_eq_obs_rad[it,1]), np.log10(flux_t30), 10, c="b", marker="x")

            #     plt.savefig("test.png")
