"""Lineshape primitives (bridge module).

These are the peak-shape definitions used by the fitter. They are kept behind
this single module on purpose: the agreed design is for the canonical shapes to
live in ``python-nmr-spectra-processing`` (``nmr_spectra_processing.lineshapes``).
Once that module is published, this file becomes a thin re-export:

    from nmr_spectra_processing.lineshapes import (
        voigt_height, voigt_area, area_from_height, height_from_area,
        multiplet, pseudo_voigt_fwhm, measure_fwhm,
    )

Until then the definitions live here so ``nmr-quant`` is buildable on its own.
Do not scatter Voigt maths elsewhere in the package; import from here.
"""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)
_MIN_SIGMA = 1e-12


def _norm_wofz(sigma: float, gamma: float) -> float:
    """Re(w(i gamma / (sigma sqrt2))) — the peak value of the unscaled Voigt."""
    return float(np.real(wofz(1j * gamma / (max(sigma, _MIN_SIGMA) * _SQRT2))))


def voigt_profile(x, center, sigma, gamma):
    """Voigt profile normalised to unit peak height."""
    sigma = max(float(sigma), _MIN_SIGMA)
    z = ((x - center) + 1j * gamma) / (sigma * _SQRT2)
    return np.real(wofz(z)) / _norm_wofz(sigma, gamma)


def voigt_height(x, height, center, sigma, gamma):
    """Voigt profile scaled so its peak equals ``height``."""
    return height * voigt_profile(x, center, sigma, gamma)


def area_from_height(height: float, sigma: float, gamma: float) -> float:
    """Analytic integral of a height-parameterised Voigt.

    area = height * sigma * sqrt(2 pi) / Re(w(i gamma / (sigma sqrt2))).
    """
    sigma = max(float(sigma), _MIN_SIGMA)
    return float(height) * sigma * _SQRT2PI / _norm_wofz(sigma, gamma)


def height_from_area(area: float, sigma: float, gamma: float) -> float:
    """Inverse of :func:`area_from_height`."""
    sigma = max(float(sigma), _MIN_SIGMA)
    return float(area) * _norm_wofz(sigma, gamma) / (sigma * _SQRT2PI)


def voigt_area(x, area, center, sigma, gamma):
    """Voigt profile normalised so its integral equals ``area``."""
    return voigt_height(x, height_from_area(area, sigma, gamma), center, sigma, gamma)


def multiplet(x, height, center, sigma, gamma, offsets, heights_rel):
    """Sum of Voigt lines sharing one lineshape, at annotation-fixed offsets.

    ``height`` scales the tallest line (``heights_rel`` is normalised so its
    max is 1); each line sits at ``center + offset`` with the shared
    ``sigma``/``gamma``.
    """
    y = np.zeros_like(np.asarray(x, dtype=float))
    for offset, h_rel in zip(offsets, heights_rel, strict=True):
        y = y + voigt_height(x, height * h_rel, center + offset, sigma, gamma)
    return y


def voigt_with_satellites(
    x, height, center, sigma, gamma, f_sat, delta_sat, sigma_sat, gamma_sat
):
    """Main Voigt plus two symmetric satellites (the TMS-style model).

    Satellites are added only when ``f_sat > 0``; each has height
    ``f_sat * height`` and sits at ``center +/- delta_sat``.
    """
    y = voigt_height(x, height, center, sigma, gamma)
    if f_sat and f_sat > 0:
        h_sat = f_sat * height
        y = y + voigt_height(x, h_sat, center - delta_sat, sigma_sat, gamma_sat)
        y = y + voigt_height(x, h_sat, center + delta_sat, sigma_sat, gamma_sat)
    return y


def voigt_multi_satellites(x, height, center, sigma, gamma, deltas, fracs,
                           sigmas=None, gammas=None):
    """Main Voigt plus N symmetric satellite pairs.

    Each pair ``k`` adds two Voigt lines of height ``fracs[k] * height`` at
    ``center +/- deltas[k]``. Satellite widths default to the main
    ``sigma``/``gamma``; pass ``sigmas``/``gammas`` (one per pair) to give a pair
    its own width (e.g. TMS ``13C`` satellites are slightly broader than the
    main line). With one tied-width pair this equals
    :func:`voigt_with_satellites`.
    """
    y = voigt_height(x, height, center, sigma, gamma)
    for k, (d, f) in enumerate(zip(deltas, fracs, strict=True)):
        if f and f > 0:
            s = sigmas[k] if sigmas is not None else sigma
            g = gammas[k] if gammas is not None else gamma
            hs = f * height
            y = y + voigt_height(x, hs, center - d, s, g)
            y = y + voigt_height(x, hs, center + d, s, g)
    return y


def pseudo_voigt_fwhm(sigma: float, gamma: float) -> float:
    """Full width at half maximum of a Voigt via the pseudo-Voigt approximation.

    Uses the Olivero-Longbothum combination of the Gaussian FWHM (2.355 sigma)
    and the Lorentzian FWHM (2 gamma).
    """
    f_l = 2.0 * float(gamma)
    f_g = 2.3548200450309493 * float(sigma)
    return 0.5346 * f_l + np.sqrt(0.2166 * f_l * f_l + f_g * f_g)


def measure_fwhm(x, y, x_center: float) -> float:
    """Numeric FWHM of a peak near ``x_center`` by linear interpolation.

    Assumes a uniform ``x`` axis. Returns NaN if either half-max crossing
    cannot be located.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x[1] - x[0]
    idx = int(np.argmin(np.abs(x - x_center)))
    half = y[idx] / 2.0
    left = right = None
    for i in range(idx, 0, -1):
        if y[i - 1] <= half:
            frac = (y[i] - half) / max(y[i] - y[i - 1], 1e-30)
            left = (i - frac) * dx + x[0]
            break
    for i in range(idx, len(x) - 1):
        if y[i + 1] <= half:
            frac = (y[i] - half) / max(y[i] - y[i + 1], 1e-30)
            right = (i + frac) * dx + x[0]
            break
    if left is None or right is None:
        return float("nan")
    return abs(right - left)
