"""Parameter-transfer priors: predict a fit parameter from a reference fit.

A lineshape parameter measured cleanly on a reference signal (e.g. the linewidth
of TMS after reference deconvolution) often predicts the same parameter on a
harder signal. Fitting that relationship once lets it be fed back into
:func:`nmr_quant.fit_dataset` as a per-spectrum prior (``fwhm_prior`` /
``fwhm_prior_r2``) to constrain subsequent fits.

This module keeps only the general maths: fit a linear model
``target ~= slope * ref + intercept`` and apply it. The data plumbing (which
parquet columns, how rows are matched, where the model JSON lives) stays with
the caller.
"""

from __future__ import annotations

import numpy as np


def fit_linear_prior(ref, target, *, mask=None) -> dict:
    """Fit ``target ~= slope * ref + intercept`` by least squares.

    Parameters
    ----------
    ref, target : array-like
        Reference and target values, same length (e.g. reference FWHM and the
        target signal's FWHM over the training spectra).
    mask : array-like of bool, optional
        Selects the training rows (e.g. quality-passed fits). Non-finite pairs
        are dropped after masking.

    Returns
    -------
    dict
        JSON-serialisable model: ``slope, intercept, r2_fit, residual_std,
        n_total, n_training``. Apply it with :func:`apply_linear_prior`.
    """
    from scipy.stats import linregress

    ref = np.asarray(ref, dtype=float)
    target = np.asarray(target, dtype=float)
    if ref.shape != target.shape:
        raise ValueError("ref and target must have the same shape")
    n_total = int(ref.size)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        ref, target = ref[m], target[m]
    finite = np.isfinite(ref) & np.isfinite(target)
    ref, target = ref[finite], target[finite]
    if ref.size < 2:
        raise ValueError("need at least 2 finite training points to fit a prior")

    slope, intercept, r_value, _, _ = linregress(ref, target)
    resid_std = float((target - (slope * ref + intercept)).std())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2_fit": float(r_value**2),
        "residual_std": resid_std,
        "n_total": n_total,
        "n_training": int(ref.size),
    }


def apply_linear_prior(model: dict, ref) -> np.ndarray:
    """Predict ``slope * ref + intercept`` elementwise (NaN ref -> NaN)."""
    ref = np.asarray(ref, dtype=float)
    return model["slope"] * ref + model["intercept"]
