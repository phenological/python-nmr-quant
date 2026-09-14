"""Standard goodness-of-fit and residual diagnostics for a single fit.

All metrics are residual-based and optimizer-agnostic, so they work with the
``scipy.optimize.minimize`` fitting core and with any other. The noise floor is
estimated from the residual **edges** (where the signal model is ~0), matching
the definition used by fovea today, so ``srr`` and the noise-adjusted ``r2_adj``
stay compatible with existing results. A caller-supplied ``noise`` overrides the
floor for ``chi2_red`` and ``snr`` only.
"""

from __future__ import annotations

import numpy as np


def _edge_noise_std(residual: np.ndarray, n_noise: int | None) -> float:
    n = residual.size
    k = n_noise if n_noise is not None else max(3, n // 10)
    k = max(1, min(k, n // 2 if n >= 2 else 1))
    edge = np.concatenate([residual[:k], residual[-k:]])
    return float(edge.std())


def fit_metrics(
    y,
    y_model,
    *,
    height: float,
    n_params: int | None = None,
    noise: float | None = None,
    n_noise: int | None = None,
) -> dict:
    """Compute fit metrics from observed ``y`` and modelled ``y_model``.

    Parameters
    ----------
    y, y_model : array-like
        Observed and modelled intensities over the fitted region.
    height : float
        Fitted peak height, used for ``srr`` and ``snr``.
    n_params : int, optional
        Number of free parameters; enables ``aic``, ``bic`` and sets the
        degrees of freedom for ``chi2_red``.
    noise : float, optional
        Known noise sigma. Overrides the edge-noise floor for ``chi2_red`` and
        ``snr`` (``srr`` and ``r2_adj`` always use the edge floor).
    n_noise : int, optional
        Points per edge for the noise estimate. Default ``max(3, n // 10)``.

    Returns
    -------
    dict
        ``r2, r2_adj, srr, sse, rmse, mae, residual_std, noise_std, aic, bic,
        dw, chi2_red, snr``.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    y_model = np.ascontiguousarray(y_model, dtype=np.float64)
    n = y.size
    resid = y - y_model

    sse = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")

    noise_std = _edge_noise_std(resid, n_noise)
    srr = height / noise_std if noise_std > 0 else float("nan")

    ss_noise = noise_std**2 * n
    denom_adj = ss_tot - ss_noise
    r2_adj = (
        float(np.clip((ss_tot - sse) / denom_adj, 0.0, 1.0))
        if denom_adj > 0
        else float("nan")
    )

    rmse = float(np.sqrt(sse / n)) if n > 0 else float("nan")
    mae = float(np.abs(resid).mean()) if n > 0 else float("nan")
    residual_std = float(resid.std())

    if n_params is not None and sse > 0 and n > 0:
        aic = n * np.log(sse / n) + 2 * n_params
        bic = n * np.log(sse / n) + n_params * np.log(n)
        aic, bic = float(aic), float(bic)
    else:
        aic = bic = float("nan")

    dw = float((np.diff(resid) ** 2).sum() / sse) if sse > 0 else float("nan")

    noise_used = float(noise) if noise is not None else noise_std
    dof = max(n - (n_params or 0), 1)
    chi2_red = sse / (noise_used**2 * dof) if noise_used > 0 else float("nan")
    snr = height / noise_used if noise_used > 0 else float("nan")

    return {
        "r2": r2,
        "r2_adj": r2_adj,
        "srr": float(srr),
        "sse": sse,
        "rmse": rmse,
        "mae": mae,
        "residual_std": residual_std,
        "noise_std": noise_std,
        "aic": aic,
        "bic": bic,
        "dw": dw,
        "chi2_red": float(chi2_red),
        "snr": float(snr),
    }
