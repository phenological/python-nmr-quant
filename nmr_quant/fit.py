"""Least-squares fitting of NMR peak lineshapes to 1D spectra.

Ported from the fovea fitting core, decoupled from any data source: the public
functions take plain arrays plus an explicit signal description (multiplet
offsets, relative heights, atom count) and return a plain record / table.

Two entry points:

* :func:`fit_spectrum` fits a single spectrum, deriving its own warm start.
* :func:`fit_dataset` fits a 2-D batch: it warm-starts once on the mean
  spectrum, then fits every row with optional per-spectrum FWHM/centre priors.

The optimiser is ``scipy.optimize.minimize`` (L-BFGS-B) on a scalar loss
(sum-of-squares or Huber), matching the reference implementation. A polynomial
baseline can be fitted simultaneously, and TMS-style satellites detected and
fitted on the residual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import _shapes
from .metrics import fit_metrics

_FWHM_G = 2.3548200450309493  # 2 sqrt(2 ln 2)


# ── result record ─────────────────────────────────────────────────────────────
@dataclass
class FitRecord:
    """One fitted spectrum: shape/quantity fields plus §metrics diagnostics."""

    height: float
    center: float
    sigma: float
    gamma: float
    f_sat: float
    delta_sat: float
    sigma_sat: float
    gamma_sat: float
    has_satellites: bool
    area_main: float
    area_total: float
    fwhm: float
    baseline_coeffs: list = field(default_factory=list)
    # metrics (filled from metrics.fit_metrics)
    r2: float = float("nan")
    r2_adj: float = float("nan")
    srr: float = float("nan")
    sse: float = float("nan")
    rmse: float = float("nan")
    mae: float = float("nan")
    residual_std: float = float("nan")
    noise_std: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    dw: float = float("nan")
    chi2_red: float = float("nan")
    snr: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


# ── baseline helpers ────────────────────────────────────────────────────────
def _x_norm(x: np.ndarray) -> np.ndarray:
    """x mapped to [-1, 1] over the ROI (for polynomial numerical stability)."""
    return 2.0 * (x - 0.5 * (x[0] + x[-1])) / (x[-1] - x[0])


def _bl_eval(coeffs, xn: np.ndarray) -> np.ndarray:
    if coeffs is None or len(coeffs) == 0:
        return np.zeros_like(xn)
    return np.polyval(coeffs, xn)


def _bl_bounds(deg: int, h_scale: float) -> list:
    lim = abs(h_scale) * 2.0
    return [(-lim, lim)] * (deg + 1) if deg >= 0 else []


def _prefit_baseline(x, y, xn, deg, region) -> list:
    """Warm-start baseline coefficients from the edge points inside ``region``."""
    if region is None or deg < 0:
        return []
    lo, hi = min(region), max(region)
    m = np.where((x >= lo) & (x <= hi))[0]
    if len(m) < 2 * (deg + 1):
        return []
    n_edge = max(deg + 1, len(m) // 8)
    idx = np.concatenate([m[:n_edge], m[-n_edge:]])
    return list(np.polyfit(xn[idx], y[idx], deg))


def _bl_init(x, y, xn, deg, region):
    """Return (baseline_init_coeffs, H0_eff, bounds) for a given degree."""
    ws = _prefit_baseline(x, y, xn, deg, region)
    if ws:
        bl_init = ws
    else:
        n_edge = max(2, len(y) // 10)
        bl_est = float(np.mean(np.concatenate([y[:n_edge], y[-n_edge:]])))
        bl_init = [0.0] * deg + [bl_est]
    h0_eff = max(float(y.max()) - float(np.polyval(bl_init, 0.0)), float(y.max()) * 0.1)
    return bl_init, h0_eff, _bl_bounds(deg, float(y.max()))


# ── loss + residuals ──────────────────────────────────────────────────────────
def _loss(r: np.ndarray, delta) -> float:
    if delta is None:
        return float(r @ r)
    a = np.abs(r)
    return float(np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)).sum())


def _res_main(p, x, y, xn, deg, delta):
    if deg < 0:
        return _loss(y - _shapes.voigt_height(x, *p), delta)
    ym = _shapes.voigt_height(x, *p[:4]) + _bl_eval(p[4:], xn)
    return _loss(y - ym, delta)


def _res_mp(p, x, y, xn, deg, delta, offsets, heights_rel):
    m = _shapes.multiplet(x, p[0], p[1], p[2], p[3], offsets, heights_rel)
    if deg < 0:
        return _loss(y - m, delta)
    return _loss(y - (m + _bl_eval(p[4:], xn)), delta)


def _res_full(p, x, y):
    return float(np.sum((y - _shapes.voigt_with_satellites(x, *p)) ** 2))


def _res_fixed_delta(p5, x, y, delta):
    h, c, sg, gm, fs = p5
    model = _shapes.voigt_with_satellites(x, h, c, sg, gm, fs, delta, sg, gm)
    return float(np.sum((y - model) ** 2))


# ── warm start (mean or single spectrum) ──────────────────────────────────────
def _warm_start(x, y, xn, *, offsets, heights_rel, is_multiplet, center,
                center_window, baseline_deg, baseline_region, fit_satellites,
                satellite_threshold, ppm_step, ppm_span):
    c_half = center_window if center_window is not None else ppm_span * 0.1
    h0 = float(y.max())
    c0 = float(x[np.argmax(y)]) if center is None else float(center)

    above = np.where(y >= h0 / 2)[0]
    fwhm_est = abs(x[above[-1]] - x[above[0]]) if len(above) >= 2 else ppm_span / 10
    sigma0 = fwhm_est / _FWHM_G
    gamma0 = sigma0 / 2
    if is_multiplet and len(offsets) > 1:
        sigma0 = min(sigma0, abs(offsets[-1] - offsets[0]) / 4)
        gamma0 = sigma0 / 2

    deg = baseline_deg if baseline_deg is not None else -1
    if deg >= 0:
        bl_init, h0_eff, bl_bounds = _bl_init(x, y, xn, deg, baseline_region)
    else:
        bl_init, h0_eff, bl_bounds = [], h0, []

    bounds = [
        (h0_eff * 0.5, h0_eff * 2.0),
        (c0 - c_half, c0 + c_half),
        (ppm_step, ppm_span / 4),
        (ppm_step / 2, ppm_span / 4),
    ] + bl_bounds
    p0 = [h0_eff, c0, sigma0, gamma0] + bl_init

    if is_multiplet:
        res = minimize(_res_mp, p0, args=(x, y, xn, deg, None, offsets, heights_rel),
                       method="L-BFGS-B", bounds=bounds)
        h, c, sg, gm = res.x[:4]
        return {"warm": [h, c, sg, gm, 0.0, 0.0, sg, gm], "fit_sat": False,
                "baseline_coeffs": list(res.x[4:]) if deg >= 0 else []}

    res = minimize(_res_main, p0, args=(x, y, xn, deg, None),
                   method="L-BFGS-B", bounds=bounds)
    h, c, sg, gm = res.x[:4]
    bl = list(res.x[4:]) if deg >= 0 else []
    y_main = _shapes.voigt_height(x, h, c, sg, gm)
    resid = (y - _bl_eval(bl, xn)) - y_main
    fit_sat = fit_satellites and (resid.max() / h > satellite_threshold)
    if not fit_sat:
        return {"warm": [h, c, sg, gm, 0.0, 0.0, sg, gm], "fit_sat": False,
                "baseline_coeffs": bl}

    fwhm_fit = _shapes.pseudo_voigt_fwhm(sg, gm)
    res_outer = resid.copy()
    res_outer[np.abs(x - c) < 3 * fwhm_fit] = 0
    sidx = int(np.argmax(res_outer))
    delta0 = abs(x[sidx] - c)
    f0 = min(max(resid[sidx] / h, 0.001), 1.0)
    fmax = min(1.0, max(0.05, f0 * 1.5))
    dmin, dmax = max(ppm_step * 4, 2 * fwhm_fit), ppm_span / 3
    if dmin >= dmax:
        dmin = ppm_step * 4
    bounds_full = [
        (h * 0.8, h * 1.2), (c - ppm_step * 5, c + ppm_step * 5),
        (ppm_step, sg * 3), (ppm_step / 2, gm * 3),
        (0.001, fmax), (dmin, dmax), (ppm_step, sg * 5), (ppm_step / 2, gm * 5),
    ]
    res_f = minimize(_res_full, [h, c, sg, gm, f0, delta0, sg, gm],
                     args=(x, y), method="L-BFGS-B", bounds=bounds_full)
    return {"warm": list(res_f.x), "fit_sat": True, "baseline_coeffs": bl}


def _prior_widths(sigma_f, gamma_f, fwhm_mean, prior, prior_r2, factor,
                  default_sigma_hi, default_gamma_hi, ppm_step):
    """Per-spectrum (sigma0, gamma0, sigma_hi, gamma_hi) from an optional prior."""
    if prior is not None and np.isfinite(prior) and prior > 0:
        ratio = prior / max(fwhm_mean, 1e-9)
        sp, gp = sigma_f * ratio, gamma_f * ratio
        conf = float(np.clip(prior_r2, 0.0, 1.0)) if (prior_r2 is not None
                                                      and np.isfinite(prior_r2)) else 1.0
        sigma0 = conf * sp + (1 - conf) * sigma_f
        gamma0 = conf * gp + (1 - conf) * gamma_f
        sigma_hi = max(conf * sp * factor + (1 - conf) * default_sigma_hi, ppm_step * 2)
        gamma_hi = max(conf * gp * factor + (1 - conf) * default_gamma_hi, ppm_step)
        return sigma0, gamma0, sigma_hi, gamma_hi
    return sigma_f, gamma_f, default_sigma_hi, default_gamma_hi


# ── per-spectrum fit ──────────────────────────────────────────────────────────
def _fit_one(x, y, xn, *, warm, offsets, heights_rel, is_multiplet, center_i,
             c_half, baseline_deg, baseline_region, fit_sat, satellite_threshold,
             loss, huber_delta_factor, sigma0, gamma0, sigma_hi, gamma_hi,
             ppm_step, noise) -> tuple:
    h0 = float(y.max())
    deg = baseline_deg if baseline_deg is not None else -1
    if deg >= 0:
        bl_init, h0_eff, bl_bounds = _bl_init(x, y, xn, deg, baseline_region)
    else:
        bl_init, h0_eff, bl_bounds = [], h0, []
    delta = h0_eff * huber_delta_factor if loss == "huber" else None
    delta_f = warm[5]

    common_bounds = [
        (h0_eff * 0.3, h0_eff * 3.0),
        (center_i - c_half, center_i + c_half),
        (ppm_step, sigma_hi),
        (ppm_step / 2, gamma_hi),
    ] + bl_bounds
    p0 = [h0_eff, center_i, sigma0, gamma0] + bl_init

    if is_multiplet:
        res = minimize(_res_mp, p0, args=(x, y, xn, deg, delta, offsets, heights_rel),
                       method="L-BFGS-B", bounds=common_bounds)
        h, c, sg, gm = res.x[:4]
        bl = list(res.x[4:]) if deg >= 0 else []
        fp = [h, c, sg, gm, 0.0, 0.0, sg, gm]
        y_fit = _shapes.multiplet(x, h, c, sg, gm, offsets, heights_rel) + _bl_eval(bl, xn)
        has_sat = False
        area_main = _shapes.area_from_height(h, sg, gm)
        area_total = area_main * sum(heights_rel)
    else:
        res = minimize(_res_main, p0, args=(x, y, xn, deg, delta),
                       method="L-BFGS-B", bounds=common_bounds)
        h, c, sg, gm = res.x[:4]
        bl = list(res.x[4:]) if deg >= 0 else []
        y_corr = (y - _bl_eval(bl, xn)) if bl else y
        resid = y_corr - _shapes.voigt_height(x, h, c, sg, gm)
        if fit_sat and resid.max() / h > satellite_threshold:
            f0 = min(max(resid.max() / h, 0.001), 1.0)
            fmax = min(1.0, max(0.05, f0 * 1.5))
            bounds_s = [(h * 0.5, h * 2.0), (c - ppm_step * 5, c + ppm_step * 5),
                        (ppm_step, sg * 5), (ppm_step / 2, gm * 5), (0.001, fmax)]
            res_s = minimize(_res_fixed_delta, [h, c, sg, gm, f0],
                             args=(x, y_corr, delta_f), method="L-BFGS-B", bounds=bounds_s)
            h, c, sg, gm, fs = res_s.x
            fp = [h, c, sg, gm, fs, delta_f, sg, gm]
            y_fit = _shapes.voigt_with_satellites(x, *fp)
            has_sat = True
        else:
            fp = [h, c, sg, gm, 0.0, 0.0, sg, gm]
            y_fit = _shapes.voigt_height(x, h, c, sg, gm)
            has_sat = False
        y_fit = y_fit + _bl_eval(bl, xn)
        area_main = _shapes.area_from_height(fp[0], fp[2], fp[3])
        area_total = area_main * (1 + 2 * fp[4]) if has_sat else area_main

    fwhm = _shapes.pseudo_voigt_fwhm(fp[2], fp[3])
    n_params = 4 + (deg + 1 if deg >= 0 else 0) + (1 if has_sat else 0)
    metrics = fit_metrics(y, y_fit, height=fp[0], n_params=n_params, noise=noise)

    rec = FitRecord(
        height=fp[0], center=fp[1], sigma=fp[2], gamma=fp[3],
        f_sat=fp[4], delta_sat=fp[5], sigma_sat=fp[6], gamma_sat=fp[7],
        has_satellites=has_sat, area_main=area_main, area_total=area_total,
        fwhm=fwhm, baseline_coeffs=bl, **metrics,
    )
    return rec, y_fit


# ── public API ────────────────────────────────────────────────────────────────
def fit_spectrum(x, y, *, offsets=(0.0,), heights_rel=(1.0,), center=None,
                 center_window=None, baseline_deg=None, baseline_region=None,
                 fit_satellites=False, satellite_threshold=0.005,
                 loss="ls", huber_delta_factor=0.5, noise=None) -> FitRecord:
    """Fit a single spectrum ``y`` over axis ``x`` and return a :class:`FitRecord`.

    ``offsets`` / ``heights_rel`` describe the (multiplet) line pattern; a
    singlet is the default ``((0.0,), (1.0,))``. The warm start is derived from
    ``y`` itself. See the module docstring for the model and options.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    xn = _x_norm(x)
    ppm_step = abs(x[1] - x[0])
    ppm_span = abs(x[-1] - x[0])
    is_multiplet = len(offsets) > 1
    c_half = center_window if center_window is not None else ppm_span * 0.1

    warm = _warm_start(x, y, xn, offsets=offsets, heights_rel=heights_rel,
                       is_multiplet=is_multiplet, center=center,
                       center_window=center_window, baseline_deg=baseline_deg,
                       baseline_region=baseline_region, fit_satellites=fit_satellites,
                       satellite_threshold=satellite_threshold,
                       ppm_step=ppm_step, ppm_span=ppm_span)
    sigma_f, gamma_f = warm["warm"][2], warm["warm"][3]
    default_hi = (sigma_f * 5, gamma_f * 5) if is_multiplet else (ppm_span / 4, ppm_span / 4)
    center_i = warm["warm"][1] if center is None else float(center)

    rec, _ = _fit_one(x, y, xn, warm=warm["warm"], offsets=offsets,
                      heights_rel=heights_rel, is_multiplet=is_multiplet,
                      center_i=center_i, c_half=c_half, baseline_deg=baseline_deg,
                      baseline_region=baseline_region, fit_sat=warm["fit_sat"],
                      satellite_threshold=satellite_threshold, loss=loss,
                      huber_delta_factor=huber_delta_factor,
                      sigma0=sigma_f, gamma0=gamma_f,
                      sigma_hi=default_hi[0], gamma_hi=default_hi[1],
                      ppm_step=ppm_step, noise=noise)
    return rec


def fit_dataset(x, spectra, *, offsets=(0.0,), heights_rel=(1.0,), n_atoms=1,
                center=None, center_window=None, baseline_deg=None,
                baseline_region=None, fit_satellites=False,
                satellite_threshold=0.005, fwhm_prior=None, fwhm_prior_r2=None,
                fwhm_bounds_factor=2.0, loss="ls", huber_delta_factor=0.5,
                norm=None, ids=None, progress=False):
    """Fit every row of ``spectra`` (shape ``(n, p)``) over shared axis ``x``.

    Warm-starts once on the mean spectrum, then fits each row. Optional
    per-spectrum ``fwhm_prior`` (+ ``fwhm_prior_r2`` confidence in [0, 1]) tightens
    the width bounds. ``norm`` (per-spectrum) yields a ``quant`` column
    (``area_total / (norm * n_atoms)``); ``ids`` (a DataFrame) is carried through.

    Returns a :class:`~nmr_quant.results.QuantResults`.
    """
    import pandas as pd

    from .results import QuantResults

    x = np.ascontiguousarray(x, dtype=np.float64)
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    xn = _x_norm(x)
    ppm_step = abs(x[1] - x[0])
    ppm_span = abs(x[-1] - x[0])
    is_multiplet = len(offsets) > 1
    c_half = center_window if center_window is not None else ppm_span * 0.1

    center_arr = (np.asarray(center, dtype=float)
                  if center is not None and not np.isscalar(center) else None)
    center_scalar = (float(np.nanmedian(center_arr)) if center_arr is not None
                     else (float(center) if np.isscalar(center) else None))
    deg_arr = (np.asarray(baseline_deg) if baseline_deg is not None
               and not np.isscalar(baseline_deg) else None)
    deg_max = (int(max((int(v) for v in deg_arr if v is not None and int(v) >= 0),
                       default=-1)) if deg_arr is not None
               else (int(baseline_deg) if np.isscalar(baseline_deg) and baseline_deg is not None
                     else -1))

    y_mean = np.nanmean(spectra, axis=0)
    warm = _warm_start(x, y_mean, xn, offsets=offsets, heights_rel=heights_rel,
                       is_multiplet=is_multiplet, center=center_scalar,
                       center_window=center_window,
                       baseline_deg=deg_max if deg_max >= 0 else None,
                       baseline_region=baseline_region, fit_satellites=fit_satellites,
                       satellite_threshold=satellite_threshold,
                       ppm_step=ppm_step, ppm_span=ppm_span)
    sigma_f, gamma_f = warm["warm"][2], warm["warm"][3]
    fwhm_mean = _shapes.pseudo_voigt_fwhm(sigma_f, gamma_f)
    default_hi = (sigma_f * 5, gamma_f * 5) if is_multiplet else (ppm_span / 4, ppm_span / 4)

    rows = range(len(spectra))
    if progress:
        from tqdm.auto import tqdm
        rows = tqdm(rows, desc="Fitting spectra")

    records = []
    for i in rows:
        y_i = spectra[i]
        if float(y_i.max()) <= 0:
            continue
        center_i = (float(center_arr[i]) if center_arr is not None
                    and i < len(center_arr) and np.isfinite(center_arr[i])
                    else warm["warm"][1])
        deg_i = (int(deg_arr[i]) if deg_arr is not None and i < len(deg_arr)
                 and int(deg_arr[i]) >= 0 else
                 (int(baseline_deg) if np.isscalar(baseline_deg)
                  and baseline_deg is not None else None))
        prior = fwhm_prior[i] if fwhm_prior is not None and i < len(fwhm_prior) else None
        prior_r2 = (fwhm_prior_r2[i] if fwhm_prior_r2 is not None
                    and i < len(fwhm_prior_r2) else None)
        sigma0, gamma0, sigma_hi, gamma_hi = _prior_widths(
            sigma_f, gamma_f, fwhm_mean, prior, prior_r2, fwhm_bounds_factor,
            default_hi[0], default_hi[1], ppm_step)

        rec, _ = _fit_one(x, y_i, xn, warm=warm["warm"], offsets=offsets,
                          heights_rel=heights_rel, is_multiplet=is_multiplet,
                          center_i=center_i, c_half=c_half, baseline_deg=deg_i,
                          baseline_region=baseline_region, fit_sat=warm["fit_sat"],
                          satellite_threshold=satellite_threshold, loss=loss,
                          huber_delta_factor=huber_delta_factor,
                          sigma0=sigma0, gamma0=gamma0, sigma_hi=sigma_hi,
                          gamma_hi=gamma_hi, ppm_step=ppm_step, noise=None)
        row = {"idx": i, **rec.as_dict()}
        if norm is not None:
            nrm = float(norm[i]) if not np.isscalar(norm) else float(norm)
            row["quant"] = (rec.area_total / (nrm * n_atoms)
                            if np.isfinite(nrm) and nrm != 0 else float("nan"))
        records.append(row)

    df = pd.DataFrame(records)
    if ids is not None and len(df):
        ids_df = ids.reset_index(drop=True)
        keep = ids_df.iloc[df["idx"].to_numpy()].reset_index(drop=True)
        for col in keep.columns:
            df[col] = keep[col].to_numpy()

    model_meta = {
        "function": "multiplet" if is_multiplet else "voigt_with_satellites",
        "offsets": [float(o) for o in offsets] if is_multiplet else None,
        "heights_rel": [float(h) for h in heights_rel] if is_multiplet else None,
        "fit_satellites": bool(warm["fit_sat"]),
        "n_atoms": n_atoms,
        "baseline_ppm_x0": float(x[0]),
        "baseline_ppm_x1": float(x[-1]),
        "fwhm_formula": "0.5346*2*gamma + sqrt(0.2166*(2*gamma)**2 + (2*sqrt(2ln2)*sigma)**2)",
        "mean_fit_params": [float(v) for v in warm["warm"]],
    }
    return QuantResults(df=df, model_meta=model_meta)
