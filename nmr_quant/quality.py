"""Generic quality-map rules engine over a fit-results table.

The rules engine is universal; the specific threshold *values* are caller
config (e.g. fovea's ``*_qmap.json``). A quality map is a flat dict of numeric
bounds plus a ``rules`` list of pandas-``eval`` expressions (OR-combined) that
reference bounds as ``@name``.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd


def load_quality_map(path: str) -> dict:
    """Load a quality map previously saved as JSON."""
    with open(path) as f:
        return json.load(f)


def _eval_rule(df: pd.DataFrame, rule: str, bounds: dict) -> pd.Series:
    """Evaluate a rule expression, substituting ``@name`` from ``bounds``."""
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in bounds:
            raise KeyError(f"Rule references '@{name}' but it is not in the qmap bounds")
        return repr(bounds[name])
    return df.eval(re.sub(r"@(\w+)", _sub, rule)).astype(bool)


def build_quality_map(df: pd.DataFrame, *, r2_min: float = 0.99,
                      r2_adj_min: float | None = None, srr_min: float | None = None,
                      kde_min: float = 0.01, fwhm_range=None, area_range=None,
                      center_range=None, center_std: float | None = 1.96,
                      rules: list[str] | None = None) -> dict:
    """Derive a quality map (flat bounds + auto-generated rules) from ``df``.

    Bounds not supplied are auto-detected from the good-fit population
    (``r2 >= r2_min``): FWHM/area from the 1st-99th percentile, center from
    mean +/- ``center_std`` sigma, r2_adj/srr from the 5th percentile.
    """
    good = df[df["r2"] >= r2_min]

    if fwhm_range is None and "fwhm" in df.columns:
        fwhm_range = (float(np.percentile(good["fwhm"], 1)),
                      float(np.percentile(good["fwhm"], 99)))
    if area_range is None and "area_total" in df.columns:
        area_range = (float(np.percentile(good["area_total"], 1)),
                      float(np.percentile(good["area_total"], 99)))
    if center_range is None and center_std is not None and "center" in df.columns:
        c_mean, c_std = float(good["center"].mean()), float(good["center"].std())
        center_range = (c_mean - center_std * c_std, c_mean + center_std * c_std)
    if r2_adj_min is None and "r2_adj" in good.columns:
        r2_adj_min = float(np.percentile(good["r2_adj"].dropna(), 5))
    if srr_min is None and "srr" in good.columns:
        srr_min = float(np.percentile(good["srr"].dropna(), 5))

    bounds: dict = {"r2_min": r2_min}
    if r2_adj_min is not None:
        bounds["r2_adj_min"] = r2_adj_min
    if srr_min is not None:
        bounds["srr_min"] = srr_min
    if kde_min and "kde_score" in df.columns:
        bounds["kde_min"] = kde_min
    if fwhm_range is not None:
        bounds["fwhm_lo"], bounds["fwhm_hi"] = fwhm_range
    if area_range is not None:
        bounds["area_lo"], bounds["area_hi"] = area_range
    if center_range is not None:
        bounds["center_lo"], bounds["center_hi"] = center_range

    if rules is None:
        parts = ["r2 >= @r2_min"]
        if r2_adj_min is not None:
            parts.append("r2_adj >= @r2_adj_min")
        if srr_min is not None:
            parts.append("srr >= @srr_min")
        if fwhm_range is not None:
            parts += ["fwhm >= @fwhm_lo", "fwhm <= @fwhm_hi"]
        if area_range is not None:
            parts += ["area_total >= @area_lo", "area_total <= @area_hi"]
        if center_range is not None:
            parts += ["center >= @center_lo", "center <= @center_hi"]
        kde_part = ["kde_score >= @kde_min"] if bounds.get("kde_min") else []
        rules = [" and ".join(parts + kde_part)]
        if kde_part:
            rules.append(" and ".join(parts))

    return {**bounds, "rules": rules}


def apply_quality_map(df: pd.DataFrame, qmap: dict) -> pd.DataFrame:
    """Return a copy of ``df`` with ``quality_pass`` and ``quality_reason`` columns.

    ``quality_reason`` lists per-criterion failures for every row (pass or fail),
    which drives badge colours in the dashboards package. Overall pass is the
    OR of the ``rules`` expressions (or, without rules, the AND of all criteria).
    """
    crit: dict[str, pd.Series] = {}

    def _lo_hi(key_lo, key_hi, key_range):
        if qmap.get(key_lo) is not None:
            return qmap[key_lo], qmap[key_hi]
        if qmap.get(key_range) is not None:
            return qmap[key_range][0], qmap[key_range][1]
        return None, None

    if qmap.get("r2_min") is not None:
        crit["r2"] = df["r2"] >= qmap["r2_min"]
    if qmap.get("r2_adj_min") is not None and "r2_adj" in df.columns:
        crit["r2_adj"] = df["r2_adj"] >= qmap["r2_adj_min"]
    if qmap.get("srr_min") is not None and "srr" in df.columns:
        crit["srr"] = df["srr"] >= qmap["srr_min"]
    fwhm_lo, fwhm_hi = _lo_hi("fwhm_lo", "fwhm_hi", "fwhm_range")
    if fwhm_lo is not None:
        crit["fwhm"] = df["fwhm"].between(fwhm_lo, fwhm_hi)
    area_lo, area_hi = _lo_hi("area_lo", "area_hi", "area_range")
    if area_lo is not None:
        crit["area"] = df["area_total"].between(area_lo, area_hi)
    center_lo, center_hi = _lo_hi("center_lo", "center_hi", "center_range")
    if center_lo is not None and "center" in df.columns:
        crit["center"] = df["center"].between(center_lo, center_hi)
    if qmap.get("kde_min") is not None and "kde_score" in df.columns:
        crit["kde_score"] = df["kde_score"] >= qmap["kde_min"]

    reasons = pd.Series([""] * len(df), index=df.index, dtype=str)
    for name, pass_mask in crit.items():
        reasons[~pass_mask] += name + " "
    reasons = reasons.str.strip()

    rules = qmap.get("rules")
    if rules:
        bounds = {k: v for k, v in qmap.items() if k != "rules" and v is not None}
        mask = pd.Series(False, index=df.index)
        for rule in rules:
            try:
                mask |= _eval_rule(df, rule, bounds)
            except Exception as exc:  # noqa: BLE001 - report and skip a bad rule
                warnings.warn(f"apply_quality_map: rule skipped — {exc}\n  rule: {rule!r}",
                              stacklevel=2)
    else:
        mask = pd.Series(True, index=df.index)
        for pass_mask in crit.values():
            mask &= pass_mask

    out = df.copy()
    out["quality_pass"] = mask
    out["quality_reason"] = reasons
    return out


def kde_score(df: pd.DataFrame, *, r2_threshold: float = 0.99) -> np.ndarray:
    """2-D KDE density score (FWHM x area_total), normalised to [0, 1].

    Fitted on the good-fit population, evaluated for all rows: higher means more
    typical, lower means outlier. This is a cohort statistic (not a per-fit
    metric), which is why it lives here rather than in :mod:`nmr_quant.metrics`.
    """
    from scipy.stats import gaussian_kde

    good = df[df["quality_pass"]] if "quality_pass" in df.columns \
        else df[df["r2"] >= r2_threshold]
    if len(good) < 10:
        warnings.warn("Too few good spectra to fit KDE — kde_score not computed.",
                      stacklevel=2)
        return np.full(len(df), np.nan)

    kde_data = np.vstack([good["fwhm"].to_numpy(), good["area_total"].to_numpy()])
    kde_data = kde_data[:, np.isfinite(kde_data).all(axis=0)]
    if kde_data.shape[1] < 10:
        warnings.warn("Too few finite good-fit points — kde_score not computed.",
                      stacklevel=2)
        return np.full(len(df), np.nan)
    kde = gaussian_kde(kde_data)

    all_data = np.vstack([df["fwhm"].to_numpy(), df["area_total"].to_numpy()])
    finite = np.isfinite(all_data).all(axis=0)
    scores = np.full(len(df), np.nan)
    if finite.any():
        scores[finite] = kde(all_data[:, finite])
        mx = np.nanmax(scores)
        if mx > 0:
            scores /= mx
    return scores
