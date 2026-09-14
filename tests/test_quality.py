"""Tests for the quality-map rules engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nmr_quant import apply_quality_map, build_quality_map, kde_score, load_quality_map

N_GOOD = 100


def _df():
    """N_GOOD tightly-clustered good rows + 2 clear outliers (low r2, off-centre)."""
    rng = np.random.default_rng(0)
    good = pd.DataFrame({
        "r2": np.full(N_GOOD, 0.999),
        "r2_adj": np.full(N_GOOD, 0.99),
        "srr": np.full(N_GOOD, 200.0),
        "fwhm": 0.011 + rng.standard_normal(N_GOOD) * 2e-4,
        "area_total": 1.0 + rng.standard_normal(N_GOOD) * 0.02,
        "center": rng.standard_normal(N_GOOD) * 5e-4,
    })
    bad = pd.DataFrame({
        "r2": [0.5, 0.8],
        "r2_adj": [0.4, 0.7],
        "srr": [3.0, 5.0],
        "fwhm": [0.05, 0.001],
        "area_total": [10.0, 0.01],
        "center": [0.5, -0.5],
    })
    return pd.concat([good, bad], ignore_index=True)


def test_apply_explicit_rule():
    df = _df()
    out = apply_quality_map(df, {"r2_min": 0.99, "rules": ["r2 >= @r2_min"]})
    assert out["quality_pass"].tolist() == [True] * N_GOOD + [False, False]


def test_quality_reason_lists_failures():
    df = _df()
    qmap = {"r2_min": 0.99, "center_lo": -0.01, "center_hi": 0.01,
            "rules": ["r2 >= @r2_min and center >= @center_lo and center <= @center_hi"]}
    out = apply_quality_map(df, qmap)
    assert "r2" in out["quality_reason"].iloc[N_GOOD]
    assert "center" in out["quality_reason"].iloc[N_GOOD]
    assert out["quality_reason"].iloc[0] == ""


def test_build_quality_map_passes_bulk():
    df = _df()
    qmap = build_quality_map(df, r2_min=0.99, kde_min=0)
    assert "rules" in qmap and qmap["rules"]
    assert "fwhm_lo" in qmap and "area_hi" in qmap
    out = apply_quality_map(df, qmap)
    # auto-bounds trim ~1% tails, so the bulk of good fits pass ...
    assert out["quality_pass"].iloc[:N_GOOD].mean() >= 0.85
    # ... and the clear outliers all fail
    assert not out["quality_pass"].iloc[N_GOOD:].any()


def test_roundtrip_qmap_json(tmp_path):
    import json
    qmap = {"r2_min": 0.99, "rules": ["r2 >= @r2_min"]}
    p = tmp_path / "qmap.json"
    p.write_text(json.dumps(qmap))
    assert load_quality_map(str(p)) == qmap


def test_kde_score_range():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "r2": [0.999] * n,
        "fwhm": 0.011 + rng.standard_normal(n) * 5e-4,
        "area_total": 1.0 + rng.standard_normal(n) * 0.05,
    })
    scores = kde_score(df)
    finite = scores[np.isfinite(scores)]
    assert finite.size == n
    assert finite.min() >= 0.0 and finite.max() <= 1.0
