"""Tests for fit_metrics."""

from __future__ import annotations

import numpy as np

from nmr_quant import fit_metrics


def test_perfect_fit():
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 0.0])
    m = fit_metrics(y, y.copy(), height=5.0, n_params=4)
    assert m["sse"] == 0.0
    assert m["r2"] == 1.0
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0


def test_basic_consistency():
    y = np.linspace(0, 1, 50)
    y_model = y + 0.1
    m = fit_metrics(y, y_model, height=1.0, n_params=4)
    n = y.size
    assert np.isclose(m["sse"], n * 0.1**2)
    assert np.isclose(m["rmse"], np.sqrt(m["sse"] / n))
    assert np.isclose(m["mae"], 0.1)


def test_dw_white_noise_near_two(rng):
    resid = rng.standard_normal(4000)
    m = fit_metrics(resid, np.zeros_like(resid), height=1.0)
    assert 1.7 < m["dw"] < 2.3


def test_dw_structured_residual_small():
    resid = np.arange(500.0)  # strongly autocorrelated ramp
    m = fit_metrics(resid, np.zeros_like(resid), height=1.0)
    assert m["dw"] < 0.1


def test_srr_from_edge_noise():
    n, k, s = 200, 20, 0.5
    resid = np.zeros(n)
    pattern = np.tile([s, -s], k // 2)  # zero-mean, std == s
    resid[:k] = pattern
    resid[-k:] = pattern
    m = fit_metrics(resid, np.zeros_like(resid), height=10.0, n_noise=k)
    assert np.isclose(m["noise_std"], s, rtol=1e-6)
    assert np.isclose(m["srr"], 10.0 / s, rtol=1e-6)


def test_noise_override_drives_chi2_and_snr():
    y = np.linspace(0, 1, 100)
    y_model = y + 0.05
    m = fit_metrics(y, y_model, height=2.0, n_params=4, noise=0.5)
    assert np.isclose(m["snr"], 2.0 / 0.5)
    dof = y.size - 4
    assert np.isclose(m["chi2_red"], m["sse"] / (0.5**2 * dof))


def test_aic_bic_require_n_params():
    y = np.linspace(0, 1, 100)
    y_model = y + 0.05
    without = fit_metrics(y, y_model, height=1.0)
    assert np.isnan(without["aic"]) and np.isnan(without["bic"])
    withp = fit_metrics(y, y_model, height=1.0, n_params=4)
    assert np.isfinite(withp["aic"]) and np.isfinite(withp["bic"])
    # for n > e^2, bic penalises more than aic
    assert withp["bic"] > withp["aic"]


def test_r2_adj_is_noise_adjusted():
    # r2_adj (above-noise variance explained) >= r2 when there is real noise
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 400)
    peak = 5.0 * np.exp(-0.5 * (x / 0.05) ** 2)
    noise = rng.standard_normal(x.size) * 0.05
    y = peak + noise
    m = fit_metrics(y, peak, height=5.0, n_params=4)
    assert 0.0 <= m["r2_adj"] <= 1.0
    assert m["r2_adj"] >= m["r2"] - 1e-9
