"""Tests for the linear parameter-transfer priors."""

from __future__ import annotations

import numpy as np
import pytest

from nmr_quant import apply_linear_prior, fit_linear_prior


def test_recovers_known_line():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 2.0, 200)
    target = 1.7 * ref + 0.3
    m = fit_linear_prior(ref, target)
    assert np.isclose(m["slope"], 1.7, rtol=1e-6)
    assert np.isclose(m["intercept"], 0.3, atol=1e-6)
    assert np.isclose(m["r2_fit"], 1.0, rtol=1e-9)
    assert m["residual_std"] < 1e-9
    assert m["n_total"] == 200 and m["n_training"] == 200


def test_mask_selects_training_rows():
    ref = np.array([1.0, 2.0, 3.0, 100.0])   # last row is a quality-fail outlier
    target = np.array([2.0, 4.0, 6.0, -50.0])
    m = fit_linear_prior(ref, target, mask=[True, True, True, False])
    assert np.isclose(m["slope"], 2.0, rtol=1e-9)
    assert m["n_total"] == 4 and m["n_training"] == 3


def test_drops_nonfinite():
    ref = np.array([1.0, 2.0, np.nan, 4.0])
    target = np.array([3.0, 5.0, 7.0, np.nan])
    m = fit_linear_prior(ref, target)
    assert m["n_training"] == 2  # only the first two pairs are finite


def test_apply_predicts_and_propagates_nan():
    model = {"slope": 2.0, "intercept": 1.0}
    out = apply_linear_prior(model, [0.0, 1.0, np.nan])
    assert np.allclose(out[:2], [1.0, 3.0])
    assert np.isnan(out[2])


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_linear_prior([1.0], [2.0])
