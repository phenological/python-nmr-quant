"""Shared fixtures: synthetic spectra built from known parameters."""

from __future__ import annotations

import numpy as np
import pytest

from nmr_quant import _shapes


@pytest.fixture
def ppm():
    """A uniform ppm axis over a small ROI (descending is not required here)."""
    return np.linspace(-1.0, 1.0, 1000)


@pytest.fixture
def clean_voigt(ppm):
    """A single clean Voigt with known parameters.

    Returns (x, y, params) where params = dict(height, center, sigma, gamma).
    """
    params = {"height": 10.0, "center": 0.05, "sigma": 0.03, "gamma": 0.02}
    y = _shapes.voigt_height(ppm, **params)
    return ppm, y, params


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
