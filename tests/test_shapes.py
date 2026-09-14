"""Tests for the lineshape primitives."""

from __future__ import annotations

import numpy as np

from nmr_quant import _shapes


def test_voigt_peak_equals_height():
    x = np.linspace(-2, 2, 2001)
    y = _shapes.voigt_height(x, height=7.5, center=0.1, sigma=0.05, gamma=0.03)
    assert np.isclose(y.max(), 7.5, rtol=1e-6)
    assert np.isclose(x[np.argmax(y)], 0.1, atol=x[1] - x[0])


def test_area_from_height_matches_numeric_integral():
    x = np.arange(-20, 20, 0.001)
    h, sigma, gamma = 3.0, 0.05, 0.05
    y = _shapes.voigt_height(x, h, 0.0, sigma, gamma)
    numeric = np.trapezoid(y, x)
    analytic = _shapes.area_from_height(h, sigma, gamma)
    assert np.isclose(numeric, analytic, rtol=5e-3)


def test_height_area_roundtrip():
    h, sigma, gamma = 4.2, 0.02, 0.07
    area = _shapes.area_from_height(h, sigma, gamma)
    assert np.isclose(_shapes.height_from_area(area, sigma, gamma), h, rtol=1e-10)


def test_gaussian_limit_area():
    # gamma -> 0: Voigt collapses to a Gaussian, area = h * sigma * sqrt(2 pi)
    h, sigma, gamma = 5.0, 0.1, 1e-7
    expected = h * sigma * np.sqrt(2 * np.pi)
    assert np.isclose(_shapes.area_from_height(h, sigma, gamma), expected, rtol=1e-4)


def test_pseudo_voigt_fwhm_matches_numeric():
    x = np.linspace(-5, 5, 20001)
    sigma, gamma = 0.08, 0.05
    y = _shapes.voigt_height(x, 1.0, 0.0, sigma, gamma)
    numeric = _shapes.measure_fwhm(x, y, 0.0)
    approx = _shapes.pseudo_voigt_fwhm(sigma, gamma)
    assert np.isclose(numeric, approx, rtol=2e-2)


def test_multiplet_doublet_positions_and_heights():
    x = np.linspace(-2, 2, 4001)
    offsets = (-0.3, 0.3)
    heights_rel = (1.0, 1.0)
    y = _shapes.multiplet(x, height=6.0, center=0.0, sigma=0.02, gamma=0.01,
                          offsets=offsets, heights_rel=heights_rel)
    left = y[x < 0]
    right = y[x > 0]
    assert np.isclose(x[x < 0][np.argmax(left)], -0.3, atol=0.01)
    assert np.isclose(x[x > 0][np.argmax(right)], 0.3, atol=0.01)
    # equal relative heights -> symmetric peaks
    assert np.isclose(left.max(), right.max(), rtol=1e-6)


def test_satellites_add_side_peaks():
    x = np.linspace(-2, 2, 4001)
    common = dict(height=10.0, center=0.0, sigma=0.01, gamma=0.005)
    no_sat = _shapes.voigt_with_satellites(x, f_sat=0.0, delta_sat=0.0,
                                           sigma_sat=0.01, gamma_sat=0.005, **common)
    with_sat = _shapes.voigt_with_satellites(x, f_sat=0.1, delta_sat=0.5,
                                             sigma_sat=0.01, gamma_sat=0.005, **common)
    # satellites only add intensity; their tails raise the main peak slightly
    assert with_sat.max() >= no_sat.max()
    assert np.isclose(with_sat.max(), no_sat.max(), rtol=1e-3)
    # a clear side peak appears near +/- delta_sat (height ~ f_sat * height)
    sat_region = (np.abs(x - 0.5) < 0.05)
    assert with_sat[sat_region].max() > 0.5 * 10.0 * 0.1


def test_measure_fwhm_gaussian():
    x = np.linspace(-3, 3, 6001)
    sigma = 0.2
    y = np.exp(-0.5 * (x / sigma) ** 2)
    fwhm = _shapes.measure_fwhm(x, y, 0.0)
    assert np.isclose(fwhm, 2.3548 * sigma, rtol=1e-3)
