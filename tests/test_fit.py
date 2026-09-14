"""Tests for fit_spectrum and fit_dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nmr_quant import _shapes, fit_dataset, fit_spectrum


def _axis(n=1600):
    return np.linspace(-1.0, 1.0, n)


class TestFitSpectrumSingle:
    def test_recovers_single_voigt(self):
        x = _axis()
        h, c, s, g = 10.0, 0.05, 0.03, 0.02
        y = _shapes.voigt_height(x, h, c, s, g)
        rec = fit_spectrum(x, y)
        assert rec.r2 > 0.999
        assert abs(rec.center - c) < 2 * (x[1] - x[0])
        assert np.isclose(rec.fwhm, _shapes.pseudo_voigt_fwhm(s, g), rtol=5e-2)
        assert np.isclose(rec.area_total, _shapes.area_from_height(h, s, g), rtol=5e-2)

    def test_noisy_peak_still_fits(self):
        rng = np.random.default_rng(1)
        x = _axis()
        h, c, s, g = 8.0, -0.1, 0.02, 0.02
        y = _shapes.voigt_height(x, h, c, s, g) + rng.standard_normal(x.size) * 0.02
        rec = fit_spectrum(x, y)
        assert rec.r2 > 0.99
        assert abs(rec.center - c) < 0.01
        assert rec.srr > 10  # strong signal vs residual noise


class TestBaseline:
    def test_recovers_area_over_linear_baseline(self):
        x = _axis()
        h, c, s, g = 10.0, 0.0, 0.03, 0.02
        true_area = _shapes.area_from_height(h, s, g)
        xn = 2.0 * (x - 0.5 * (x[0] + x[-1])) / (x[-1] - x[0])
        baseline = 2.0 * xn + 1.5  # slope + offset
        y = _shapes.voigt_height(x, h, c, s, g) + baseline
        rec = fit_spectrum(x, y, baseline_deg=1)
        assert len(rec.baseline_coeffs) == 2
        assert rec.r2 > 0.999
        assert np.isclose(rec.area_total, true_area, rtol=5e-2)


class TestSatellites:
    def test_detects_and_fits_satellites(self):
        x = _axis(2400)
        h, c, s, g = 10.0, 0.0, 0.02, 0.012
        y = _shapes.voigt_with_satellites(x, h, c, s, g,
                                          f_sat=0.1, delta_sat=0.3,
                                          sigma_sat=s, gamma_sat=g)
        rec = fit_spectrum(x, y, fit_satellites=True)
        assert rec.has_satellites
        assert 0.05 < rec.f_sat < 0.2
        assert abs(rec.delta_sat - 0.3) < 0.05

    def test_no_satellites_when_absent(self):
        x = _axis()
        y = _shapes.voigt_height(x, 10.0, 0.0, 0.03, 0.02)
        rec = fit_spectrum(x, y, fit_satellites=True)
        assert not rec.has_satellites


class TestMultiplet:
    def test_doublet_center_and_area(self):
        x = _axis(2000)
        h, c, s, g = 6.0, 0.0, 0.015, 0.01
        offsets, heights_rel = (-0.05, 0.05), (1.0, 1.0)
        y = _shapes.multiplet(x, h, c, s, g, offsets, heights_rel)
        rec = fit_spectrum(x, y, offsets=offsets, heights_rel=heights_rel)
        assert rec.r2 > 0.999
        assert abs(rec.center - c) < 0.005
        expected = _shapes.area_from_height(h, s, g) * sum(heights_rel)
        assert np.isclose(rec.area_total, expected, rtol=8e-2)


class TestFitDataset:
    def _batch(self, n=6):
        x = _axis()
        rng = np.random.default_rng(0)
        heights = np.linspace(5, 15, n)
        centers = np.linspace(-0.05, 0.05, n)
        spectra = np.vstack([
            _shapes.voigt_height(x, h, c, 0.03, 0.02)
            + rng.standard_normal(x.size) * 0.01
            for h, c in zip(heights, centers, strict=True)
        ])
        return x, spectra, heights, centers

    def test_shape_and_columns(self):
        x, spectra, heights, centers = self._batch()
        res = fit_dataset(x, spectra)
        df = res.df
        assert len(df) == len(spectra)
        for col in ("idx", "center", "area_total", "fwhm", "r2", "r2_adj", "srr", "dw"):
            assert col in df.columns
        assert np.allclose(df["center"].to_numpy(), centers, atol=0.01)
        assert (df["r2"] > 0.99).all()

    def test_norm_gives_quant_and_ids_carried(self):
        x, spectra, heights, centers = self._batch()
        norm = np.full(len(spectra), 2.0)
        ids = pd.DataFrame({"nmrFolderId": np.arange(100, 100 + len(spectra))})
        res = fit_dataset(x, spectra, n_atoms=1, norm=norm, ids=ids)
        assert "quant" in res.df.columns
        assert "nmrFolderId" in res.df.columns
        assert np.allclose(res.df["quant"].to_numpy(),
                           res.df["area_total"].to_numpy() / 2.0)

    def test_single_row(self):
        x = _axis()
        y = _shapes.voigt_height(x, 10.0, 0.0, 0.03, 0.02)
        res = fit_dataset(x, y[None, :])
        assert len(res.df) == 1


class TestPriors:
    def test_prior_does_not_break_fit(self):
        x = _axis()
        y = _shapes.voigt_height(x, 10.0, 0.0, 0.03, 0.02)
        true_fwhm = _shapes.pseudo_voigt_fwhm(0.03, 0.02)
        spectra = y[None, :]
        # trusted tight prior
        r1 = fit_dataset(x, spectra, fwhm_prior=np.array([true_fwhm]),
                         fwhm_prior_r2=np.array([1.0]))
        # ignored prior (confidence 0)
        r0 = fit_dataset(x, spectra, fwhm_prior=np.array([true_fwhm * 3]),
                         fwhm_prior_r2=np.array([0.0]))
        assert np.isclose(r1.df["fwhm"].iloc[0], true_fwhm, rtol=1e-1)
        assert np.isclose(r0.df["fwhm"].iloc[0], true_fwhm, rtol=1e-1)
