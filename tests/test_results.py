"""Tests for QuantResults save/load and model reconstruction."""

from __future__ import annotations

import numpy as np

from nmr_quant import QuantResults, _shapes, fit_dataset


def _dataset():
    x = np.linspace(-1.0, 1.0, 1200)
    spectra = np.vstack([
        _shapes.voigt_height(x, h, 0.0, 0.03, 0.02)
        for h in (8.0, 10.0, 12.0)
    ])
    return x, spectra


def test_roundtrip_without_dataset(tmp_path):
    x, spectra = _dataset()
    res = fit_dataset(x, spectra, baseline_deg=1)
    path = tmp_path / "acetate"
    res.save(str(path))
    assert (tmp_path / "acetate.parquet").exists()
    assert (tmp_path / "acetate.json").exists()

    loaded = QuantResults.load(str(path))
    assert loaded.model_meta == res.model_meta
    assert len(loaded.df) == len(res.df)
    for col in ("center", "area_total", "r2", "fwhm"):
        assert np.allclose(loaded.df[col].to_numpy(), res.df[col].to_numpy())
    # baseline_coeffs survives the json-string round trip as a list
    assert isinstance(loaded.df["baseline_coeffs"].iloc[0], list)
    assert len(loaded.df["baseline_coeffs"].iloc[0]) == 2


def test_model_curve_reconstructs_fit(tmp_path):
    x, spectra = _dataset()
    res = fit_dataset(x, spectra)
    row = res.df.iloc[1]
    curve = res.model_curve(row, x)
    # reconstructed curve should closely match the (clean) input spectrum
    assert np.corrcoef(curve, spectra[1])[0, 1] > 0.999


def test_model_curve_with_baseline(tmp_path):
    x, spectra = _dataset()
    xn = 2.0 * (x - 0.5 * (x[0] + x[-1])) / (x[-1] - x[0])
    spectra_bl = spectra + (1.5 * xn + 0.5)
    res = fit_dataset(x, spectra_bl, baseline_deg=1)
    row = res.df.iloc[0]
    curve = res.model_curve(row, x)
    assert np.corrcoef(curve, spectra_bl[0])[0, 1] > 0.999
