# nmr-quant

Least-squares fitting of NMR peak lineshapes (single Voigt, Voigt with
satellites, and annotation-driven multiplets) to 1D spectra, with standard
fit metrics and quantities (peak areas, FWHM, optional normalised
concentration).

The package is general-purpose: it takes plain NumPy arrays in and returns a
plain pandas table out. It knows nothing about any specific data source, LIMS,
or plotting. Reference deconvolution and other row processing live in
`python-nmr-spectra-processing`; visualisation lives in
`python-nmr-quant-dashboards`.

## Status

Feature-complete. Public API:

- `fit_spectrum(x, y, ...)` — fit one spectrum -> `FitRecord`. Single Voigt,
  Voigt with satellites, or an annotation-driven multiplet, with an optional
  simultaneous polynomial baseline and LS or Huber loss.
- `fit_dataset(x, spectra, ...)` — fit a 2-D batch -> `QuantResults`.
  Warm-starts on the mean spectrum, then fits each row with optional
  per-spectrum FWHM/centre priors; `norm` yields a `quant` column and `ids`
  are carried through.
- `QuantResults` — the fit table + model metadata, with `save`/`load`
  (parquet + json, no dataset needed) and `model_curve` for reconstruction.
- `fit_metrics` — standard goodness-of-fit and residual diagnostics
  (`r2`, noise-adjusted `r2_adj`, `srr`, `sse`, `rmse`, `mae`, `residual_std`,
  `noise_std`, `aic`, `bic`, Durbin-Watson `dw`, `chi2_red`, `snr`).
- `build_quality_map` / `apply_quality_map` / `load_quality_map` / `kde_score`
  — the generic quality-map rules engine (thresholds are caller config).
- `nmr_quant._shapes` — Voigt / multiplet lineshape primitives (bridge module;
  these become a re-export from `python-nmr-spectra-processing.lineshapes`
  once that module is published).

34 tests, ~94% coverage.

## Install (dev)

```bash
uv sync
uv run pytest
```

## Design

- Arrays in, table out. No coupling to any loader or to matplotlib.
- Inputs coerced to C-contiguous `float64`; batches are one 2-D
  `(n_spectra, n_points)` array.
- Metrics are computed for every fit, not left to the caller.

## License

MIT.
