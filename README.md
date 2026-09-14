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

Early. Implemented so far:

- `nmr_quant.fit_metrics` — standard goodness-of-fit and residual diagnostics
  (`r2`, noise-adjusted `r2_adj`, `srr`, `sse`, `rmse`, `mae`, `residual_std`,
  `noise_std`, `aic`, `bic`, Durbin-Watson `dw`, `chi2_red`, `snr`).
- `nmr_quant._shapes` — Voigt / multiplet lineshape primitives (bridge module;
  these move to `python-nmr-spectra-processing` once its `lineshapes` module is
  published).

Coming: `fit_spectrum`, `fit_dataset`, the `QuantResults` save/load container,
and the quality-map rules engine.

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
