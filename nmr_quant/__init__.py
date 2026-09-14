"""nmr-quant — least-squares fitting of NMR peak lineshapes to 1D spectra.

Public API:
    fit_spectrum       fit a single spectrum -> FitRecord
    fit_dataset        fit a 2-D batch -> QuantResults
    FitRecord          one fitted spectrum (shapes, quantities, metrics)
    QuantResults       fit table + model metadata, with parquet/json save/load
    fit_metrics        standard goodness-of-fit and residual diagnostics
    build_quality_map / apply_quality_map / load_quality_map / kde_score
                       generic quality-map rules engine over a results table

Lineshape primitives live behind ``nmr_quant._shapes`` (a bridge that will
become a re-export from ``nmr_spectra_processing.lineshapes``).
"""

from __future__ import annotations

from .fit import FitRecord, fit_dataset, fit_spectrum
from .metrics import fit_metrics
from .priors import apply_linear_prior, fit_linear_prior
from .quality import apply_quality_map, build_quality_map, kde_score, load_quality_map
from .results import QuantResults

__all__ = [
    "FitRecord",
    "QuantResults",
    "apply_linear_prior",
    "apply_quality_map",
    "build_quality_map",
    "fit_dataset",
    "fit_linear_prior",
    "fit_metrics",
    "fit_spectrum",
    "kde_score",
    "load_quality_map",
]
