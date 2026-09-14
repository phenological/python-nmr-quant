"""nmr-quant — least-squares fitting of NMR peak lineshapes to 1D spectra.

Public API (growing):
    fit_metrics   standard goodness-of-fit and residual diagnostics
    _shapes       lineshape primitives (bridge; see module docstring)

Fitting entry points (fit_spectrum, fit_dataset), the QuantResults container,
and the quality-map rules engine are added in later increments.
"""

from __future__ import annotations

from .metrics import fit_metrics

__all__ = ["fit_metrics"]
