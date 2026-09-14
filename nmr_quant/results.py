"""QuantResults: the per-spectrum fit table plus model metadata, with I/O.

Saves to ``{path}.parquet`` (the table) + ``{path}.json`` (model metadata and
optional quality map). Unlike the fovea original, :meth:`load` needs no dataset:
the results stand on their own, and model curves are reconstructed on demand
from an ``x`` axis passed in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import _shapes

_PARAM_COLS = ["height", "center", "sigma", "gamma",
               "f_sat", "delta_sat", "sigma_sat", "gamma_sat"]
_ID_COLS = ["idx", "nmrFolderId", "dataPath"]


@dataclass
class QuantResults:
    """Container for a fit table (``df``) and its ``model_meta``."""

    df: pd.DataFrame
    model_meta: dict = field(default_factory=dict)
    qmap: dict | None = None

    # ── I/O ────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Write ``{path}.parquet`` + ``{path}.json``."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        id_cols = [c for c in _ID_COLS if c in self.df.columns]
        param_cols = [c for c in _PARAM_COLS if c in self.df.columns]
        rest = [c for c in self.df.columns if c not in id_cols and c not in param_cols]
        df_save = self.df[id_cols + param_cols + rest].copy()
        if "baseline_coeffs" in df_save.columns:
            df_save["baseline_coeffs"] = df_save["baseline_coeffs"].apply(
                lambda v: json.dumps(list(v)) if isinstance(v, (list, np.ndarray)) else "[]"
            )
        df_save.to_parquet(str(base) + ".parquet", index=False)

        with open(str(base) + ".json", "w") as f:
            json.dump({"model": self.model_meta, "qmap": self.qmap}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> QuantResults:
        """Read results saved by :meth:`save`. No dataset required."""
        base = Path(path)
        df = pd.read_parquet(str(base) + ".parquet")
        if "baseline_coeffs" in df.columns:
            df["baseline_coeffs"] = df["baseline_coeffs"].apply(
                lambda v: json.loads(v) if isinstance(v, str)
                else (list(v) if v is not None else [])
            )
        meta = {}
        json_path = Path(str(base) + ".json")
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
        return cls(df=df, model_meta=meta.get("model", {}), qmap=meta.get("qmap"))

    # ── model reconstruction ─────────────────────────────────────────────────
    def model_curve(self, row, x) -> np.ndarray:
        """Reconstruct the fitted model curve for one results ``row`` on axis ``x``.

        ``row`` is a mapping/Series with the parameter columns; the model form
        (single/satellite Voigt vs multiplet) and baseline come from
        ``model_meta``. Used by the dashboards package to overlay fits.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        meta = self.model_meta
        h, c = float(row["height"]), float(row["center"])
        sg, gm = float(row["sigma"]), float(row["gamma"])

        if meta.get("function") == "multiplet":
            y = _shapes.multiplet(x, h, c, sg, gm,
                                  meta.get("offsets", [0.0]),
                                  meta.get("heights_rel", [1.0]))
        else:
            y = _shapes.voigt_with_satellites(
                x, h, c, sg, gm,
                float(row.get("f_sat", 0.0) or 0.0),
                float(row.get("delta_sat", 0.0) or 0.0),
                float(row.get("sigma_sat", sg) or sg),
                float(row.get("gamma_sat", gm) or gm),
            )

        coeffs = row.get("baseline_coeffs")
        if coeffs is not None and len(coeffs):
            x0 = meta.get("baseline_ppm_x0", x[0])
            x1 = meta.get("baseline_ppm_x1", x[-1])
            xn = 2.0 * (x - 0.5 * (x0 + x1)) / (x1 - x0)
            y = y + np.polyval(coeffs, xn)
        return y

    def __len__(self) -> int:
        return len(self.df)
