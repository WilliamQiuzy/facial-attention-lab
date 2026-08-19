"""House-Brackmann grading metrics.

HB Grades are ordinal (I = normal through VI = total paralysis), so we want
metrics that respect ordinality:
  - Cohen's quadratic-weighted kappa: penalizes far-off mistakes more than
    near-misses. Standard for ordinal medical scales.
  - Plain accuracy: for sanity, but it's a poor primary metric here.
  - Mean Absolute Error in grades: 0 = perfect; useful intuitive readout.
  - Confusion matrix: which grades get confused with which.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HB_GRADE_NAMES = ["I", "II", "III", "IV", "V", "VI"]


def hb_label_string(grade_zero_indexed: int) -> str:
    """Convert a 0-indexed class id to the clinical HB roman-numeral string."""
    return HB_GRADE_NAMES[int(grade_zero_indexed)]


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Cohen's kappa with quadratic weights.

    Range: typically -1 to 1; 0 = chance; 1 = perfect. Clinically:
      <0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 good, >0.8 very good.
    """
    o = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        o[int(t), int(p)] += 1
    if o.sum() == 0:
        return float("nan")
    o /= o.sum()

    # expected matrix from marginals
    row = o.sum(axis=1)
    col = o.sum(axis=0)
    e = np.outer(row, col)

    # quadratic weights
    idx = np.arange(n_classes)
    w = (idx[:, None] - idx[None, :]) ** 2 / max(1, (n_classes - 1)) ** 2

    num = (w * o).sum()
    den = (w * e).sum()
    if den < 1e-12:
        return 1.0  # perfect-agreement degenerate case
    return 1.0 - num / den


@dataclass
class HBMetrics:
    accuracy: float
    mae_grades: float
    quadratic_kappa: float
    confusion: np.ndarray  # (n_classes, n_classes) int counts

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> "HBMetrics":
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
        mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float("nan")
        kappa = _quadratic_weighted_kappa(y_true, y_pred, n_classes)

        return cls(accuracy=acc, mae_grades=mae, quadratic_kappa=kappa, confusion=cm)

    def pretty(self) -> str:
        lines = [
            f"accuracy:        {self.accuracy:.3f}",
            f"MAE (grades):    {self.mae_grades:.3f}",
            f"quadratic kappa: {self.quadratic_kappa:.3f}",
            "confusion (rows=true, cols=pred; HB I..VI):",
            "       " + " ".join(f"{n:>4s}" for n in HB_GRADE_NAMES),
        ]
        for i, name in enumerate(HB_GRADE_NAMES):
            row = " ".join(f"{int(self.confusion[i, j]):>4d}" for j in range(len(HB_GRADE_NAMES)))
            lines.append(f"  {name:<4s} {row}")
        return "\n".join(lines)
