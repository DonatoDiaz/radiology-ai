"""Detection and classification metrics for multi-label predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Return per-class AUROC / AP plus macro averages.

    y_true, y_pred: (N, C) float/0-1 arrays.
    """
    n = y_pred.shape[1]
    results: dict[str, dict[str, float]] = {"per_class": {}, "macro": {}}

    aurocs, aps = [], []
    for i in range(n):
        # Skip degenerate classes present in exactly one value.
        if len(np.unique(y_true[:, i])) < 2:
            results["per_class"][labels[i]] = {"auroc": np.nan, "ap": np.nan}
            continue
        ai = roc_auc_score(y_true[:, i], y_pred[:, i])
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        aurocs.append(ai)
        aps.append(ap)
        results["per_class"][labels[i]] = {"auroc": float(ai), "ap": float(ap)}

    results["macro"]["auroc"] = float(np.nanmean(aurocs)) if aurocs else 0.0
    results["macro"]["ap"] = float(np.nanmean(aps)) if aps else 0.0
    return results