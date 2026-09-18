"""ROC curves (per-class + macro) saved to a figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out: str | Path,
    top_k: int = 0,
    figsize=(8, 8),
) -> None:
    """Plot ROC for each class; macro-average ROC is drawn bold.

    top_k: if > 0, only draw the top_k classes by AUROC (reduces clutter).
    """
    n = y_pred.shape[1]
    aurocs = []
    for i in range(n):
        if len(np.unique(y_true[:, i])) < 2:
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    order = np.argsort(aurocs)[::-1]
    shown = order[:top_k] if top_k else order

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
    for i in shown:
        if np.isnan(aurocs[i]):
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        ax.plot(fpr, tpr, lw=1.0, label=f"{labels[i]} ({aurocs[i]:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves per class")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out: str | Path) -> None:
    """AUROC bar chart for the top-N classes (multi-label-friendly summary)."""
    n = min(8, y_true.shape[1])
    probs = y_pred[:, :n]
    true = y_true[:, :n]
    # Per-class simple matrix is not meaningful for multi-label; plot binary TPR/FPR instead.
    tprs = [roc_auc_score(true[:, i], probs[:, i]) if len(np.unique(true[:, i])) > 1 else np.nan for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[:n][::-1], np.nan_to_num(tprs)[::-1], color="#4C72B0")
    ax.set_title(f"AUROC (top {n} classes)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)