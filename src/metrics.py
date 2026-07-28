"""Métricas correctas para un problema desbalanceado.

La accuracy engaña: con 80% de negativos, predecir siempre "normal" da 80%.
Se reportan AUROC (discriminación), AUPRC (mejor con desbalance),
sensibilidad/especificidad a un umbral explícito, y matriz de confusión.
Los intervalos de confianza se estiman por bootstrap.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    single_class = len(np.unique(y_true)) < 2
    return {
        "n": int(len(y_true)),
        "prevalencia": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "auprc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "sensibilidad": float(sens),
        "especificidad": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan"),
        # FNR = tasa de infradiagnóstico: la métrica central del análisis de sesgo
        # (Seyyed-Kalantari et al., Nat Med 2021).
        "fnr": float(1 - sens) if not np.isnan(sens) else float("nan"),
        "fpr": float(1 - spec) if not np.isnan(spec) else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)) if not single_class else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Umbral que maximiza sensibilidad + especificidad - 1 (índice J de Youden).

    Debe elegirse SIEMPRE en validación, nunca en test.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def threshold_for_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target: float = 0.90) -> float:
    """Umbral más alto que aún alcanza la sensibilidad objetivo.

    En cribado interesa no perder enfermos: se fija la sensibilidad y se acepta
    la especificidad resultante.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    return float(thresholds[ok[0]]) if len(ok) else 0.5


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """IC percentil por bootstrap sobre las muestras del conjunto de test."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(binary_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    if not values:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str = "") -> None:
    """Guarda ROC + curva precision-recall en una figura."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="azar")
    axes[0].set_xlabel("1 - especificidad (FPR)")
    axes[0].set_ylabel("sensibilidad (TPR)")
    axes[0].set_title("Curva ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, lw=2, color="darkorange", label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(y_true.mean(), ls="--", color="grey", lw=1, label=f"prevalencia = {y_true.mean():.3f}")
    axes[1].set_xlabel("recall (sensibilidad)")
    axes[1].set_ylabel("precisión (PPV)")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend(loc="lower left")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int),
                          labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = ["NORMAL", "NEUMONÍA"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicción")
    ax.set_ylabel("verdad")
    ax.set_title(f"Matriz de confusión (umbral = {threshold:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
