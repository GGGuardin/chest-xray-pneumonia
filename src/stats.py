"""Comparación estadística entre modelos sobre las mismas muestras.

Decir "0,727 frente a 0,720, los intervalos se solapan" es juzgar a ojo. Dos
AUROC medidos sobre **las mismas imágenes** están correlacionados, así que sus
intervalos independientes exageran la incertidumbre de la diferencia: el test
de DeLong y el bootstrap pareado la miden bien.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Rangos con promedio en los empates (el midrank de DeLong)."""
    orden = np.argsort(x)
    ordenado = x[orden]
    n = len(x)
    rangos = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and ordenado[j + 1] == ordenado[i]:
            j += 1
        rangos[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    salida = np.empty(n, dtype=float)
    salida[orden] = rangos
    return salida


def _estructural(pos: np.ndarray, neg: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC y componentes de varianza V10/V01 de un predictor (DeLong 1988)."""
    m, n = len(pos), len(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    tx, ty = _midrank(pos), _midrank(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v10, v01


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Test de DeLong para dos AUROC medidos sobre las mismas muestras.

    Devuelve ambos AUC, la diferencia, su error estándar y el p-valor bilateral
    de la hipótesis nula "los dos modelos discriminan igual".
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a, prob_b = np.asarray(prob_a, float), np.asarray(prob_b, float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Hace falta al menos un positivo y un negativo.")

    mask_pos = y_true == 1
    auc_a, v10_a, v01_a = _estructural(prob_a[mask_pos], prob_a[~mask_pos])
    auc_b, v10_b, v01_b = _estructural(prob_b[mask_pos], prob_b[~mask_pos])
    m, n = int(mask_pos.sum()), int((~mask_pos).sum())

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / m + s01 / n

    diferencia = auc_a - auc_b
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    ee = float(np.sqrt(max(var, 1e-300)))
    z = diferencia / ee if ee > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "auroc_a": round(auc_a, 4),
        "auroc_b": round(auc_b, 4),
        "diferencia": round(diferencia, 4),
        "error_estandar": round(ee, 4),
        "ic95_diferencia": [round(diferencia - 1.96 * ee, 4), round(diferencia + 1.96 * ee, 4)],
        "z": round(float(z), 3),
        "p_valor": float(f"{p:.3g}"),
        "n": int(len(y_true)),
        "n_positivos": m,
        "significativo_005": bool(p < 0.05),
    }


def bootstrap_pareado(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metrica: str = "auroc",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """IC de la diferencia remuestreando **los mismos índices** en ambos modelos.

    Es lo que hace pareada la comparación: si se remuestrease por separado, la
    correlación entre modelos se perdería y el intervalo saldría inflado.
    """
    from .metrics import binary_metrics

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    prob_a, prob_b = np.asarray(prob_a, float), np.asarray(prob_b, float)

    diferencias = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        ma = binary_metrics(y_true[idx], prob_a[idx])[metrica]
        mb = binary_metrics(y_true[idx], prob_b[idx])[metrica]
        diferencias.append(ma - mb)

    diferencias = np.asarray(diferencias)
    lo, hi = np.percentile(diferencias, [2.5, 97.5])
    return {
        "metrica": metrica,
        "diferencia_media": round(float(diferencias.mean()), 4),
        "ic95": [round(float(lo), 4), round(float(hi), 4)],
        "cruza_cero": bool(lo <= 0 <= hi),
        "n_remuestreos": int(len(diferencias)),
    }
