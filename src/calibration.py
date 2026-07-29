"""Calibración y corrección por cambio de prevalencia.

El resultado más llamativo del proyecto —AUROC 0,92 en el conjunto pediátrico
con sensibilidad 0,36— no es un fallo de discriminación sino de calibración: el
umbral se fijó en una población con 23% de prevalencia y se aplicó a otra con
73%. Aquí se mide esa descalibración y se corrige de forma principiada.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """ECE: discrepancia media entre confianza declarada y acierto observado.

    Se agrupan las predicciones en tramos de probabilidad y se compara, en cada
    tramo, la probabilidad media predicha con la frecuencia real de positivos.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, tramos = 0.0, 0.0, []

    for lo, hi in zip(bordes[:-1], bordes[1:]):
        sel = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not sel.any():
            continue
        conf = float(y_prob[sel].mean())
        obs = float(y_true[sel].mean())
        peso = float(sel.sum()) / len(y_prob)
        brecha = abs(conf - obs)
        ece += peso * brecha
        mce = max(mce, brecha)
        tramos.append({"rango": [round(lo, 2), round(hi, 2)], "n": int(sel.sum()),
                       "confianza_media": round(conf, 4), "frecuencia_real": round(obs, 4)})

    return {"ece": round(ece, 4), "mce_max": round(mce, 4), "tramos": tramos}


def corregir_por_prevalencia(
    y_prob: np.ndarray,
    prevalencia_entrenamiento: float,
    prevalencia_destino: float,
) -> np.ndarray:
    """Reajusta las probabilidades a una prevalencia distinta (corrección de Elkan).

    Un clasificador entrenado con prevalencia p_train aplicado a una población con
    prevalencia p_dest tiene sus probabilidades sesgadas por el cambio de a priori.
    La corrección reescala las odds por el cociente de odds a priori:

        odds' = odds · [p_dest/(1-p_dest)] / [p_train/(1-p_train)]

    No mejora la discriminación —el AUROC no cambia, el orden se conserva—, pero
    devuelve las probabilidades a una escala en la que un umbral tiene sentido.
    Requiere conocer o estimar la prevalencia de destino, cosa que en la práctica
    clínica rara vez es gratis.
    """
    y_prob = np.clip(np.asarray(y_prob, float), 1e-7, 1 - 1e-7)
    odds = y_prob / (1 - y_prob)
    factor = ((prevalencia_destino / (1 - prevalencia_destino))
              / (prevalencia_entrenamiento / (1 - prevalencia_entrenamiento)))
    odds_corregidas = odds * factor
    return odds_corregidas / (1 + odds_corregidas)


def sensibilidad_a_especificidad(y_true: np.ndarray, y_prob: np.ndarray,
                                 especificidad_objetivo: float = 0.90) -> dict:
    """Sensibilidad alcanzable fijando la especificidad.

    Permite comparar modelos en un punto de operación común sin depender del
    umbral heredado, que es justo lo que se rompe al cambiar de población.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, umbrales = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, float))
    ok = np.where(fpr <= 1 - especificidad_objetivo)[0]
    i = int(ok[-1]) if len(ok) else 0
    return {
        "especificidad_objetivo": especificidad_objetivo,
        "sensibilidad": round(float(tpr[i]), 4),
        "umbral": round(float(umbrales[i]), 4),
    }


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, out_path: str,
                     titulo: str = "", n_bins: int = 10) -> None:
    """Diagrama de fiabilidad: confianza declarada frente a frecuencia observada."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = expected_calibration_error(y_true, y_prob, n_bins)
    conf = [t["confianza_media"] for t in d["tramos"]]
    obs = [t["frecuencia_real"] for t in d["tramos"]]

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="calibración perfecta")
    ax.plot(conf, obs, "o-", lw=2, color="steelblue", label=f"modelo (ECE = {d['ece']:.3f})")
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia real de positivos")
    ax.set_title(titulo or "Diagrama de fiabilidad")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
