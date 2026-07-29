"""Análisis comparativo entre los dos modelos, a partir de las predicciones guardadas.

Cierra tres debilidades del primer experimento sin volver a tocar la GPU:

1. **Comparación pareada.** El modelo A se evaluó sobre NIH completo y el B sobre
   su test retenido: conjuntos distintos. Aquí se intersecan por ruta de imagen
   para comparar sobre exactamente las mismas radiografías.
2. **Prueba estadística.** Test de DeLong y bootstrap pareado en vez de mirar si
   los intervalos se solapan.
3. **Calibración.** Se mide el ECE y se corrige el cambio de prevalencia, que es
   lo que hunde la sensibilidad en el conjunto pediátrico.

    python scripts/analisis_comparativo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.calibration import (  # noqa: E402
    corregir_por_prevalencia,
    expected_calibration_error,
    plot_reliability,
    sensibilidad_a_especificidad,
)
from src.metrics import binary_metrics  # noqa: E402
from src.stats import bootstrap_pareado, delong_test  # noqa: E402

DESCARGAS = RAIZ / ".kaggle_out"
SALIDA = RAIZ / "results" / "comparativa"


def cargar(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df["clave"] = df["image_path"].map(lambda p: Path(str(p)).name)
    return df


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    informe: dict = {}

    # ---------------------------------------------------------------- NIH --
    a_nih = cargar(DESCARGAS / "rsna/reports/externo_nih/predictions.csv")
    b_nih = cargar(DESCARGAS / "nih/reports/nih_test/predictions.csv")

    # El modelo A vio NIH entero; el B solo puede juzgarse en su test retenido.
    # La intersección es ese test: mismas imágenes para los dos.
    comun = a_nih.merge(b_nih[["clave", "prob", "label"]], on="clave", suffixes=("_a", "_b"))
    assert (comun["label_a"] == comun["label_b"]).all(), "Las etiquetas no coinciden"
    print(f"NIH pareado: {len(comun)} imágenes comunes "
          f"(A evaluó {len(a_nih)}, B evaluó {len(b_nih)})")

    y = comun["label_a"].values
    ma = binary_metrics(y, comun["prob_a"].values)
    mb = binary_metrics(y, comun["prob_b"].values)
    informe["nih_pareado"] = {
        "n": len(comun),
        "prevalencia": round(float(y.mean()), 4),
        "modelo_a_rsna": {"auroc": round(ma["auroc"], 4), "auprc": round(ma["auprc"], 4)},
        "modelo_b_nih": {"auroc": round(mb["auroc"], 4), "auprc": round(mb["auprc"], 4)},
        "delong": delong_test(y, comun["prob_a"].values, comun["prob_b"].values),
        "bootstrap_pareado_auroc": bootstrap_pareado(y, comun["prob_a"].values,
                                                     comun["prob_b"].values, "auroc"),
    }

    # -------------------------------------------------------- pediátrico --
    a_ped = cargar(DESCARGAS / "rsna/reports/externo_pediatrico/predictions.csv")
    b_ped = cargar(DESCARGAS / "nih/reports/externo_pediatrico/predictions.csv")
    ped = a_ped.merge(b_ped[["clave", "prob", "label"]], on="clave", suffixes=("_a", "_b"))
    yp = ped["label_a"].values
    print(f"Pediátrico pareado: {len(ped)} imágenes")

    informe["pediatrico_pareado"] = {
        "n": len(ped),
        "prevalencia": round(float(yp.mean()), 4),
        "delong": delong_test(yp, ped["prob_a"].values, ped["prob_b"].values),
        "bootstrap_pareado_auroc": bootstrap_pareado(yp, ped["prob_a"].values,
                                                     ped["prob_b"].values, "auroc"),
    }

    # ------------------------------------------- semillas contra modelo B --
    # Dos fuentes de ruido distintas y conviene no confundirlas: la varianza de
    # inicialización (entre semillas) y la de muestreo (finitud del test). Aquí
    # se miden ambas para saber cuál limita la comparación.
    dir_semillas = DESCARGAS / "semillas/reports"
    if dir_semillas.exists():
        filas, delongs = [], []
        for s in (42, 1337, 2024):
            ruta = dir_semillas / f"s{s}_nih_test/predictions.csv"
            if not ruta.exists():
                continue
            a_s = cargar(ruta)
            par = a_s.merge(b_nih[["clave", "prob"]], on="clave", suffixes=("_a", "_b"))
            d = delong_test(par["label"].values, par["prob_a"].values, par["prob_b"].values)
            filas.append(d["auroc_a"])
            delongs.append({"semilla": s, **d})
            print(f"semilla {s}: AUROC {d['auroc_a']:.4f} vs B {d['auroc_b']:.4f} "
                  f"| dif {d['diferencia']:+.4f} p={d['p_valor']}")

        if filas:
            import numpy as np

            informe["semillas_vs_modelo_b"] = {
                "auroc_por_semilla": filas,
                "media": round(float(np.mean(filas)), 4),
                "desviacion_entre_semillas": round(float(np.std(filas, ddof=1)), 4),
                "auroc_modelo_b": delongs[0]["auroc_b"],
                "todas_superan_a_b": bool(all(v > delongs[0]["auroc_b"] for v in filas)),
                "error_estandar_muestreo_delong": delongs[0]["error_estandar"],
                "delong_por_semilla": delongs,
                "lectura": (
                    "La dirección del efecto es consistente entre semillas, pero el error "
                    "estándar de muestreo es un orden de magnitud mayor que la dispersión "
                    "entre semillas: el factor limitante es el número de positivos del test "
                    "de NIH, no la inicialización."
                ),
            }

    # ------------------------------------------------------- calibración --
    # El modelo A entrenó con 22,6% de prevalencia; el pediátrico tiene 73%.
    PREV_ENTRENAMIENTO = 0.2259
    prev_destino = float(yp.mean())
    umbral = 0.5831

    prob_cruda = ped["prob_a"].values
    prob_corregida = corregir_por_prevalencia(prob_cruda, PREV_ENTRENAMIENTO, prev_destino)

    antes = binary_metrics(yp, prob_cruda, umbral)
    despues = binary_metrics(yp, prob_corregida, umbral)
    informe["calibracion_pediatrico"] = {
        "prevalencia_entrenamiento": PREV_ENTRENAMIENTO,
        "prevalencia_destino": round(prev_destino, 4),
        "umbral": umbral,
        "sin_corregir": {k: round(antes[k], 4) for k in
                         ("auroc", "sensibilidad", "especificidad", "f1", "brier", "accuracy")},
        "corregido_por_prevalencia": {k: round(despues[k], 4) for k in
                                      ("auroc", "sensibilidad", "especificidad", "f1", "brier",
                                       "accuracy")},
        "ece_sin_corregir": expected_calibration_error(yp, prob_cruda)["ece"],
        "ece_corregido": expected_calibration_error(yp, prob_corregida)["ece"],
        "punto_operacion_comun": {
            "modelo_a_esp90": sensibilidad_a_especificidad(yp, prob_cruda, 0.90),
            "modelo_b_esp90": sensibilidad_a_especificidad(yp, ped["prob_b"].values, 0.90),
        },
    }

    plot_reliability(yp, prob_cruda, str(SALIDA / "fiabilidad_pediatrico_sin_corregir.png"),
                     "Pediátrico — sin corregir")
    plot_reliability(yp, prob_corregida, str(SALIDA / "fiabilidad_pediatrico_corregido.png"),
                     "Pediátrico — corregido por prevalencia")

    # Calibración del modelo A en su propio test, como referencia
    a_test = cargar(DESCARGAS / "rsna/reports/rsna_test/predictions.csv")
    informe["calibracion_test_interno"] = {
        "ece": expected_calibration_error(a_test["label"].values, a_test["prob"].values)["ece"],
        "brier": round(binary_metrics(a_test["label"].values, a_test["prob"].values)["brier"], 4),
    }
    plot_reliability(a_test["label"].values, a_test["prob"].values,
                     str(SALIDA / "fiabilidad_test_interno.png"), "RSNA test interno")

    with open(SALIDA / "comparativa.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    print("\n" + json.dumps(informe, indent=2, ensure_ascii=False))
    print(f"\nArtefactos en {SALIDA}")


if __name__ == "__main__":
    main()
