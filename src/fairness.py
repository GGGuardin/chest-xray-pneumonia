"""Análisis de subgrupos: infradiagnóstico por sexo, edad y proyección.

Réplica en miniatura del diseño de Seyyed-Kalantari, Zhang, McDermott, Chen y
Ghassemi (Nature Medicine 27:2176-2182, 2021): la métrica central es la tasa de
falsos negativos (FNR) por subgrupo, es decir, cuántos pacientes enfermos son
etiquetados como sanos. Se añade la proyección AP/PA porque es un parámetro
técnico de adquisición conocido por inducir sesgo (y correlacionado con la
gravedad del paciente: los enfermos graves se radiografían en decúbito, AP).

Uso:
    python -m src.fairness --predictions reports/rsna_test/predictions.csv --out-dir reports/fairness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import binary_metrics
from .utils import save_json, setup_logging

log = setup_logging()

AGE_BINS = [0, 18, 40, 60, 75, 200]
AGE_LABELS = ["0-17", "18-39", "40-59", "60-74", "75+"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns and df["age"].notna().any():
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
        df.loc[df["age"].isna(), "age_group"] = "UNKNOWN"
    else:
        df["age_group"] = "UNKNOWN"
    return df


def subgroup_table(df: pd.DataFrame, by: str, threshold: float, min_n: int = 30) -> pd.DataFrame:
    """Métricas por subgrupo. Descarta subgrupos con muy pocas muestras."""
    rows = []
    for value, part in df.groupby(by, dropna=False):
        if len(part) < min_n or part["label"].nunique() < 2:
            log.info("Subgrupo %s=%s omitido (n=%d, clases=%d)",
                     by, value, len(part), part["label"].nunique())
            continue
        m = binary_metrics(part["label"].values, part["prob"].values, threshold)
        rows.append({
            "atributo": by,
            "subgrupo": str(value),
            "n": m["n"],
            "prevalencia": round(m["prevalencia"], 4),
            "auroc": round(m["auroc"], 4),
            "sensibilidad": round(m["sensibilidad"], 4),
            "especificidad": round(m["especificidad"], 4),
            "fnr_infradiagnostico": round(m["fnr"], 4),
            "fpr": round(m["fpr"], 4),
        })
    tabla = pd.DataFrame(rows)
    if tabla.empty:  # todos los subgrupos quedaron por debajo de min_n
        return tabla
    return tabla.sort_values("fnr_infradiagnostico", ascending=False)


def intersectional_table(df: pd.DataFrame, attrs: list[str], threshold: float,
                         min_n: int = 30) -> pd.DataFrame:
    """Subgrupos interseccionales (p. ej. sexo x grupo de edad).

    Es justo donde Seyyed-Kalantari et al. encontraron el peor infradiagnóstico.
    """
    df = df.copy()
    df["_inter"] = df[attrs].astype(str).agg(" | ".join, axis=1)
    tabla = subgroup_table(df, "_inter", threshold, min_n)
    if not tabla.empty:
        tabla["atributo"] = " x ".join(attrs)
    return tabla


def main() -> None:
    ap = argparse.ArgumentParser(description="Análisis de sesgo por subgrupos")
    ap.add_argument("--predictions", required=True, help="CSV generado por src.evaluate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Por defecto usa la columna 'pred' ya umbralizada del CSV")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument(
        "--attributes",
        default="sex,age_group,view",
        help="Atributos a estratificar, separados por comas. `age_group` se deriva de `age`. "
             "Para dermatología: fitzpatrick,sex,age_group",
    )
    ap.add_argument(
        "--intersect",
        default="sex,age_group",
        help="Par de atributos para el análisis interseccional, o cadena vacía para omitirlo",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # Recupera el umbral implícito en la columna 'pred'
        pos = df.loc[df["pred"] == 1, "prob"]
        threshold = float(pos.min()) if len(pos) else 0.5
    df = add_age_group(df)

    atributos = [a.strip() for a in args.attributes.split(",") if a.strip()]
    tablas = []
    for attr in atributos:
        if attr in df.columns and df[attr].astype(str).nunique() > 1:
            t = subgroup_table(df, attr, threshold, args.min_n)
            if not t.empty:
                tablas.append(t)
        else:
            log.info("Atributo '%s' ausente o constante: se omite", attr)

    cruce = [a.strip() for a in args.intersect.split(",") if a.strip()]
    if len(cruce) >= 2 and set(cruce).issubset(df.columns):
        t = intersectional_table(df, cruce, threshold, args.min_n)
        if not t.empty:
            tablas.append(t)

    if not tablas:
        print("No hay metadatos de subgrupo utilizables (sexo/edad/proyección) en este manifiesto.\n"
              "El dataset Kaggle 'Chest X-Ray Images (Pneumonia)' no los incluye; RSNA sí, "
              "vía cabeceras DICOM.")
        return

    tabla = pd.concat(tablas, ignore_index=True)
    tabla.to_csv(out_dir / "subgrupos.csv", index=False)

    global_m = binary_metrics(df["label"].values, df["prob"].values, threshold)
    brechas = {}
    for attr, part in tabla.groupby("atributo"):
        brechas[attr] = {
            "fnr_max": float(part["fnr_infradiagnostico"].max()),
            "fnr_min": float(part["fnr_infradiagnostico"].min()),
            "brecha_fnr": float(part["fnr_infradiagnostico"].max() - part["fnr_infradiagnostico"].min()),
            "peor_subgrupo": str(part.iloc[0]["subgrupo"]),
        }
    save_json({"threshold": threshold, "global": global_m, "brechas": brechas}, out_dir / "fairness.json")

    _plot(tabla, out_dir / "fnr_por_subgrupo.png", global_m["fnr"])

    print(f"\nFNR global (infradiagnóstico) = {global_m['fnr']:.4f} @ umbral {threshold:.3f}\n")
    print(tabla.to_string(index=False))
    print("\nBrechas de FNR entre subgrupos:")
    for attr, b in brechas.items():
        print(f"  {attr:22s} brecha = {b['brecha_fnr']:.4f}  (peor: {b['peor_subgrupo']})")
    print(f"\nArtefactos en {out_dir}")


def _plot(tabla: pd.DataFrame, out_path: Path, fnr_global: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tabla.sort_values(["atributo", "fnr_infradiagnostico"])
    etiquetas = [f"{r.atributo}: {r.subgrupo} (n={r.n})" for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.4 * len(t) + 1.2)))
    ax.barh(etiquetas, t["fnr_infradiagnostico"], color="steelblue")
    ax.axvline(fnr_global, ls="--", color="crimson", label=f"FNR global = {fnr_global:.3f}")
    ax.set_xlabel("tasa de falsos negativos (infradiagnóstico)")
    ax.set_title("Infradiagnóstico por subgrupo")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
