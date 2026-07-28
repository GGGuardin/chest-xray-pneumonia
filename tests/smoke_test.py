"""Prueba de humo del pipeline completo sobre datos sintéticos.

Ejecuta, encadenados: generación de datos -> manifiesto + split por paciente ->
entrenamiento breve -> evaluación -> Grad-CAM -> análisis de subgrupos.
Todo en CPU, en menos de un par de minutos, sin descargar ningún dataset.

    python tests/smoke_test.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TMP = RAIZ / ".smoke"


def run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    r = subprocess.run([sys.executable, *args], cwd=RAIZ)
    if r.returncode != 0:
        raise SystemExit(f"FALLO en: {' '.join(args)}")


def main() -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    datos = TMP / "raw"
    manifest = TMP / "manifest.csv"

    run(["tests/make_synthetic_data.py", "--out", str(datos), "--patients", "40",
         "--per-patient", "2", "--size", "128"])

    run(["-m", "src.prepare_data", "--dataset", "folder", "--root", str(datos),
         "--out", str(manifest)])

    df = pd.read_csv(manifest)
    # Metadatos sintéticos de subgrupo: el builder 'folder' no los trae, pero el
    # análisis de fairness sí debe ejercitarse en la prueba.
    rng = np.random.default_rng(0)
    pacientes = df["patient_id"].unique()
    sexo = dict(zip(pacientes, rng.choice(["M", "F"], len(pacientes))))
    edad = dict(zip(pacientes, rng.integers(20, 85, len(pacientes))))
    df["sex"] = df["patient_id"].map(sexo)
    df["age"] = df["patient_id"].map(edad)
    df["view"] = rng.choice(["AP", "PA"], len(df))
    df.to_csv(manifest, index=False)

    # Comprobación explícita de la propiedad crítica: ningún paciente cruza splits
    cruces = df.groupby("patient_id")["split"].nunique()
    assert (cruces == 1).all(), "FUGA DE DATOS: hay pacientes en más de un split"
    print(f"\nOK: {len(df)} imágenes, {df['patient_id'].nunique()} pacientes, sin fuga entre splits.")

    cfg = TMP / "smoke.yaml"
    cfg.write_text(
        f"manifest: {manifest.as_posix()}\n"
        f"out_dir: {(TMP / 'run').as_posix()}\n"
        "seed: 42\narch: densenet121\npretrained: false\ndropout: 0.1\n"
        "img_size: 96\nbatch_size: 8\nnum_workers: 0\nepochs: 2\n"
        "lr: 3.0e-4\nweight_decay: 1.0e-4\nearly_stopping_patience: 3\n"
        "use_pos_weight: true\nbalance_train: false\n",
        encoding="utf-8",
    )

    run(["-m", "src.train", "--config", str(cfg)])
    run(["-m", "src.evaluate", "--checkpoint", str(TMP / "run" / "best.pth"),
         "--manifest", str(manifest), "--split", "test",
         "--out-dir", str(TMP / "reporte"), "--num-workers", "0", "--n-boot", "100"])
    run(["-m", "src.explain", "--checkpoint", str(TMP / "run" / "best.pth"),
         "--manifest", str(manifest), "--split", "test", "--n", "4",
         "--out-dir", str(TMP / "gradcam")])
    run(["-m", "src.fairness", "--predictions", str(TMP / "reporte" / "predictions.csv"),
         "--out-dir", str(TMP / "fairness"), "--min-n", "5"])

    esperados = [
        TMP / "run" / "best.pth",
        TMP / "run" / "history.csv",
        TMP / "reporte" / "metrics.json",
        TMP / "reporte" / "curvas.png",
        TMP / "reporte" / "matriz_confusion.png",
        TMP / "gradcam" / "shortcut_audit.json",
        TMP / "fairness" / "subgrupos.csv",
    ]
    faltan = [p for p in esperados if not p.exists()]
    if faltan:
        raise SystemExit("Faltan artefactos: " + ", ".join(str(p) for p in faltan))

    print("\n" + "=" * 70)
    print("PRUEBA DE HUMO SUPERADA: el pipeline completo funciona de punta a punta.")
    print(f"Artefactos temporales en {TMP} (puedes borrar la carpeta).")
    print("=" * 70)


if __name__ == "__main__":
    main()
