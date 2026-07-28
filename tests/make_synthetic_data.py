"""Genera un dataset sintético con la forma de un dataset de tórax real.

No son radiografías: son siluetas toscas con o sin una "opacidad" gaussiana.
Sirven para probar que el pipeline completo funciona (splits, entrenamiento,
métricas, Grad-CAM, fairness) sin descargar gigabytes ni tocar datos clínicos.
Varias imágenes por paciente, para que el split por paciente sea significativo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _chest_like(rng: np.random.Generator, size: int = 256, opacidad: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2, size / 2

    fondo = np.full((size, size), 18.0, dtype=np.float32)
    torax = ((xx - cx) / (size * 0.36)) ** 2 + ((yy - cy) / (size * 0.44)) ** 2 < 1.0
    fondo[torax] = 95.0

    # Dos campos pulmonares oscuros
    for signo in (-1, 1):
        lung = ((xx - (cx + signo * size * 0.17)) / (size * 0.13)) ** 2 + \
               ((yy - cy * 1.05) / (size * 0.28)) ** 2 < 1.0
        fondo[lung] = 45.0

    # Mediastino / silueta cardiaca, desplazada a la izquierda del paciente
    corazon = ((xx - (cx + size * 0.05)) / (size * 0.11)) ** 2 + \
              ((yy - cy * 1.25) / (size * 0.15)) ** 2 < 1.0
    fondo[corazon] = 110.0

    if opacidad:
        oy = cy * 1.05 + rng.uniform(-0.12, 0.12) * size
        ox = cx + rng.choice([-1, 1]) * size * rng.uniform(0.10, 0.24)
        sigma = size * rng.uniform(0.05, 0.09)
        blob = 70.0 * np.exp(-(((xx - ox) ** 2 + (yy - oy) ** 2) / (2 * sigma ** 2)))
        fondo = fondo + blob

    fondo += rng.normal(0, 6, fondo.shape)
    return np.clip(fondo, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset sintético para pruebas del pipeline")
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--patients", type=int, default=60, help="Pacientes por clase")
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    total = 0
    for clase, opacidad, offset in [("NORMAL", False, 0), ("PNEUMONIA", True, 5000)]:
        d = out / clase
        d.mkdir(parents=True, exist_ok=True)
        for p in range(args.patients):
            for k in range(args.per_patient):
                img = _chest_like(rng, args.size, opacidad)
                # El prefijo "personNNNN" es lo que agrupa por paciente en el
                # manifiesto (mismo criterio que el dataset de Kaggle).
                Image.fromarray(img).save(d / f"person{offset + p:04d}_{clase.lower()}_{k}.png")
                total += 1
    print(f"{total} imágenes sintéticas en {out} "
          f"({args.patients} pacientes/clase x {args.per_patient} imágenes)")


if __name__ == "__main__":
    main()
