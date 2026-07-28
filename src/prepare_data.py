"""Construye el manifiesto normalizado y el split por paciente.

Uso:
    python -m src.prepare_data --dataset rsna --root data/raw/rsna --out data/manifest_rsna.csv
    python -m src.prepare_data --dataset kaggle_pneumonia --root data/raw/chest_xray \
        --out data/manifest_kaggle.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import MANIFEST_BUILDERS, make_splits, split_summary
from .utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera manifiesto + split por paciente")
    ap.add_argument("--dataset", required=True, choices=sorted(MANIFEST_BUILDERS))
    ap.add_argument("--root", required=True, help="Directorio con el dataset descomprimido")
    ap.add_argument("--out", required=True, help="Ruta del CSV de salida")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude-not-normal",
        action="store_true",
        help="Solo RSNA: descarta la clase ambigua 'No Lung Opacity / Not Normal'",
    )
    ap.add_argument(
        "--target",
        default="Pneumonia",
        help="Solo NIH: etiqueta positiva (Pneumonia, Effusion, Cardiomegaly... o 'any')",
    )
    ap.add_argument(
        "--frontal-only",
        action="store_true",
        help="Solo NIH: conserva únicamente proyecciones PA/AP",
    )
    args = ap.parse_args()

    log = setup_logging()
    builder = MANIFEST_BUILDERS[args.dataset]
    kwargs: dict = {}
    if args.dataset == "rsna":
        kwargs = {"exclude_not_normal": args.exclude_not_normal}
    elif args.dataset == "nih":
        kwargs = {"target": args.target, "frontal_only": args.frontal_only}
    df = builder(args.root, **kwargs)
    log.info("Manifiesto: %d imágenes, %d pacientes, %.1f%% positivos",
             len(df), df["patient_id"].nunique(), 100 * df["label"].mean())

    df = make_splits(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Guardado en %s", out)
    print(split_summary(df).to_string())


if __name__ == "__main__":
    main()
