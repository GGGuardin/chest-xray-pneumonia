"""Evaluación en test (o en un dataset externo) con métricas honestas.

Uso:
    python -m src.evaluate --checkpoint runs/rsna/best.pth --manifest data/manifest_rsna.csv \
        --split test --out-dir reports/rsna_test

    # Validación externa: entrenado en RSNA, evaluado en otro dataset completo
    python -m src.evaluate --checkpoint runs/rsna/best.pth --manifest data/manifest_kaggle.csv \
        --split all --out-dir reports/externo_kaggle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import ChestXrayDataset, build_transforms
from .metrics import binary_metrics, bootstrap_ci, plot_confusion, plot_curves
from .model import load_checkpoint
from .utils import get_device, save_json, seed_everything, setup_logging

log = setup_logging()


@torch.no_grad()
def predict(model, df: pd.DataFrame, img_size: int = 224, batch_size: int = 32,
            num_workers: int = 2, device=None) -> np.ndarray:
    """Probabilidades predichas para cada fila del manifiesto, en orden."""
    from torch.utils.data import DataLoader

    device = device or get_device()
    model = model.to(device).eval()
    ds = ChestXrayDataset(df, build_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    probs = []
    for images, _ in tqdm(loader, desc="inferencia", unit="batch", leave=False):
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return np.concatenate(probs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalúa un checkpoint en un split o dataset externo")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", help="train/val/test o 'all'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Por defecto usa el umbral de Youden fijado en validación")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seed_everything(42)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    threshold = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    log.info("Checkpoint época %s | AUROC val %.4f | umbral %.3f",
             ckpt.get("epoch"), ckpt.get("val_auroc", float("nan")), threshold)

    manifest = pd.read_csv(args.manifest)
    df = manifest if args.split == "all" else manifest[manifest["split"] == args.split]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No hay filas para split='{args.split}' en {args.manifest}")
    log.info("Evaluando %d imágenes (%d pacientes), prevalencia %.3f",
             len(df), df["patient_id"].nunique(), df["label"].mean())

    probs = predict(model, df, img_size=cfg.get("img_size", 224), batch_size=args.batch_size,
                    num_workers=args.num_workers, device=device)
    y = df["label"].values

    preds = df.copy()
    preds["prob"] = probs
    preds["pred"] = (probs >= threshold).astype(int)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    m = binary_metrics(y, probs, threshold)
    lo_roc, hi_roc = bootstrap_ci(y, probs, "auroc", n_boot=args.n_boot, threshold=threshold)
    lo_prc, hi_prc = bootstrap_ci(y, probs, "auprc", n_boot=args.n_boot, threshold=threshold)
    m["auroc_ci95"] = [lo_roc, hi_roc]
    m["auprc_ci95"] = [lo_prc, hi_prc]
    m["checkpoint"] = args.checkpoint
    m["manifest"] = args.manifest
    m["split"] = args.split

    plot_curves(y, probs, str(out_dir / "curvas.png"), title=f"{Path(args.manifest).stem} / {args.split}")
    plot_confusion(y, probs, threshold, str(out_dir / "matriz_confusion.png"))
    save_json(m, out_dir / "metrics.json")

    print("\n=== Resultados ===")
    print(f"  n                = {m['n']} imágenes ({df['patient_id'].nunique()} pacientes)")
    print(f"  prevalencia      = {m['prevalencia']:.3f}")
    print(f"  AUROC            = {m['auroc']:.4f}  IC95% [{lo_roc:.4f}, {hi_roc:.4f}]")
    print(f"  AUPRC            = {m['auprc']:.4f}  IC95% [{lo_prc:.4f}, {hi_prc:.4f}]")
    print(f"  umbral           = {threshold:.3f}")
    print(f"  sensibilidad     = {m['sensibilidad']:.4f}")
    print(f"  especificidad    = {m['especificidad']:.4f}")
    print(f"  PPV / NPV        = {m['ppv']:.4f} / {m['npv']:.4f}")
    print(f"  accuracy         = {m['accuracy']:.4f}   <- no la uses como métrica principal")
    print(f"  Brier            = {m['brier']:.4f}")
    print(f"\nArtefactos en {out_dir}")

    if not np.isnan(m["auroc"]) and m["auroc"] < 0.85:
        log.warning("AUROC < 0,85: por debajo del umbral de éxito fijado en la guía del proyecto.")


if __name__ == "__main__":
    main()
