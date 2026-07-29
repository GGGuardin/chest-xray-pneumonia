"""Entrenamiento por fine-tuning, con selección de modelo por AUROC de validación.

Uso:
    python -m src.train --config configs/rsna.yaml
    python -m src.train --config configs/rsna.yaml --epochs 2 --limit-train 500   # prueba rápida
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import assert_no_patient_leakage, make_loaders, pos_weight_from
from .metrics import binary_metrics, youden_threshold
from .model import build_model
from .utils import AverageMeter, get_device, load_config, save_json, seed_everything, setup_logging

log = setup_logging()


# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc="") -> tuple:
    """Una pasada completa. Con `optimizer` entrena; sin él, evalúa."""
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_meter = AverageMeter()
    probs, targets = [], []

    amp_enabled = device.type == "cuda"
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy().ravel())
        targets.append(labels.detach().cpu().numpy().ravel())
        bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg, np.concatenate(probs), np.concatenate(targets)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuning de un clasificador binario de tórax")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None, help="Sobrescribe el manifiesto del config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="Sobrescribe la semilla: repetir con varias mide la varianza de "
                         "inicialización, sin la cual una diferencia entre modelos no es "
                         "interpretable")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="Usa solo N imágenes de train (prueba rápida de humo)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for key, value in [("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("arch", args.arch), ("manifest", args.manifest),
                       ("out_dir", args.out_dir), ("seed", args.seed)]:
        if value is not None:
            cfg[key] = value

    seed_everything(cfg.get("seed", 42))
    device = get_device(args.device)
    out_dir = Path(cfg.get("out_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Dispositivo: %s | salida: %s", device, out_dir)

    manifest = pd.read_csv(cfg["manifest"])
    assert_no_patient_leakage(manifest)
    if args.limit_train:
        train_part = manifest[manifest["split"] == "train"].sample(
            n=min(args.limit_train, (manifest["split"] == "train").sum()),
            random_state=cfg.get("seed", 42),
        )
        manifest = pd.concat([train_part, manifest[manifest["split"] != "train"]])
        log.warning("Modo prueba: train limitado a %d imágenes", len(train_part))

    loaders = make_loaders(
        manifest,
        img_size=cfg.get("img_size", 224),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 2),
        use_clahe=cfg.get("use_clahe", False),
        balance_train=cfg.get("balance_train", False),
    )
    if "val" not in loaders:
        raise ValueError("El manifiesto no tiene split 'val'; ejecuta src.prepare_data primero.")

    model = build_model(
        arch=cfg.get("arch", "densenet121"),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.0),
        freeze_backbone=cfg.get("freeze_backbone", False),
    ).to(device)

    # Desbalance: se compensa con pos_weight en la loss (o con sampler balanceado).
    pos_weight = pos_weight_from(manifest) if cfg.get("use_pos_weight", True) else 1.0
    log.info("pos_weight = %.3f", pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_auroc, best_epoch, patience = -1.0, -1, int(cfg.get("early_stopping_patience", 5))
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_probs, train_y = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler, desc=f"train {epoch}/{epochs}"
        )
        val_loss, val_probs, val_y = run_epoch(
            model, loaders["val"], criterion, device, desc=f"  val {epoch}/{epochs}"
        )
        scheduler.step()

        train_m = binary_metrics(train_y, train_probs)
        val_m = binary_metrics(val_y, val_probs)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auroc": train_m["auroc"],
            "val_auroc": val_m["auroc"],
            "val_auprc": val_m["auprc"],
            "segundos": round(time.time() - t0, 1),
        }
        history.append(row)
        log.info(
            "época %02d | train loss %.4f auroc %.4f | val loss %.4f auroc %.4f auprc %.4f | %.0fs",
            epoch, train_loss, train_m["auroc"], val_loss, val_m["auroc"], val_m["auprc"], row["segundos"],
        )

        if val_m["auroc"] > best_auroc:
            best_auroc, best_epoch = val_m["auroc"], epoch
            # El umbral operativo se fija en VALIDACIÓN, nunca en test.
            threshold = youden_threshold(val_y, val_probs)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_auroc": best_auroc,
                    "threshold": threshold,
                    "val_metrics": val_m,
                    # Necesaria para corregir el a priori al aplicar el modelo a
                    # una población con otra prevalencia (ver src/calibration.py).
                    "train_prevalence": float(
                        manifest[manifest["split"] == "train"]["label"].mean()
                    ),
                },
                out_dir / "best.pth",
            )
            log.info("  -> nuevo mejor modelo (AUROC val %.4f, umbral Youden %.3f)", best_auroc, threshold)

        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        if epoch - best_epoch >= patience:
            log.info("Early stopping: %d épocas sin mejorar.", patience)
            break

    save_json({"config": cfg, "best_epoch": best_epoch, "best_val_auroc": best_auroc,
               "device": str(device)}, out_dir / "train_summary.json")
    log.info("Listo. Mejor AUROC de validación: %.4f (época %d). Checkpoint: %s",
             best_auroc, best_epoch, out_dir / "best.pth")


if __name__ == "__main__":
    main()
