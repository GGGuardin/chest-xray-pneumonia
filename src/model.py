"""Modelo: transfer learning con una sola salida logit (clasificación binaria).

DenseNet-121 es el estándar de facto en radiografía de tórax (CheXNet,
torchxrayvision). Se permite cualquier backbone de `timm` como alternativa
(EfficientNet-B0, ConvNeXt-Tiny, ResNet-50...).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger("cxr.model")


def build_model(
    arch: str = "densenet121",
    pretrained: bool = True,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Devuelve un modelo con cabeza de 1 logit.

    `arch` acepta "densenet121" (vía torchvision) o cualquier nombre de timm.
    """
    if arch == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
            if dropout > 0
            else nn.Linear(in_features, 1)
        )
    else:
        import timm

        model = timm.create_model(arch, pretrained=pretrained, num_classes=1, drop_rate=dropout)

    if freeze_backbone:
        head_names = {"classifier", "fc", "head"}
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(h) for h in head_names)
        log.info("Backbone congelado: solo se entrena la cabeza.")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Modelo %s | %.1fM parámetros (%.1fM entrenables)", arch, n_params / 1e6, n_train / 1e6)
    return model


def find_target_layer(model: nn.Module) -> nn.Module:
    """Última capa convolucional, para Grad-CAM.

    Se buscan primero los nombres conocidos y, si no hay coincidencia, se toma
    el último `nn.Conv2d` del grafo en orden de declaración.
    """
    named = dict(model.named_modules())
    for key in ("features.denseblock4", "features.norm5", "layer4", "stages.3"):
        if key in named:
            return named[key]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No hay capas Conv2d en el modelo: Grad-CAM no aplica.")
    return convs[-1]


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Reconstruye el modelo desde un checkpoint guardado por `train.py`."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(
        arch=cfg.get("arch", "densenet121"),
        pretrained=False,
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
