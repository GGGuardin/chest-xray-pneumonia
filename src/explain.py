"""Grad-CAM + comprobación cuantitativa de *shortcut learning*.

No basta con generar mapas bonitos: la pregunta es si el modelo mira el
parénquima pulmonar o los bordes, marcadores de lateralidad y texto quemado
(DeGrave, Janizek & Lee, Nat Mach Intell 2021). Por eso, además del mapa de
calor, se calcula la fracción de energía del CAM que cae en el marco exterior
de la imagen: si es alta, hay indicios de atajo espurio.

Uso:
    python -m src.explain --checkpoint runs/rsna/best.pth --manifest data/manifest_rsna.csv \
        --split test --n 12 --out-dir reports/gradcam
    python -m src.explain --checkpoint runs/rsna/best.pth --image ruta/a/imagen.dcm --out-dir reports/gradcam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_image
from .model import find_target_layer, load_checkpoint
from .utils import get_device, save_json, seed_everything, setup_logging

log = setup_logging()


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) sobre la última capa convolucional.

    Se implementa con hooks en vez de una dependencia externa: son ~30 líneas y
    deja explícito de dónde salen activaciones y gradientes.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model.eval()
        self.target_layer = target_layer or find_target_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Devuelve (cam normalizado [H, W] en [0,1], probabilidad predicha)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)          # (1, 1)
        prob = float(torch.sigmoid(logits).item())
        logits[:, 0].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No se capturaron activaciones/gradientes en la capa objetivo.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # GAP de los gradientes
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, prob

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


BORDER_FRAC = 0.15


def uniform_baseline(border_frac: float = BORDER_FRAC) -> float:
    """Energía en bordes que produciría un mapa uniforme.

    Es la referencia contra la que hay que comparar: con un marco del 15% por
    lado, el marco ocupa 1 - 0.7^2 = 51% del área, así que un CAM sin estructura
    da ~0.51. Por debajo = atención concentrada en el centro (pulmón); por
    encima = atención en bordes, marcadores o texto quemado.
    """
    return float(1.0 - (1.0 - 2.0 * border_frac) ** 2)


def border_energy_fraction(cam: np.ndarray, border_frac: float = BORDER_FRAC) -> float:
    """Fracción de la masa del CAM que cae en el marco exterior de la imagen.

    Devuelve NaN si el mapa es todo ceros, cosa que ocurre de forma sistemática
    en negativos bien clasificados: el Grad-CAM de la clase positiva no tiene
    nada que señalar y la ReLU lo anula. Compara siempre con `uniform_baseline()`.
    """
    h, w = cam.shape
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    inner = cam[bh:h - bh, bw:w - bw]
    total = cam.sum()
    if total <= 0:
        return float("nan")
    return float(1.0 - inner.sum() / total)


def resumen_atajos(detalle: list[dict], threshold: float) -> dict:
    """Agrega la auditoría separando detecciones positivas del resto.

    Promediar todas las imágenes juntas no significa nada: en los casos que el
    modelo da por negativos el mapa es ruido difuso y su energía en bordes es
    alta por construcción, lo que contamina la media y simula un atajo espurio
    donde no lo hay. La pregunta con sentido es: **cuando el modelo cree ver una
    opacidad, ¿dónde mira?**
    """
    base = uniform_baseline()
    detecciones = [d["border_energy"] for d in detalle
                   if d.get("prob", 0) >= threshold and not np.isnan(d.get("border_energy", np.nan))]
    resto = [d["border_energy"] for d in detalle
             if d.get("prob", 0) < threshold and not np.isnan(d.get("border_energy", np.nan))]
    todas = [d["border_energy"] for d in detalle if not np.isnan(d.get("border_energy", np.nan))]

    def _media(xs):
        return float(np.mean(xs)) if xs else float("nan")

    media_det = _media(detecciones)
    return {
        "baseline_uniforme": round(base, 4),
        "umbral_operativo": round(float(threshold), 4),
        "marco_evaluado": f"{int(BORDER_FRAC * 100)}% exterior (~{base:.0%} del área total)",
        "detecciones_positivas": {
            "n": len(detecciones),
            "energia_bordes_media": round(media_det, 4),
            "veredicto": (
                "atención centrada en el pulmón" if media_det < base * 0.75
                else "difusa" if media_det < base
                else "ATAJO ESPURIO: atención en bordes"
            ) if detecciones else "sin detecciones positivas en la muestra",
        },
        "resto_de_casos": {
            "n": len(resto),
            "energia_bordes_media": round(_media(resto), 4),
            "nota": "mapa sin señal; energía alta esperada, no indica atajo",
        },
        "mapas_nulos": sum(1 for d in detalle if np.isnan(d.get("border_energy", np.nan))),
        "energia_bordes_media_global": round(_media(todas), 4),
    }


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpone el mapa de calor (jet) sobre la radiografía."""
    import cv2

    cam_resized = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * image_rgb + alpha * heat, 0, 255).astype(np.uint8)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Tensor normalizado -> uint8 RGB, para pintar debajo del mapa de calor."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def explain_image(model, image_path: str, img_size: int = 224, device=None) -> dict:
    """Grad-CAM de una sola imagen. Devuelve cam, probabilidad y diagnóstico de atajo."""
    device = device or get_device()
    model = model.to(device)
    tf = build_transforms(img_size, train=False)
    raw = load_image(image_path)
    tensor = tf(image=raw)["image"].unsqueeze(0).to(device)

    with GradCAM(model) as cam_fn:
        cam, prob = cam_fn(tensor)

    return {
        "image_path": str(image_path),
        "prob": prob,
        "cam": cam,
        "input_rgb": denormalize(tensor[0]),
        "border_energy": border_energy_fraction(cam),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera mapas Grad-CAM y audita shortcut learning")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--image", default=None, help="Explica una única imagen")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12, help="Número de imágenes a explicar")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not args.manifest and not args.image:
        ap.error("Indica --manifest o --image")

    seed_everything(42)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint(args.checkpoint, map_location=device)
    img_size = ckpt.get("config", {}).get("img_size", 224)

    if args.image:
        rows = [{"image_path": args.image, "label": -1}]
    else:
        df = pd.read_csv(args.manifest)
        df = df if args.split == "all" else df[df["split"] == args.split]
        # Muestra equilibrada de positivos y negativos
        half = max(1, args.n // 2)
        pos = df[df["label"] == 1].sample(n=min(half, (df["label"] == 1).sum()), random_state=42)
        neg = df[df["label"] == 0].sample(n=min(args.n - len(pos), (df["label"] == 0).sum()), random_state=42)
        rows = pd.concat([pos, neg]).to_dict("records")

    results = []
    for row in rows:
        r = explain_image(model, row["image_path"], img_size=img_size, device=device)
        r["label"] = int(row.get("label", -1))
        results.append(r)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
        axes[0].imshow(r["input_rgb"])
        axes[0].set_title("radiografía")
        axes[1].imshow(overlay_cam(r["input_rgb"], r["cam"]))
        axes[1].set_title(f"Grad-CAM | p(neumonía) = {r['prob']:.3f}")
        for ax in axes:
            ax.axis("off")
        etiqueta = {0: "NORMAL", 1: "NEUMONÍA"}.get(r["label"], "sin etiqueta")
        fig.suptitle(f"verdad: {etiqueta} | energía en bordes: {r['border_energy']:.2f}")
        fig.tight_layout()
        fig.savefig(out_dir / f"gradcam_{Path(r['image_path']).stem}.png", dpi=140)
        plt.close(fig)

    detalle = [
        {"image_path": r["image_path"], "label": r["label"], "prob": r["prob"],
         "border_energy": r["border_energy"]}
        for r in results
    ]
    threshold = float(ckpt.get("threshold", 0.5))
    resumen = {"n_imagenes": len(results), **resumen_atajos(detalle, threshold), "detalle": detalle}
    save_json(resumen, out_dir / "shortcut_audit.json")

    det = resumen["detecciones_positivas"]
    print(f"\n{len(results)} mapas Grad-CAM guardados en {out_dir}")
    print(f"Baseline de un mapa uniforme: {resumen['baseline_uniforme']:.3f}")
    print(f"Detecciones positivas (n={det['n']}): energía en bordes {det['energia_bordes_media']:.3f}"
          f"  -> {det['veredicto']}")
    print(f"Resto de casos (n={resumen['resto_de_casos']['n']}): "
          f"{resumen['resto_de_casos']['energia_bordes_media']:.3f} "
          f"(sin señal, no informa sobre atajos)")
    if resumen["mapas_nulos"]:
        print(f"{resumen['mapas_nulos']} mapas nulos (negativos con predicción muy baja).")


if __name__ == "__main__":
    main()
