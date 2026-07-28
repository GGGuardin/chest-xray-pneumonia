"""Demo Gradio: sube una radiografía y obtén probabilidad + mapa Grad-CAM.

Pensada para desplegarse en Hugging Face Spaces (tier CPU gratuito).

Ejecutar en local:
    python app/app.py --checkpoint runs/rsna_densenet121/best.pth
En Spaces: copia `best.pth` junto a este fichero y define CHECKPOINT=best.pth.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import border_energy_fraction, explain_image, overlay_cam  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

DESCARGO = """
> ### ⚠️ NO ES UNA HERRAMIENTA DIAGNÓSTICA
> Proyecto **educativo y experimental**. El modelo no ha sido validado
> clínicamente, no es un dispositivo médico y **no debe usarse para ninguna
> decisión sanitaria**. Entrenado con datasets públicos de-identificados;
> su rendimiento cae fuera de esa distribución. **No subas radiografías de
> personas reales identificables.**
"""

EXPLICACION = """
El mapa de calor (Grad-CAM) muestra dónde mira el modelo. Si la atención cae en
los bordes, esquinas, marcadores de lateralidad o texto quemado en vez de en el
parénquima pulmonar, la predicción se apoya en un atajo espurio y no en la
patología (DeGrave, Janizek & Lee, *Nature Machine Intelligence*, 2021).
"""


def build_interface(checkpoint_path: str):
    import gradio as gr

    model, ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = cfg.get("img_size", 224)
    threshold = float(ckpt.get("threshold", 0.5))

    def predecir(imagen: np.ndarray | None):
        if imagen is None:
            return None, "Sube una radiografía de tórax.", ""
        # Gradio entrega un array RGB: se persiste para reutilizar el mismo
        # camino de lectura que el resto del pipeline.
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(imagen.astype(np.uint8)).save(tmp.name)
            ruta = tmp.name
        try:
            r = explain_image(model, ruta, img_size=img_size)
        finally:
            os.unlink(ruta)

        prob = r["prob"]
        etiqueta = "OPACIDAD / NEUMONÍA" if prob >= threshold else "SIN OPACIDAD"
        borde = border_energy_fraction(r["cam"])
        aviso = ""
        if not np.isnan(borde) and borde > 0.35:
            aviso = ("\n\n⚠️ Gran parte de la atención cae en los bordes de la imagen: "
                     "posible atajo espurio, interpreta la predicción con cautela.")
        texto = (
            f"**p(opacidad pulmonar) = {prob:.3f}**\n\n"
            f"Clasificación al umbral operativo {threshold:.3f}: **{etiqueta}**\n\n"
            f"Energía del Grad-CAM en el marco exterior: {borde:.2f}{aviso}"
        )
        return overlay_cam(r["input_rgb"], r["cam"]), texto, {"normal": 1 - prob, "neumonía": prob}

    with gr.Blocks(title="Detector experimental de neumonía en radiografía de tórax") as demo:
        gr.Markdown("# Detector experimental de neumonía en radiografía de tórax")
        gr.Markdown(DESCARGO)
        with gr.Row():
            with gr.Column():
                entrada = gr.Image(type="numpy", label="Radiografía de tórax (PA/AP)")
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column():
                salida_img = gr.Image(label="Grad-CAM: dónde mira el modelo")
                salida_lbl = gr.Label(label="Probabilidades", num_top_classes=2)
                salida_txt = gr.Markdown()
        gr.Markdown(EXPLICACION)
        gr.Markdown(
            f"**Modelo:** {cfg.get('arch', 'densenet121')} · "
            f"AUROC de validación {ckpt.get('val_auroc', float('nan')):.3f} · "
            f"entrenado con `{Path(str(cfg.get('manifest', 'n/d'))).name}`"
        )
        boton.click(predecir, inputs=entrada, outputs=[salida_img, salida_txt, salida_lbl])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", "best.pth"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"No encuentro el checkpoint '{args.checkpoint}'. Entrena primero con "
            f"`python -m src.train --config configs/rsna.yaml`."
        )
    build_interface(args.checkpoint).launch(share=args.share)


if __name__ == "__main__":
    main()
