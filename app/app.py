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

from src.calibration import corregir_por_prevalencia  # noqa: E402
from src.explain import (  # noqa: E402
    border_energy_fraction,
    explain_image,
    overlay_cam,
    uniform_baseline,
)
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

PREVALENCIA_AYUDA = """
### Por qué hay un control de prevalencia

Un clasificador no devuelve "cuánta enfermedad hay en esta imagen", sino
**cuánta probabilidad tiene dado lo que vio durante el entrenamiento** — y eso
incluye la frecuencia de la enfermedad en aquella población. Aplicarlo donde la
prevalencia es distinta desplaza todas las probabilidades y deja el umbral en el
sitio equivocado.

Este proyecto lo midió: el modelo alcanza **AUROC 0,922** sobre un conjunto
pediátrico con 73% de prevalencia y, aun así, al umbral heredado se pierde el
**64%** de las neumonías. Corrigiendo el a priori, la sensibilidad sube a
**0,950** — y el AUROC no cambia ni un decimal, porque la transformación es
monótona y no altera el orden.

Mueve el control y observa la consecuencia: **la capacidad de discriminar es la
misma, lo que cambia es dónde cae la frontera de decisión.** Ese es el punto.
Y el aviso realista: para corregir hay que *conocer* la prevalencia de destino,
que en la práctica clínica rara vez está disponible.
"""


def build_interface(checkpoint_path: str):
    import gradio as gr

    model, ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = cfg.get("img_size", 224)
    threshold = float(ckpt.get("threshold", 0.5))
    # Los checkpoints antiguos no la guardan; 0,226 es la de RSNA, con la que se
    # entrenó el modelo de referencia de este proyecto.
    prev_entrenamiento = float(ckpt.get("train_prevalence", 0.2259))

    def predecir(imagen: np.ndarray | None, prevalencia_pct: float):
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

        prob_cruda = r["prob"]
        prev_destino = max(0.001, min(0.999, float(prevalencia_pct) / 100.0))
        prob = float(corregir_por_prevalencia(
            np.array([prob_cruda]), prev_entrenamiento, prev_destino)[0])
        corregida = abs(prev_destino - prev_entrenamiento) > 1e-4

        positivo = prob >= threshold
        etiqueta = "OPACIDAD / NEUMONÍA" if positivo else "SIN OPACIDAD"
        borde = border_energy_fraction(r["cam"])
        base = uniform_baseline()

        # La energía en bordes solo informa cuando hay detección: en un negativo
        # el mapa de la clase positiva no señala nada y su energía es alta por
        # construcción, no por atajo espurio.
        if np.isnan(borde):
            nota = "Mapa vacío: el modelo no encuentra nada que señalar."
        elif not positivo:
            nota = (f"Energía en bordes {borde:.2f} (referencia {base:.2f}). "
                    "Sin detección positiva, este número no informa sobre atajos.")
        elif borde > base:
            nota = (f"⚠️ Energía en bordes {borde:.2f}, por encima de la referencia "
                    f"{base:.2f}: la atención se va al marco de la imagen en vez de al "
                    "pulmón. Interpreta la predicción con cautela.")
        else:
            nota = (f"Energía en bordes {borde:.2f} frente a {base:.2f} de un mapa sin "
                    "estructura: la atención se concentra en la zona central.")

        if corregida:
            cabecera = (
                f"**p(opacidad) = {prob:.3f}**  ·  sin corregir: {prob_cruda:.3f}\n\n"
                f"Ajustada de una prevalencia de entrenamiento del "
                f"{prev_entrenamiento:.1%} a la población indicada ({prev_destino:.1%})."
            )
        else:
            cabecera = (f"**p(opacidad pulmonar) = {prob:.3f}**\n\n"
                        f"Sin corrección: la prevalencia indicada coincide con la de "
                        f"entrenamiento ({prev_entrenamiento:.1%}).")

        texto = (
            f"{cabecera}\n\n"
            f"Clasificación al umbral operativo {threshold:.3f}: **{etiqueta}**\n\n"
            f"{nota}"
        )
        return overlay_cam(r["input_rgb"], r["cam"]), texto, {"normal": 1 - prob, "neumonía": prob}

    with gr.Blocks(title="Detector experimental de neumonía en radiografía de tórax") as demo:
        gr.Markdown("# Detector experimental de neumonía en radiografía de tórax")
        gr.Markdown(DESCARGO)
        with gr.Row():
            with gr.Column():
                entrada = gr.Image(type="numpy", label="Radiografía de tórax (PA/AP)")
                prevalencia = gr.Slider(
                    minimum=1, maximum=90, value=round(prev_entrenamiento * 100, 1), step=0.5,
                    label="Prevalencia esperada en la población (%)",
                    info=(f"Entrenado con {prev_entrenamiento:.1%}. Muévelo y las "
                          "probabilidades se reajustan al a priori de la población "
                          "donde apliques el modelo."),
                )
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column():
                salida_img = gr.Image(label="Grad-CAM: dónde mira el modelo")
                salida_lbl = gr.Label(label="Probabilidades", num_top_classes=2)
                salida_txt = gr.Markdown()
        gr.Markdown(EXPLICACION)
        gr.Markdown(PREVALENCIA_AYUDA)
        gr.Markdown(
            f"**Modelo:** {cfg.get('arch', 'densenet121')} · "
            f"AUROC de validación {ckpt.get('val_auroc', float('nan')):.3f} · "
            f"entrenado con `{Path(str(cfg.get('manifest', 'n/d'))).name}`"
        )
        boton.click(predecir, inputs=[entrada, prevalencia],
                    outputs=[salida_img, salida_txt, salida_lbl])
        # Recalcular al mover el control deja ver el efecto sin volver a pulsar
        prevalencia.release(predecir, inputs=[entrada, prevalencia],
                            outputs=[salida_img, salida_txt, salida_lbl])
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
