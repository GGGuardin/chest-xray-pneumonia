"""Recalcula la auditoría de shortcut learning sobre un shortcut_audit.json ya generado.

La energía en bordes por imagen no cambia; lo que cambia es cómo se agrega.
Permite corregir informes antiguos sin volver a pasar el modelo por la GPU.

    python scripts/recalcular_atajos.py .kaggle_out/rsna/reports/gradcam/shortcut_audit.json 0.5831
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import resumen_atajos  # noqa: E402


def main() -> None:
    ruta = Path(sys.argv[1])
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)

    detalle = datos["detalle"]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else datos.get("umbral_operativo", 0.5)

    nuevo = {"n_imagenes": len(detalle), **resumen_atajos(detalle, threshold), "detalle": detalle}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(nuevo, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in nuevo.items() if k != "detalle"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
