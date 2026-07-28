import json
import sys

ruta = sys.argv[1] if len(sys.argv) > 1 else ".kaggle_out/nih-cxr14-densenet121.log"
with open(ruta, encoding="utf-8") as f:
    entradas = json.load(f)

for e in entradas:
    texto = (e.get("data") or "").rstrip()
    if texto:
        print(f"[{e.get('stream_name', '?')}] {texto}")
