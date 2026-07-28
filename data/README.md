# Datos

**En este directorio no se versiona ninguna imagen.** `.gitignore` excluye
`data/raw/` y los manifiestos. Nunca subas radiografías de personas reales
identificables a un repositorio público — ni siquiera de tu familia: un DICOM
lleva PHI en los metadatos (nombre, fecha de nacimiento, ID, institución) y a
veces texto "quemado" en los propios píxeles.

## Descarga

Necesitas la CLI de Kaggle y un token de API (`kaggle.json`, en
`C:\Users\<usuario>\.kaggle\`):

```powershell
pip install kaggle
```

### Opción A — RSNA Pneumonia Detection Challenge (recomendada)

26.684 radiografías en DICOM, con `Target` binario y metadatos de sexo, edad y
proyección AP/PA en las cabeceras — lo que habilita el análisis de subgrupos.
Hay que aceptar las reglas de la competición en la web antes de descargar.

```powershell
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/raw/rsna
Expand-Archive data/raw/rsna/rsna-pneumonia-detection-challenge.zip -DestinationPath data/raw/rsna
python -m src.prepare_data --dataset rsna --root data/raw/rsna --out data/manifest_rsna.csv
```

### Opción B — Chest X-Ray Images (Pneumonia)

~5.860 imágenes JPEG, población pediátrica de un único hospital (Guangzhou).
Más ligero y sin DICOM, pero generaliza mal: úsalo sobre todo como conjunto
**externo** contra un modelo entrenado en RSNA.

```powershell
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw
Expand-Archive data/raw/chest-xray-pneumonia.zip -DestinationPath data/raw
python -m src.prepare_data --dataset kaggle_pneumonia --root data/raw/chest_xray --out data/manifest_kaggle.csv
```

Los splits originales de este dataset (`train/val/test`) se **ignoran**: el `val`
oficial tiene 16 imágenes y no garantizan separación por paciente. `prepare_data`
los rehace agrupando por paciente.

### Sin descargar nada — datos sintéticos

Para probar el pipeline entero sin datasets reales:

```powershell
python tests/make_synthetic_data.py --out data/raw/synthetic
python -m src.prepare_data --dataset folder --root data/raw/synthetic --out data/manifest_synthetic.csv
```

## Licencias

| Dataset | Condiciones |
|---|---|
| RSNA Pneumonia Detection Challenge | Uso académico/no comercial con atribución; sujeto a las reglas de la competición de Kaggle |
| Chest X-Ray Images (Pneumonia) | CC BY 4.0 (Kermany, Zhang & Goldbaum, Mendeley Data) |
| NIH ChestX-ray14 | Abierto, con atribución al NIH Clinical Center |
| CheXpert / MIMIC-CXR / BRAX / VinDr-CXR | Requieren registro, DUA y (PhysioNet) curso CITI. **Prohibido redistribuir las imágenes o pasarlas por APIs de terceros.** |

Revisa siempre las condiciones antes de publicar un modelo entrenado o cualquier
imagen derivada.

## Formato del manifiesto

`prepare_data` normaliza cualquier dataset a un CSV único:

| columna | descripción |
|---|---|
| `image_path` | ruta absoluta o relativa al fichero (DICOM/PNG/JPEG) |
| `patient_id` | identificador de paciente — **la unidad del split** |
| `label` | 0 = normal / sin opacidad, 1 = neumonía / opacidad pulmonar |
| `source` | dataset de origen |
| `view` | proyección (AP/PA/UNKNOWN) |
| `sex` | M/F/UNKNOWN |
| `age` | edad en años (NaN si no consta) |
| `split` | train / val / test |
