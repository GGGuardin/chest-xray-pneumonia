# Detección de neumonía en radiografía de tórax — proyecto de portafolio

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GGGuardin/chest-xray-pneumonia/blob/main/notebooks/00_colab_entrenamiento.ipynb)

> ## ⚠️ NO ES UNA HERRAMIENTA DIAGNÓSTICA
> Proyecto **educativo y experimental**. El modelo no está validado clínicamente,
> no es un dispositivo médico y **no debe usarse para ninguna decisión sanitaria**.
> Sigue la misma línea que `torchxrayvision` ("NOT FOR MEDICAL USE") y RAD-DINO
> ("It is not meant to be used for clinical practice").

Clasificación **binaria** (opacidad pulmonar compatible con neumonía sí/no) sobre
radiografías de tórax, mediante *fine-tuning* de una DenseNet-121 preentrenada.

Lo que este repositorio intenta demostrar no es un número alto de *accuracy*, sino
**rigor metodológico**: split por paciente, métricas adecuadas al desbalance,
verificación de que el modelo mira el pulmón, validación externa y análisis de
sesgo por subgrupos.

---

## Decisiones de diseño (y por qué)

| Decisión | Motivo |
|---|---|
| **Split por paciente**, nunca por imagen | Un paciente puede tener varias radiografías; dividir por imagen mete al mismo paciente en train y test y el modelo lo memoriza. `assert_no_patient_leakage()` aborta la ejecución si ocurre. |
| **AUROC / AUPRC** como métricas principales | Con 75-80% de negativos, predecir siempre "normal" da una accuracy altísima e inútil. La accuracy se reporta, pero nunca como métrica de decisión. |
| **Umbral fijado en validación** (índice de Youden) | Elegir el umbral mirando el test es una fuga de información sutil pero real. |
| **Sin flip horizontal** en el *augmentation* | El corazón está a la izquierda del paciente; voltear la imagen crea una anatomía de *situs inversus* (~1:10.000) y destruye la información lateral en la que se apoya, por ejemplo, la cardiomegalia. |
| **`pos_weight` en la loss** | Compensa el desbalance sin descartar datos. Alternativa disponible: sampler ponderado (`balance_train`). |
| **Grad-CAM + métrica de energía en bordes** | Un mapa de calor bonito no prueba nada. Se cuantifica qué fracción de la atención cae en el marco exterior: si es alta, hay indicios de *shortcut learning* (marcadores, texto quemado, bordes). |
| **Validación externa obligatoria** | Un modelo entrenado en un hospital cae al evaluarse en otro. Si el AUROC baja más de 0,10, eso es el resultado y hay que analizarlo, no esconderlo. |
| **Análisis de subgrupos (FNR)** | El infradiagnóstico selectivo en poblaciones desatendidas está documentado; medirlo es parte del trabajo, no un extra. |

---

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sin GPU local, el entrenamiento real conviene hacerlo en **Google Colab** (T4
gratuita) o **Kaggle Notebooks** (~30 h/semana de GPU). El código detecta el
dispositivo automáticamente y activa *mixed precision* solo en CUDA.

### Comprobación en 2 minutos, sin descargar ningún dataset

```powershell
python tests/smoke_test.py
```

Genera un dataset sintético, construye el manifiesto, entrena 2 épocas, evalúa,
produce Grad-CAM y el análisis de subgrupos. Si esto pasa, el pipeline funciona.

---

## Uso

### 1. Datos y manifiesto

Instrucciones de descarga y licencias en [`data/README.md`](data/README.md).

```powershell
python -m src.prepare_data --dataset rsna --root data/raw/rsna --out data/manifest_rsna.csv
```

Salida típica:

```
split  imagenes  pacientes  positivos  prevalencia
train     18679      18679       5341       0.2860
val        4002       4002       1144       0.2859
test       4003       4003       1144       0.2858
```

### 2. Entrenamiento

```powershell
python -m src.train --config configs/rsna.yaml
```

Guarda en `runs/<experimento>/`: `best.pth` (mejor AUROC de validación, con el
umbral de Youden dentro), `history.csv` y `train_summary.json`.

### 3. Evaluación en test

```powershell
python -m src.evaluate --checkpoint runs/rsna_densenet121/best.pth `
    --manifest data/manifest_rsna.csv --split test --out-dir reports/rsna_test
```

Produce `metrics.json` (con IC95% por bootstrap), `predictions.csv`, `curvas.png`
(ROC + PR) y `matriz_confusion.png`.

### 4. Explicabilidad y auditoría de atajos

```powershell
python -m src.explain --checkpoint runs/rsna_densenet121/best.pth `
    --manifest data/manifest_rsna.csv --split test --n 12 --out-dir reports/gradcam
```

`shortcut_audit.json` incluye la energía media del CAM en el marco exterior de la
imagen. **Señal de alarma:** si supera ~0,35, el modelo está mirando fuera del
pulmón.

### 5. Validación externa

```powershell
python -m src.evaluate --checkpoint runs/rsna_densenet121/best.pth `
    --manifest data/manifest_kaggle.csv --split all --out-dir reports/externo_kaggle
```

**Criterio:** una caída de AUROC > 0,10 respecto al test interno indica un
problema de generalización que hay que documentar y analizar (distinta población,
distinto equipo, distinta prevalencia, proyección AP vs PA).

### 6. Análisis de sesgo

```powershell
python -m src.fairness --predictions reports/rsna_test/predictions.csv --out-dir reports/fairness
```

Tabla de **FNR (infradiagnóstico) por sexo, grupo de edad, proyección** y sus
intersecciones, más la brecha máxima entre subgrupos.

### 7. Demo

```powershell
python app/app.py --checkpoint runs/rsna_densenet121/best.pth
```

Interfaz Gradio con imagen de entrada, probabilidad, Grad-CAM y descargo de
responsabilidad visible. Desplegable en Hugging Face Spaces (CPU gratuito):
sube `app/app.py`, `src/`, `requirements.txt` y `best.pth`, y define
`CHECKPOINT=best.pth`.

---

## Estructura

```
.
├── configs/            # hiperparámetros por experimento (YAML)
├── data/               # solo instrucciones de descarga; NUNCA imágenes
├── src/
│   ├── data.py         # manifiestos, split POR PACIENTE, Dataset, transforms
│   ├── model.py        # transfer learning (DenseNet-121 / timm)
│   ├── metrics.py      # AUROC, AUPRC, sens/esp, bootstrap, gráficas
│   ├── prepare_data.py # CLI: dataset -> manifiesto + split
│   ├── train.py        # bucle de entrenamiento
│   ├── evaluate.py     # test e evaluación externa
│   ├── explain.py      # Grad-CAM + auditoría de shortcut learning
│   └── fairness.py     # métricas por subgrupo (FNR)
├── app/app.py          # demo Gradio
└── tests/              # datos sintéticos + prueba de humo end-to-end
```

---

## Resultados

<!-- Rellenar tras el primer entrenamiento completo -->

| Conjunto | n | AUROC (IC95%) | AUPRC | Sens. | Esp. |
|---|---|---|---|---|---|
| Test interno (RSNA) | — | — | — | — | — |
| Externo (Kaggle pneumonia) | — | — | — | — | — |

**Umbral de éxito del proyecto:** AUROC > 0,85 en test con split por paciente.

---

## Limitaciones (léelas antes que los resultados)

- **Las etiquetas no son verdad clínica.** En RSNA proceden de anotaciones de
  radiólogos sobre opacidad pulmonar, no de un diagnóstico confirmado de neumonía;
  en NIH/CheXpert/MIMIC se extrajeron de informes por NLP, con una tasa de error
  estimada por encima del 10%.
- **Una opacidad no es una neumonía.** El propio RSNA separa "No Lung Opacity /
  Not Normal": hay anormalidad sin opacidad. Este proyecto detecta un patrón
  radiológico, no una enfermedad.
- **Generalización limitada.** Cambian el equipo, el protocolo, la población y la
  prevalencia. La validación externa forma parte del reporte precisamente por eso.
- **Atajos espurios.** Se auditan con Grad-CAM, pero la ausencia de evidencia de
  atajo no es evidencia de ausencia: modelos de neumotórax han aprendido a
  detectar los drenajes torácicos, no el hallazgo.
- **Sesgo demográfico.** Se mide, no se corrige. Un FNR más alto en un subgrupo
  significa que ese subgrupo recibiría menos atención.
- **Sin cabecera clínica.** Sin edad de presentación, síntomas, analítica ni
  historia previa, ninguna decisión médica real se toma con una imagen aislada.

---

## Contexto regulatorio (por qué esto no es un producto)

En EE. UU. el software radiológico con IA se regula como *Software as a Medical
Device*: las herramientas de triaje asistido (CADt) son Clase II (21 CFR 892.2080)
y acceden al mercado por 510(k) o De Novo. En la UE aplica el marcado CE bajo el
MDR (Reglamento UE 2017/745), regla 11 del anexo VIII — típicamente Clase IIa —
con organismo notificado, sistema de gestión de calidad, IEC 62304 y vigilancia
poscomercialización. **Nada de esto se ha hecho aquí**, y por eso este trabajo se
queda explícitamente del lado del experimento.

Sobre los datos: usa solo datasets públicos ya de-identificados. La
de-identificación de un DICOM se rige por DICOM PS3.15 anexo E y, en EE. UU., por
el método *Safe Harbor* de HIPAA (45 CFR §164.514(b), 18 categorías de
identificadores), que incluye el texto quemado en los píxeles. `pydicom` edita
*tags*, pero no elimina PHI de la imagen.

---

## Referencias

- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225
- Zech, Badgeley, Liu, Costa, Titano & Oermann (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine. doi:10.1371/journal.pmed.1002683
- Raghu, Zhang, Kleinberg & Bengio (2019). *Transfusion: Understanding Transfer Learning for Medical Imaging.* NeurIPS. arXiv:1902.07208
- DeGrave, Janizek & Lee (2021). *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence 3:610-619. doi:10.1038/s42256-021-00338-7
- Seyyed-Kalantari, Zhang, McDermott, Chen & Ghassemi (2021). *Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations.* Nature Medicine 27:2176-2182. doi:10.1038/s41591-021-01595-0
- Cohen et al. (2022). *TorchXRayVision: A library of chest X-ray datasets and models.* MIDL. arXiv:2111.00595
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV

## Licencia

Código bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los datasets mantienen sus
propias licencias, que **no** se ven afectadas por esta.
