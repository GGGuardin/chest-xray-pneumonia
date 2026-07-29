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
| **Grad-CAM + métrica de energía en bordes** | Un mapa de calor bonito no prueba nada. Se cuantifica qué fracción de la atención cae en el marco exterior, **solo sobre las detecciones positivas** y contra el baseline de un mapa uniforme (0,51): en los negativos el mapa es ruido y contamina la media. |
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

Dos modelos DenseNet-121 entrenados de forma independiente en una T4 de Kaggle:
uno en **RSNA** (66 min, *early stopping* en la época 7 conservando la 3) y otro
en **NIH ChestX-ray14** (84 min, época 5 conservando la 2). Artefactos completos
en [`results/`](results).

**Modelo A — entrenado en RSNA** (opacidad pulmonar anotada por radiólogos)

| Conjunto | n | AUROC (IC95%) | AUPRC | Sens. | Esp. |
|---|---|---|---|---|---|
| **Test interno** (RSNA, split por paciente) | 3.812 | **0,885** (0,872–0,897) | 0,718 | 0,776 | 0,813 |
| Externo — NIH ChestX-ray14 | 112.120 | 0,727 (0,714–0,740) | 0,035 | 0,542 | 0,787 |
| Externo — pediátrico (Guangzhou) | 5.856 | 0,922 (0,914–0,930) | 0,969 | **0,362** | 0,992 |

**Modelo B — entrenado en NIH** (neumonía extraída por NLP de informes)

| Conjunto | n | AUROC (IC95%) | AUPRC | Sens. | Esp. |
|---|---|---|---|---|---|
| Test interno (NIH, split por paciente) | 16.703 | 0,720 (0,688–0,754) | 0,035 | 0,590 | 0,736 |
| Externo — pediátrico (Guangzhou) | 5.856 | **0,524** (0,508–0,539) | 0,789 | 0,508 | 0,523 |

**Umbral de éxito del proyecto:** AUROC > 0,85 en test con split por paciente —
**cumplido por el modelo A**, con el intervalo de confianza entero por encima.
El modelo B se queda en 0,720, por debajo de la referencia de CheXNet (0,768)
para esta misma etiqueta y dataset.

### Suelo de ruido: tres semillas

Sin esto, ninguna diferencia entre modelos es interpretable. Tres entrenamientos
idénticos salvo la inicialización y el orden de los lotes, mismo split:

| Métrica | Valores | Media | Desv. típica | Rango |
|---|---|---|---|---|
| AUROC (test RSNA) | 0,8810 · 0,8895 · 0,8851 | **0,8852** | **0,0043** | 0,0085 |
| AUPRC (test RSNA) | 0,7057 · 0,7177 · 0,7172 | 0,7135 | 0,0068 | 0,0120 |

**Cualquier diferencia de AUROC menor que ~0,009 en este montaje es ruido de
inicialización.** El 0,885 del modelo de referencia es representativo, no un
golpe de suerte.

### El resultado más incómodo: entrenar en NIH no aportó ventaja medible

Comparación **pareada** sobre las mismas 16.703 imágenes del test retenido de NIH,
con test de DeLong y las tres semillas ([`results/comparativa/`](results/comparativa)):

| Evaluado sobre el test de NIH (n=16.703, 222 positivos) | AUROC | p (DeLong) |
|---|---|---|
| Modelo **B**, entrenado *en* NIH con 78.000 imágenes | 0,7202 | — |
| Modelo **A**, semilla 42 | 0,7316 | 0,44 |
| Modelo **A**, semilla 1337 | 0,7369 | 0,21 |
| Modelo **A**, semilla 2024 | 0,7300 | 0,50 |
| **Modelo A, media** | **0,7328** | no significativa |

Un modelo que **nunca vio una sola imagen de NIH** rinde igual que uno entrenado
con sus 78.000. **Las tres semillas superan al modelo B**, así que la dirección
del efecto es consistente; pero ninguna comparación alcanza significación, de modo
que la afirmación defendible es *empate*, no *"A gana"*.

Y hay un detalle metodológico que merece la pena porque señala qué haría falta
para resolverlo:

| Fuente de ruido | Magnitud |
|---|---|
| Dispersión entre semillas | 0,0036 |
| **Error estándar de muestreo (DeLong)** | **0,0146** |

El ruido de muestreo es **cuatro veces mayor** que el de inicialización. El factor
limitante no son las semillas ni el entrenamiento: son los **222 positivos** que
tiene el test de NIH con una prevalencia del 1,3%. Para zanjar la cuestión no hace
falta entrenar mejor, hace falta un conjunto de evaluación con más casos positivos.

La conclusión sustantiva se sostiene igual: 78.000 imágenes etiquetadas por NLP no
aportaron ventaja medible sobre transferir desde 18.000 anotadas por radiólogos.

Al salir de ambos dominios, en cambio, la diferencia es aplastante y sí
significativa:

| Conjunto pediátrico (n=5.856) | AUROC |
|---|---|
| Modelo A (RSNA) | 0,922 |
| Modelo B (NIH) | 0,524 — indistinguible del azar |
| **Diferencia** | **+0,399** (IC95% 0,384–0,414) · z = 52,4 · **p < 10⁻¹⁵** |

Y no es un artefacto del umbral: fijando ambos modelos a una especificidad común
del 90%, el A alcanza sensibilidad **0,808** y el B **0,225**.

La explicación más plausible es la calidad de la etiqueta: RSNA marca *opacidad
pulmonar* delimitada visualmente por radiólogos; NIH marca *neumonía* inferida de
texto libre por un sistema NLP con una tasa de error estimada por encima del 10% y
una prevalencia del 1,3%. El modelo A aprendió algo que viaja; el B aprendió algo
que solo existe dentro de NIH.

### El fallo del conjunto pediátrico era calibración, no discriminación

AUROC 0,922 con sensibilidad 0,362 parece una contradicción, y tiene una causa
concreta: el umbral se fijó en una población con **22,6%** de prevalencia y se
aplicó a otra con **73,0%**. Corrigiendo ese desplazamiento de a priori con la
fórmula de Elkan —reescalar las odds por el cociente de odds a priori, una línea
de código y ni un gramo de reentrenamiento:

| Métrica | Sin corregir | Corregido |
|---|---|---|
| Sensibilidad | 0,362 | **0,950** |
| Especificidad | 0,992 | 0,613 |
| F1 | 0,531 | **0,908** |
| Brier | 0,237 | **0,113** |
| ECE | 0,335 | **0,094** |
| AUROC | 0,922 | 0,922 *(no cambia: la transformación es monótona)* |

El modelo pasa de perderse el 64% de las neumonías a detectar el 95%. **La
capacidad de discriminación siempre estuvo ahí; lo que estaba roto era la escala
de probabilidad.** Que el AUROC no se mueva un ápice mientras la sensibilidad casi
se triplica es la demostración más limpia de por qué una sola métrica no basta.

El matiz que impide vender esto como solución: la corrección **exige conocer la
prevalencia de destino**, y en la práctica clínica esa cifra rara vez está
disponible de antemano. Además, ni siquiera en su propio test el modelo está bien
calibrado (ECE 0,138), efecto esperable del `pos_weight` que infla las
probabilidades.

### Los tres hallazgos que importan

**1. La generalización entre hospitales se rompe.** Al evaluar en NIH el AUROC cae
0,157, muy por encima del umbral de alarma de 0,10 que fija este proyecto. Es el
efecto que documentaron Zech et al. (2018) reproducido de forma independiente.
Parte de la caída es desplazamiento de etiqueta —RSNA marca *opacidad pulmonar*
anotada por radiólogos y NIH marca *neumonía* extraída por NLP—, pero no todo.

**2. Un AUROC alto puede esconder un modelo inservible.** En el conjunto
pediátrico el AUROC *sube* a 0,922, y sin embargo la sensibilidad al umbral de
operación es 0,362: **el modelo se pierde el 64% de las neumonías** (2.726 falsos
negativos). La discriminación transfirió; la calibración no. El AUROC es
independiente del umbral, la utilidad clínica no lo es. Ninguna métrica sola
habría revelado esto: hace falta mirar sensibilidad y matriz de confusión junto
al AUROC.

**3. El sesgo grave no es demográfico, es técnico — y se replica.**

Tasa de falsos negativos (infradiagnóstico) por proyección, en los dos modelos
entrenados por separado sobre datasets distintos:

| Modelo | FNR en PA | FNR en AP | Brecha |
|---|---|---|---|
| A (RSNA) | 0,553 | 0,138 | **0,415** |
| B (NIH) | 0,615 | 0,267 | **0,348** |

| Atributo (modelo A) | Brecha de FNR | Peor subgrupo |
|---|---|---|
| **Proyección AP/PA** | **0,415** | PA: FNR 0,553 · AP: FNR 0,138 |
| Edad × sexo | 0,260 | Hombres 60-74 (FNR 0,337) |
| Edad | 0,253 | 60-74 (FNR 0,333) |
| Sexo | 0,003 | sin sesgo apreciable |

Por sexo no hay brecha en ninguno de los dos (0,003 y 0,031). La enorme es la
proyección, **en la misma dirección y con magnitud parecida en ambos modelos**:
en PA (pacientes ambulantes, baja prevalencia) se pierden más de la mitad de los
casos; en AP (portátil, pacientes encamados y más graves) capturan mucho más a
costa de disparar los falsos positivos. Que el patrón se replique en dos datasets
y dos entrenamientos independientes descarta que sea casualidad de un ajuste
concreto: **los modelos usan la geometría de adquisición como sustituto de la
gravedad del paciente.**

### Grad-CAM: la atención sí cae en el pulmón

Auditoría sobre **400 imágenes** del test (186 detecciones positivas), que es lo
que permite afirmar algo:

| Grupo | n | Energía en bordes |
|---|---|---|
| **Detecciones positivas** (p ≥ umbral) | 186 | **0,266** |
| Resto de casos | 206 | 0,741 |
| Mapas nulos | 8 | — |
| *Baseline de un mapa uniforme* | — | *0,510* |
| *Media global, sin separar grupos* | 400 | *0,516* |

Cuando el modelo da una imagen por positiva, la atención se concentra en la zona
central: **0,266 frente al 0,510** de un mapa sin estructura, sobre 186 casos.
Inspeccionando los mapas se ve que ignora electrodos de ECG, cables y marcadores
de lateralidad.

Fíjate en la última fila, que es la lección metodológica: **la media global es
0,516, prácticamente idéntica al baseline uniforme.** Quien reportase solo ese
número concluiría que el modelo no localiza nada — exactamente la conclusión
contraria a la correcta. El promedio queda arrastrado por los 206 casos negativos,
donde el Grad-CAM de la clase positiva no tiene nada que señalar y su energía en
bordes es alta por construcción. Por eso `explain.py` separa ambos grupos y compara
contra el baseline uniforme en lugar de contra un umbral inventado.

En la primera versión esta auditoría se hizo con 16 imágenes y 4 detecciones, lo
que no sostenía ninguna afirmación; con 186 sí.

Un detalle metodológico que costó descubrir: **promediar todas las imágenes juntas
da 0,488 y simula un atajo espurio inexistente.** En los negativos bien
clasificados el Grad-CAM de la clase positiva es todo ceros o ruido difuso —no
tiene nada que señalar— y su energía en bordes es alta por construcción. La
pregunta con sentido es *cuando el modelo cree ver algo, ¿dónde mira?*, y por eso
`explain.py` separa ambos grupos y compara contra el baseline uniforme en vez de
contra un umbral arbitrario.

### Qué se publica y qué no

`results/` contiene métricas, curvas, matrices de confusión y el análisis de
subgrupos. **No incluye los mapas Grad-CAM**, porque llevan superpuestas
radiografías reales de RSNA y las reglas de la competición no permiten
redistribuir las imágenes. Se pueden ver ejecutando `src/explain.py` con los datos
descargados, o en el notebook de Kaggle, que es donde su visualización está
amparada.

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
