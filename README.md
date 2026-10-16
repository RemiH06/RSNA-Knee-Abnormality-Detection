![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

```ascii
██╗  ██╗ █████╗ ██████╗
██║ ██╔╝██╔══██╗██╔══██╗
█████╔╝ ███████║██║  ██║
██╔═██╗ ██╔══██║██║  ██║
██║  ██╗██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
       by Hex (@RemiH06)          version 0.2.0
```

![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)

## Resumen

### Descripción general

KAD (Knee Abnormality Detection) es nuestra solución para la competencia [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) de Kaggle. Para cada estudio de resonancia magnética de rodilla predice la probabilidad de 12 hallazgos: ACL, MCL, menisco medial y lateral, osteoartritis medial, lateral y patelofemoral, derrame, sinovitis, quiste de Baker, contusión y fractura. La métrica es el promedio del ROC-AUC de los 12 hallazgos.

El reto principal no es el modelo sino las etiquetas: de 4,407 estudios de entrenamiento, solo 58 traen las 12 etiquetas. Los demás traen el reporte de radiología en texto libre (en varios idiomas), y el reporte no existe en test. Por eso el texto solo sirve para fabricar etiquetas: un LLM lee cada reporte y produce una probabilidad por hallazgo (etiquetas suaves), y un modelo de imagen aprende a reproducirlas. Los 58 estudios con etiqueta real quedan fuera del entrenamiento y sirven solo para validar.

El modelo es 2.5D: tres cortes vecinos de la resonancia forman los tres canales de una imagen, un CoAtNet preentrenado (384 px) extrae rasgos de cada ventana, y una capa de atención con pesos propios por hallazgo decide qué cortes mirar para cada uno. Cada estudio se reduce a 64 cortes en 5 grupos (sagital y coronal con y sin supresión de grasa, y axial), ordenados por posición física y recortados a 140 mm de anatomía.

```diff
- La competencia cierra el 2026-10-22; el score privado (final) se publica después.
- Los datos de la competencia no están en el repo (~265 GB comprimidos). Se bajan con los scripts, con una cuenta de Kaggle inscrita en la competencia.
- Validar contra 58 estudios es una regla corta: diferencias menores a ~0.02 de AUC no son medibles.
```

### Resultados

| | AUC macro |
|---|---|
| Leaderboard público de Kaggle, modelo propio solo (envío 57232135) | **0.919** |
| Validación contra los 58 estudios gold (v2, promedio de pesos de los epochs 8 a 10) | 0.892 (IC 95 % bootstrap: 0.856 a 0.922) |
| Etiquetas LLM usadas para entrenar, contra los mismos 58 estudios | 0.893 |

AUC macro contra los 58 estudios gold por epoch (v2, corpus de 64 cortes):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.695 | 0.795 | 0.812 | 0.853 | 0.848 | 0.865 | 0.881 | 0.887 | 0.890 | 0.888 | 0.886 | 0.884 |

Después del epoch 9 la pérdida sigue bajando pero el AUC gold baja: el modelo empieza a memorizar el ruido de las etiquetas.

### Notebooks

- [`notebooks/01_trayecto.ipynb`](notebooks/01_trayecto.ipynb): el trayecto completo, fase por fase, con el análisis estadístico multivariado (vector de medias, matrices de covarianza y correlación, tablas conjuntas, pruebas de razón de verosimilitudes, valores propios, máxima verosimilitud y bootstrap). Para la materia Análisis Estadístico Multivariable (ITESO).
- [`notebooks/02_version_definitiva.ipynb`](notebooks/02_version_definitiva.ipynb): el notebook de Kaggle que genera el envío final.
- [`notebooks/03_interpretacion.ipynb`](notebooks/03_interpretacion.ipynb): dónde mira el modelo. La atención por hallazgo se trata como una distribución de probabilidad sobre los cortes (marginales por tipo de serie, entropía, Mann-Whitney entre positivos y negativos) y Grad-CAM localiza la región dentro del corte. Insumos: `scripts/attention_maps.py`.

## Installation

El entrenamiento corre en Windows con una GPU NVIDIA (probado en una RTX 3060 de 12 GB) y Python 3.10.

1. Clonar el repo y crear un archivo `.env` en la raíz con el token de Kaggle (no se versiona):
   ```bash
   git clone https://github.com/RemiH06/RSNA-Knee-Abnormality-Detection.git
   cd RSNA-Knee-Abnormality-Detection
   echo KAGGLE_TOKEN=<tu token> > .env
   ```
2. Crear el entorno (venv con torch CUDA, timm, pydicom, kagglehub) y bajar el corpus preprocesado público y las etiquetas LLM (~22 GB):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_pixilate.ps1
   ```
3. Bajar el `archive.zip` completo de la competencia (265 GB, reanudable) y generar el corpus de 64 cortes (~22 min con 4 procesos):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_token.ps1 scripts\download_archive.py
   .venv\Scripts\python.exe scripts\build_corpus.py --workers 4
   ```
4. Entrenar (~10 h en una RTX 3060):
   ```powershell
   .venv\Scripts\python.exe scripts\train_raptor.py --resume --corpus data\corpus_v4 --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --norm imagenet --grad_ckpt --epochs 12 --bs 3 --micro_bs 1 --vram_gb 7 --k 12 --k_eval 24 --workers 4 --tag coatnet_v2 --labels <ruta a llm_labels_v4_blend.csv>
   ```

## Launch arguments

`scripts/train_raptor.py` (además de los del entrenador original):

- `--corpus <carpeta>` corpus con `all_vols.npy`, `all_masks.npy` y `all_ids.npy`; sin él usa el corpus público de 44 cortes.
- `--labels <archivo>` etiquetas suaves por estudio, en CSV o parquet.
- `--resume` sigue desde el último epoch cerrado (`runs/raptor_last_<tag>.pt`) tras un corte de luz o un reinicio.
- `--micro_bs <n>` procesa cada batch en partes de `n` estudios y acumula el gradiente: mismo resultado, menos VRAM.
- `--vram_gb <GB>` tope de VRAM del proceso; ante falta de memoria espera y reintenta en lugar de caerse.

`scripts/build_corpus.py`:

- `--workers <n>` procesos en paralelo.
- `--limit <n>` solo los primeros `n` estudios, para pruebas.
- `--out <carpeta>` destino del corpus (por defecto `data/corpus_v4`).

## Features

- Etiquetas débiles: etiquetas suaves derivadas de los reportes por LLM, comparadas contra los 58 estudios con etiqueta real antes de elegir el juego a usar.
- Preprocesamiento sin descomprimir: `build_corpus.py` lee los DICOM directo del zip de 265 GB, ordena los cortes por geometría (`ImagePositionPatient`), normaliza intensidad por serie y recorta por milímetros, no por píxeles.
- Modelo 2.5D con atención por hallazgo (multiple instance learning), con promedio de pesos de los mejores epochs.
- Entrenamiento tolerante: reanudación por epoch, tope de VRAM y reintento ante falta de memoria, para convivir con otros usos de la GPU.
- Combinación por rangos con los mejores modelos públicos de la competencia: como el AUC solo depende del orden, cada modelo se convierte a rangos antes de promediar.
- Basado en trabajo público de la comunidad: el entrenador y el preprocesamiento adaptan los notebooks de [dreaddevelopment](https://www.kaggle.com/dreaddevelopment) (Apache 2.0); las etiquetas son de [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (CC0).

## Future Features

- Segundo modelo propio (v3, pérdida ponderada por la confianza de cada etiqueta) como brazo extra de la combinación.
- Pesos por hallazgo en la combinación por rangos, según el AUC de cada modelo.

## Autoría

por Hex ([@RemiH06](https://github.com/RemiH06))
