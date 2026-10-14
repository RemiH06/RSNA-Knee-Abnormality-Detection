ok, veamos, como queda poco tiempo y quiero hacer una buena entrega pero no puedo estar siempre en casa, usaremos la skill de Bakery para trabajar en esto.

# RSNA Knee Abnormality Detection — plan de trabajo

Todo lo de esta sección quedó verificado directo con la API de Kaggle (`kaggle competitions pages --content`) el 12 de septiembre de 2026, ya con la cuenta unida a la competencia. Reemplaza cualquier supuesto anterior sacado de fuentes de terceros.

## La competencia, confirmado

- **Tarea**: para cada estudio de MRI de rodilla, predecir la probabilidad de 12 hallazgos binarios: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- **Métrica oficial**: promedio macro del ROC-AUC entre los 12 targets (`Final Score = (1/12) * sum(AUC_i)`).
- **Formato de envío**: `submission.csv` con columnas `StudyInstanceUID` + las 12 etiquetas (probabilidad, no clase).
- **Es code competition**: el envío se hace vía Kaggle Notebook, límite de 9 horas (CPU o GPU), **sin acceso a internet durante el run**. Se permite usar datos y modelos preentrenados libres/públicos siempre que estén adjuntos como Kaggle Dataset (no se pueden llamar APIs externas porque no hay internet). Archivo de salida debe llamarse exactamente `submission.csv`.
- **Fecha límite final**: 2026-10-22 23:59 (confirmado vía API; una nota de prensa mencionaba el 15 de octubre pero esa parece ser la fecha de inscripción/entry, no la de envío final). Bolsa de 77,000 USD, 3,633 equipos inscritos al momento de revisar.
- **Punto clave que cambia la estrategia**: el reporte de radiología (`Report`) viene en `train.csv` pero **no** se provee en test (`test.csv` no trae esa columna). Es decir, el texto sirve solo como señal de entrenamiento (para derivar pseudo-etiquetas en los ~4,349 estudios sin etiqueta gold), pero el modelo final tiene que predecir a partir de imagen únicamente, porque en inferencia no hay texto disponible.

## Esquema real de los datos (ya descargado en `data/`)

- **`train.csv`** (4,407 filas): `StudyInstanceUID`, `Report` (texto libre, multilenguaje según el sitio: se vieron reportes en español y neerlandés/alemán en la muestra), y las 12 columnas de etiqueta.
  - Confirmado con pandas: **solo 58 de 4,407 estudios tienen las 12 etiquetas llenas**, el resto vienen vacías. Es decir, weak supervision real, no un supuesto de terceros. El 100% de las filas sí tienen `Report`.
- **`train_series.csv`** (24,371 filas): una fila por serie de MRI. Columnas: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane` (`Sagittal`/`Coronal`/`Axial`). Promedio de 5.5 series por estudio.
- **`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`**: un archivo por corte (slice). Series de 20 a 45 cortes típicamente (mediana 30), con cola larga de algunas hasta varios cientos. DICOMs con transfer syntax mixto (Explicit/Implicit VR Little Endian, JPEG Lossless, JPEG 2000) y solo 86 tags de metadata permitidos (el resto fue despojado).
- **`test.csv`**: solo 3 estudios de ejemplo en el set público; en el `test.csv` real (al momento de scoring) hay **~1,300 estudios**, y ahí no viene columna `Report`.
- **`test_series.csv`** / **`test_series/`**: mismo esquema que las versiones de train, con datos de ejemplo que Kaggle reemplaza por los reales al momento de evaluar.
- **`sample_submission.csv`**: plantilla válida con las 12 columnas en `0.5`.
- Aviso oficial de la competencia: la prevalencia de cada hallazgo puede variar entre train, leaderboard público y evaluación final.

## Tamaño del dataset y su implicación práctica

Medido el 2026-10-13 leyendo el directorio central del zip oficial (`archive.zip`, versión del 2026-09-19) por HTTP Range, sin descargarlo:

- `archive.zip`: **265 GB comprimido**. `train_series/`: **819,078 cortes, 569 GB descomprimido** (264 GB comprimido). Las 24,371 series tienen mediana de 30 cortes (p90 39, p99 160, máx 320).
- El servidor acepta `Range`, así que se puede leer cualquier miembro del zip sin bajar el resto. Listar archivo por archivo con la API de Kaggle da 429 (rate limit) a los ~100 requests: no sirve para bajar 819k archivos.
- Muestra aleatoria de 40 cortes: todos `Explicit VR Little Endian` (sin JPEG/J2K), `pydicom` los decodifica sin plugins. Resolución nativa variable, de 192 a 1024 px (mediana ~512, muchos de 384).
- Los 58 estudios gold: 336 series, 10,528 cortes, 3.4 GB comprimido.

Pixilate tiene **78.6 GB libres** (no 94): no cabe ni el zip ni el dataset descomprimido. Sí cabe redimensionado a uint8 (normalizado por serie con percentiles):

| Lado | Todos los cortes, uint8 sin comprimir |
|---|---|
| 224 px | 41 GB |
| 256 px | 54 GB |
| 320 px | 84 GB (no cabe sin compresión) |
| 384 px | 121 GB (no cabe) |

Plan de datos: un script en Pixilate lee el zip remoto por rangos serie por serie, decodifica en memoria, redimensiona a 256 px y guarda un `.npy`/`.npz` comprimido por serie. El crudo nunca toca disco, es reanudable (salta series ya hechas) y en paralelo con los 8 núcleos. Bajada medida en Pixilate: ~52 MB/s, unas 1.5 h para los 264 GB. El mismo preprocesamiento se reusa tal cual en el notebook de inferencia de Kaggle.

## Entorno local

- Python 3.13.3, pip, `kaggle` CLI 2.2.4 y `kagglehub` 1.0.2 instalados. Sin conda, sin GPU NVIDIA local.
- Token de Kaggle en `.env` (`KAGGLE_TOKEN`), pero el CLI espera la variable `KAGGLE_API_TOKEN` — hay que exportarla con ese nombre antes de usar `kaggle`.
- Cuenta ya unida a la competencia (`userHasEntered: True`), descargas por API funcionando.

## Fases

- [x] **Fase 0 — Acceso y entorno**: cuenta unida, token funcionando, CSVs chicos descargados (`sample_submission.csv`, `test.csv`, `test_series.csv`, `train.csv`, `train_series.csv`).
- [ ] **Fase 1 — EDA**: prevalencia real de cada uno de los 12 hallazgos entre los 58 casos gold, calidad/longitud/idiomas de los reportes en las 4,407 filas, relación entre `Fluid_Sensitive`/`Fat_Suppression`/`Anatomical_Plane` y qué series conviene mirar por hallazgo, inspección visual de DICOMs de al menos algunos de los 58 casos gold.
- [ ] **Fase 2 — Derivar pseudo-etiquetas del reporte**: como el texto no está en test, su único uso es generar labels ruidosas para los 4,349 estudios sin gold label (weak supervision), validando esas pseudo-etiquetas contra los 58 casos gold antes de confiar en ellas.
- [ ] **Fase 3 — Baseline de imagen**: pipeline de preprocesamiento de series MRI (selección de serie relevante por plano/secuencia, normalización) y modelo 2.5D/3D entrenado con gold + pseudo-etiquetas, corriendo en Kaggle Notebook (GPU T4).
- [ ] **Fase 4 — Validación**: cross-validation agrupado por estudio (y quizás por sitio, si se puede inferir) para evitar fuga de datos, optimizando macro ROC-AUC.
- [ ] **Fase 5 — Ajustar a restricciones de code competition**: notebook de inferencia debe correr en ≤9 horas sin internet, usando solo el estudio de imagen (sin `Report`) como entrada, con pesos/modelos preentrenados adjuntos como Kaggle Dataset.
- [ ] **Fase 6 — Iteración y envío final**: ensamble, ajuste de umbrales/calibración, antes del 2026-10-22 23:59.

## Cambio de plan: entrenar en la PC con RTX 3060

Se decidió mover el entrenamiento de imagen a una PC de escritorio con GPU (RTX 3060) en vez de depender solo de Kaggle Notebooks. Esta laptop no tiene GPU NVIDIA, así que sirve para EDA, prototipos chicos y preprocesamiento; el entrenamiento real corre en la PC de escritorio.

El proyecto no estaba en git (por eso "clonar" no aplicaba de inicio) y la carpeta, aunque tiene "OneDrive" en la ruta, no sincroniza de verdad (nombre simbólico). Ya se inicializó un repo git local aquí con `.gitignore` (excluye `.env`, `data/`, `*.dcm`, venvs) y `requirements.txt` (dependencias sin GPU: kaggle, pandas, pydicom, pillow, numpy). `torch` se deja fuera del requirements.txt a propósito porque su instalación depende del driver de NVIDIA de cada máquina.

Para completar la migración falta:
1. Crear un repo vacío en GitHub (privado, recomendado ya que sube reportes/labels de la competencia aunque `data/` esté excluido) y pasar la URL para hacer `git remote add origin <url> && git push -u origin master`.
2. En la PC con la 3060: instalar Git y Python (evitar el alias de Microsoft Store en Windows, mismo problema que en esta laptop), clonar el repo, crear venv, `pip install -r requirements.txt`.
3. Instalar PyTorch con soporte CUDA acorde al driver de esa PC (`nvidia-smi` para ver la versión soportada, luego el comando exacto de pytorch.org/get-started).
4. Recrear el archivo `.env` con el mismo `KAGGLE_TOKEN` manualmente en esa PC (no viaja por git, está en `.gitignore` a propósito).
5. Verificar acceso: exportar `KAGGLE_API_TOKEN` y correr `kaggle competitions files -c rsna-knee-abnormality-detection`; verificar GPU con `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

## Siguiente paso inmediato

Fase 1: EDA real. Empezar por el texto (ya está todo local) y por la exploración de imágenes de los 58 casos con etiqueta gold. En paralelo, terminar la migración de arriba para tener la 3060 lista para cuando empiece Fase 3 (entrenamiento de imagen).

## Qué hace la comunidad (revisado 2026-10-13)

Leaderboard público: top 0.966; los notebooks públicos llegan a ~0.956 mezclando modelos ajenos. Notebooks leídos: `pilkwang/rsna-knee-baseline-v1`, `dreaddevelopment/knee-mri-training-the-twelve-finding-model` (Raptor), `prvsiyan/rsna-knee-read-the-report-then-the-knee`, `goodpjw2008/...-mil-lb-0-954`, `mattiaangeli/bend-the-knee-to-dinov3-ensembled`, `romantamrazov/rsna-knee-dinosaur-the-last`.

- **Etiquetas**: nadie hace clustering. Todos convierten los 4,407 reportes en etiquetas suaves (0–1) con un LLM (o reglas multilenguaje) y entrenan la imagen contra eso, como maestro-texto → alumno-imagen. Los 58 gold se usan solo como validación. Datasets públicos: `pilkwang/rsna-knee-llm-labels`, `stevenleehans/rsna-knee-llm-report-labels`.
- **Datos ya preprocesados públicos**: `dreaddevelopment/knee-raptor-corpus` (+ `-ext`): los 4,407 estudios como `uint8` de 44×336×336, ~22 GB. Evita bajar y procesar los 264 GB.
- **Preprocesamiento que importa**: ordenar cortes por geometría (`ImagePositionPatient`·normal, no por nombre de archivo), recorte a campo de visión físico fijo (~140 mm), normalizar lateralidad (rodilla derecha/izquierda; `Laterality` falta en ~50 % de estudios), slots por plano/secuencia (sagital FS, sagital no-FS, coronal FS, coronal no-FS, axial).
- **Modelo**: 2.5D (tripletas de cortes adyacentes como 3 canales) + encoder 2D (CoAtNet-384, ConvNeXt, DINOv2/v3 ViT-S) + MIL con atención separada por hallazgo. Encoder afinado, no congelado. Promedio de pesos de los mejores epochs; TTA con espejo.
- **Combinación**: promedio de rangos (la métrica es AUC, solo importa el orden), pesos por hallazgo.
- **Inferencia**: lector tolerante a DICOMs rotos; un error en 1 de ~1,300 estudios tumbó envíos ajenos.

Traspaso al 2026-10-13:
- Actualización: Pixilate ya tiene ~560 GB libres en `C:` (se borró Linux). Big Chungus ya no hace falta y cualquier resolución de la tabla cabe.
- Pendiente del usuario, en casa: limpiar espacio en Pixilate y conectarle Big Chungus (HDD externo de 1 TB, casi lleno; confirmar cuánto espacio libre queda antes de usarlo como destino). Al retomar, volver a medir el espacio libre de ambos (`Get-PSDrive`) antes de decidir la resolución, porque 320 px podría caber.
- Pendiente de decidir: hacer privado el repo (las reglas solo permiten compartir código públicamente en Kaggle).
- Listo para usar: `scripts/zip_index.py` (lee el índice del zip remoto en ~40 s y escribe un CSV con offset y tamaño de cada miembro) y `scripts/zip_sample.py` (baja y decodifica miembros sueltos por rango). Base del preprocesamiento.
- Siguiente: con el visto bueno del usuario, crear el venv en Pixilate (torch CUDA, pydicom, pandas, kaggle), escribir el preprocesamiento por rangos a `.npz` por serie y lanzarlo con `job.ps1`. Empezar por los 58 estudios gold (3.4 GB) para validar el pipeline y hacer el EDA de imagen.
