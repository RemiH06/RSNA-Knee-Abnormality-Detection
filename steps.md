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
- [x] **Fase 1 — EDA**: prevalencia real de cada uno de los 12 hallazgos entre los 58 casos gold, calidad/longitud/idiomas de los reportes en las 4,407 filas, relación entre `Fluid_Sensitive`/`Fat_Suppression`/`Anatomical_Plane` y qué series conviene mirar por hallazgo, inspección visual de DICOMs de al menos algunos de los 58 casos gold.
- [x] **Fase 2 — Derivar pseudo-etiquetas del reporte**: como el texto no está en test, su único uso es generar labels ruidosas para los 4,349 estudios sin gold label (weak supervision), validando esas pseudo-etiquetas contra los 58 casos gold antes de confiar en ellas.
- [x] **Fase 3 — Baseline de imagen**: pipeline de preprocesamiento de series MRI (selección de serie relevante por plano/secuencia, normalización) y modelo 2.5D/3D entrenado con gold + pseudo-etiquetas, corriendo en Kaggle Notebook (GPU T4).
- [x] **Fase 4 — Validación**: cross-validation agrupado por estudio (y quizás por sitio, si se puede inferir) para evitar fuga de datos, optimizando macro ROC-AUC.
- [x] **Fase 5 — Ajustar a restricciones de code competition**: notebook de inferencia debe correr en ≤9 horas sin internet, usando solo el estudio de imagen (sin `Report`) como entrada, con pesos/modelos preentrenados adjuntos como Kaggle Dataset.
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

Traspaso al 2026-10-13 (noche):
- Pixilate: ~560 GB libres en `C:` (se borró Linux); Big Chungus ya no hace falta. Proyecto en `~\Desktop\Personal\iroFactory%.Knee_Abnormality_Detection` con `.venv` (Python 3.10, torch 2.11+cu128, timm, kagglehub 1.0.2), `.env` copiado por `scp` (está en `.gitignore`).
- Python 3.10 en Pixilate: `kaggle` 2.x y `numpy` 2.3 piden 3.11+. Por eso `scripts/setup_pixilate.ps1` instala sin versiones fijas y baja con `kagglehub` (`scripts/download_data.py`), que sí lee `KAGGLE_API_TOKEN`. Trampa vista: `timm` arrastra `torchvision` de PyPI y éste un torch solo-CPU; el setup instala ambos desde el índice cu128.
- Datos en Pixilate (`data/kagglehub/`): corpus Raptor (4,407 × 44 × 336 × 336, del 15 al 85 % de cada serie), etiquetas LLM de pilkwang y stevenleehans, `train.csv`, `train_series.csv`. Todos CC0.
- Etiquetas contra los 58 gold (AUC macro): steven_v4 0.893 (la que usamos), steven_v2 0.887, pilkwang_v2 0.870 (le falta 1 estudio). Las más débiles: Synovitis ~0.79, Lateral OA ~0.83.
- `scripts/train_raptor.py`: copia del entrenador público de Raptor (Apache 2.0) con rutas por defecto a `data/` y `runs/`, y etiquetas en CSV. Prueba de humo bien. Corriendo `knee-coatnet-v1`: CoAtNet-384, `--norm imagenet` (igual que la inferencia pública), bs 3, k 12, 12 epochs, ~11.1 de 12 GB de VRAM.
- 2026-10-14: `archive.zip` completo en Pixilate (`data/archive.zip`, 265 GB, 819,640 miembros) con `scripts/download_archive.py` (reanudable, renueva la URL firmada). `scripts/build_corpus.py` generó `data/corpus_v4` (4,407 × 64 × 336 × 336, del 6 al 94 %, misma lógica que `build_study` de la inferencia pública) en 22 min con 4 procesos, sin cortes ilegibles. Revisado a ojo: slots y orden de cortes correctos.
- v1 (corpus público de 44 cortes) se detuvo en el epoch 4 por un reinicio de Pixilate. AUC gold por epoch: 0.697, 0.804, 0.833, 0.851. Checkpoint del epoch 3 en `runs/raptor_ft_coatnet_v1.pt`. Cada epoch tarda 50 a 80 min.
- Corriendo `knee-coatnet-v2`: igual que v1 pero con `--corpus data\corpus_v4`. Comparar contra la curva de v1 epoch por epoch.
- Apagón en la madrugada del 2026-10-14 (Pixilate volvió a las 9:50): v2 murió en el epoch 2 (ep0 0.690, ep1 0.757). Desde entonces `train_raptor.py` guarda el estado completo en `runs/raptor_last_<tag>.pt` al cerrar cada epoch y acepta `--resume` (probado). Tras un corte, relanzar **el mismo comando**: sigue desde el último epoch cerrado. v2 relanzado de cero a las 10:00 con `--resume`.
- 2026-10-14 tarde: v2 cerró hasta ep4 (ep1 0.795, ep2 0.812, ep3 0.853, ep4 0.848; v1 en ep3 tenía 0.851) y murió a las 15:55 por CUDA OOM: ocupaba 11.4 de 12 GB y el usuario usó la PC. El usuario quiere que se use mucho la PC pero dejando margen para jugar. `train_raptor.py` ganó `--micro_bs` (acumula gradiente: misma matemática y mismos pasos del optimizador, compatible con `--resume`), `--vram_gb` (tope de VRAM) y reintento del batch ante OOM. v2 relanzado con `--micro_bs 1 --vram_gb 7 -AtStartup` (5.2 GB de VRAM; se relanza solo si Windows arranca tras un apagón). Al terminar v2, hacer `stop` para que no se relance en cada arranque.
- Noche del 14 al 15: la evaluación del final de cada epoch no cabía en el tope de 7 GB (2 estudios × 24 ventanas) y se quedaba reintentando; ahora el backbone evalúa por tramos (probado: mismo resultado ±0.001, pico 1.5 GB). Se perdió el ep5 una vez. Curva v2: ep5 0.865, ep6 0.881, ep7 0.887 (ep7 tardó 59 min por un juego abierto, normal 49). Debe terminar hacia las 3:30 am del 15. Pendiente: `--micro_bs` deja fuera del reintento el `x.to(dev)` (~64 MB); corregir en la próxima corrida.
- **v2 terminado (2026-10-15 03:12).** AUC gold: mejor epoch ep8 0.8902; SWA de ep7-9 **0.8916** (el elegido: `runs/raptor_ft_coatnet_v2_swa.pt`, 279 MB, en Pixilate); promedio de predicciones 0.8915. Bajó tras ep8 (0.888, 0.886, 0.884): empieza a sobreajustar a las etiquetas LLM. Por hallazgo (ep8): Medial OA 0.983, Baker's 0.976, Contusion 0.961, Medial Meniscus 0.953, ACL 0.949, Effusion 0.913, MCL 0.893, Lateral Meniscus 0.832, Fracture 0.826, PF OA 0.812, Synovitis 0.795, Lateral OA 0.789. Los débiles coinciden con las etiquetas LLM débiles. Referencia pública con la misma receta (16 epochs, bs 8): 0.905 en gold, 0.914 LB. Tarea `knee-coatnet-v2` detenida; GPU libre. `runs/raptor_last_coatnet_v2.pt` (2 GB) ya no hace falta.

## Siguiente sesión (empezar aquí)

Fase: inferencia y primer envío. Contexto de la tarea de clase en la memoria `rsna-knee-assignment` (cada centésima cuenta; el usuario aceptó combinar con modelos públicos).

1. Subir `runs/raptor_ft_coatnet_v2_swa.pt` como dataset **privado** de Kaggle, directo desde Pixilate (`kagglehub.dataset_upload`, con `scripts/run_with_token.ps1`).
2. Notebook de inferencia: partir de `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` (ya leído; su `build_study` es la misma geometría que `data/corpus_v4`: 64 cortes, 6-94 %, 140 mm, 336 px). Cambiar `ARMS` a nuestro archivo (arch `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, res 384, `NORM="imagenet"`, `K_EVAL` 42 como el público). Cuidar: sin internet, < 9 h en T4 para ~1,300 estudios, fp16 en T4 (no bf16), lector tolerante a DICOM rotos (un fallo no debe tumbar el envío), `submission.csv`.
3. Probar con los 3 estudios de ejemplo, enviar y anotar el score.
4. Combinar por rangos con los mejores públicos (`goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954`, `heliosli` 0.956/0.957, ver "Qué hace la comunidad").
5. En paralelo, en Pixilate: segundo modelo con pérdida ponderada por confianza de la etiqueta (`2·|p−0.5|`, en steven_v4 0.5 = "el reporte no lo menciona"), para Synovitis y Lateral OA. Usar `--resume --micro_bs 1 --vram_gb 7 -AtStartup`; meter `x.to(dev)` dentro del reintento ante OOM.
6. Al final: dos notebooks (trayecto + versión definitiva) y actualizar el README con el score.

**Hecho 2026-10-15 (mañana):** pasos 1 a 3.
- Pesos en el dataset privado `hctorheredia/kad-knee-weights` (subido con `scripts/upload_weights.py` vía `run_with_token.ps1`, que ahora reenvía argumentos).
- Notebook `kaggle/infer_v2/` → kernel privado `hctorheredia/kad-knee-infer-v2` (v1): el de Raptor con nuestros pesos, `kaggle kernels push -p kaggle/infer_v2`. Con los 3 estudios de ejemplo: 39 s, sin fallos, carga estricta de pesos OK.
- Primer envío: submission 57232135 (`kaggle competitions submit -c rsna-knee-abnormality-detection -k hctorheredia/kad-knee-infer-v2 -v 1 -f submission.csv`). Score público: **0.919** (el público con la misma receta saca 0.914).
- Paso 4 (2026-10-15 tarde): `kaggle/blend/` = fork de `heliosli/this-one-is-old-version-0-957-new-one-you-can-use` (R7 + Raptor 384/224, LB 0.957), generado con `python kaggle/blend/build.py <ipynb público> <peso>`. Nuestra etapa corre en segundo plano en la GPU 1 (nice 15) durante las etapas Raptor; si no terminó al empezar la etapa 2 se mata y el envío queda igual al público (`KAD_SKIPPED`). Ojo: el `heliosli/this-is-new-version-0-9576...` ya no corre tal cual, su dataset `rsna-knee-r6-frozen-fe055406` pasó a v2 (bundle R7 8c7b666e) y falla su assert de BUNDLE_ID.
  - Kernel `hctorheredia/kad-knee-blend` v2 = peso 0.15 → submission 57240482. Score: **0.956**.
  - v3 = peso 0 (control, público tal cual) → submission 57240695. Score: **0.956** (el original dice 0.957; puede ser redondeo o la imagen Docker: el fork usa la imagen por defecto, no la privada del autor). Kaggle solo da 3 decimales: con peso 0.15 nuestro modelo no mueve el score de forma visible.
  - Si v2 > v3, nuestro modelo suma; si empatan, revisar si fue `KAD_SKIPPED` (no se ve en el envío: probar con más tiempo o sin nice).
- Paso 5: corrección al plan, en steven_v4 el 0.5 casi no existe (≤0.2 %); lo incierto son valores intermedios (Synovitis 47 % entre 0.3 y 0.7, Lateral OA 4 %). `train_raptor.py --conf_weight 0.25` pondera la BCE por `max(0.25, 2|y-0.5|)`; además `x.to(dev)` ya está dentro del reintento ante OOM. Corriendo `knee-coatnet-v3` (igual que v2 + `--conf_weight 0.25 --seed 43`, `-AtStartup`), 5.4 GB de VRAM, ~10 h, termina hacia la noche del 15. Al terminar: `stop` y matar Python huérfano; comparar curva con v2; si suma en gold, subir con `upload_weights.py` como versión nueva del dataset y agregarlo como segundo brazo propio.
- Fase 1 (EDA) nunca se hizo como tal: no hay notebooks ni scripts de EDA en ningún equipo. Va en el notebook de "trayecto".
- Bakery: `job.ps1 stop` dijo "Detenido" pero dejó vivos al proceso de Python y a sus workers del DataLoader (11.5 GB de VRAM); hubo que matarlos con `Stop-Process`. Arreglar en `29.Bakery`.
- WSL 2 con Debian ya instalado en Pixilate (2026-10-14); `bakery` y `hardware` actualizadas en SkillShot y sincronizadas.
- Inferencia: cada modelo necesita la geometría de su corpus (v1: 44 cortes del 15 al 85 %; v2: 64 del 6 al 94 %); el notebook debe llevarla por modelo.
- Siguiente (antes): (1) con el AUC gold de v1, armar el notebook de inferencia reusando el de Raptor (`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`) con nuestros pesos como dataset de Kaggle; (2) propuesta pendiente de aprobar: bajar `archive.zip` completo (264 GB, ~1.5 h) a Pixilate y regenerar el corpus a 64 cortes del 6 al 94 % con `build_study` del notebook de inferencia, que es la geometría de los pesos públicos de 0.924; (3) después, combinar por rangos con los mejores modelos públicos.
- Pendiente de decidir: remoto de git (privado) para dejar de pasar código por `scp`.

**Hecho 2026-10-15 (tarde): formalidades de entrega (materia: Análisis Estadístico Multivariable, Prof. Luis Alvarado).**
- `notebooks/01_trayecto.ipynb` (ejecutado, corre en la laptop con `data/` + `results/`): Fases 0 a 6 con los temas de la materia. Hallazgos que no estaban en el plan: `Fluid_Sensitive` == `Fat_Suppression` en las 24,371 series; series por estudio no son Poisson (dispersión 0.35); 23/66 pares de hallazgos rechazan independencia por LRT (1 con Bonferroni); valores propios 3.08 y 1.93 (ejes degenerativo/traumático y lesión aguda); las etiquetas LLM conservan las correlaciones positivas (r = 0.74) pero aplanan las negativas; el modelo supera a su etiqueta en contusión (0.96 vs 0.86); IC95 bootstrap del AUC gold [0.856, 0.922] y las variantes de v2 no se distinguen entre sí.
- Insumos en `results/`: `gold/*.json` (historiales), `submissions.csv` (actualizar a mano con cada envío), y `.npz` ignorados por git (derivados de datos de la competencia): `gold/raptor_gold_*.npz` se copian de `runs/` de Pixilate y `gold_slices.npz` sale de `scripts/extract_gold_slices.py`.
- `.gitignore` arreglado: `runs/` no se estaba ignorando (estaba pegado a `.ipynb_checkpoints/`).
- `notebooks/02_version_definitiva.ipynb`: el notebook de Kaggle ganador con portada y explicación generado con `python notebooks/build_definitiva.py kaggle/blend/kad-knee-blend.ipynb notebooks/02_version_definitiva.ipynb <peso> <score> <submission>` (portada + celdas de Kaggle sin cambios).

**Eficiencia (2026-10-15 tarde).** Premio aparte: `Eficiencia = AUC/(Benchmark − maxAUC) + segundos/32400`, menor es mejor; 0.01 de AUC ≈ 11.6 min. Solo cuentan los 2 envíos finales seleccionados. Leaderboard (solo lugares): `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`. Top 11 de eficiencia tienen AUC 0.955-0.963; las combinaciones públicas de 0.957 tardan ≥ ~4 h (mediana de la cota deducida de los lugares).
- `kaggle/bench/` (kernel `hctorheredia/kad-knee-bench`): nuestro v2 sobre los 58 gold en 2×T4. Secuencial 2.57 s/estudio (lectura 1.74, ventanas 0.12, GPU 0.71) → ~56 min para ~1,300. Paralelo 58 estudios en 56 s (vs 149) → ~20 min. Resultados en `results/bench/`.
- AUC gold contra K (ventanas): 4 0.812, 8 0.817, 12 0.859, 16 0.879, 20 0.887, 24 0.891, 32 0.892, **42 0.896**, 62 0.893. K = 42 minimiza la eficiencia (K = 24 ahorra ~3 min pero cuesta ~6 min equivalentes de AUC). Se queda en 42.
- `kad-knee-infer-v2` v2: inferencia paralela (un proceso por GPU con 2 lectores de DICOM; respeta `CUDA_VISIBLE_DEVICES` heredado para el blend; lo que no regrese lo termina el camino secuencial). Salida idéntica a v1 en los 3 de ejemplo. Enviado: submission 57243872. Score: **0.919** (igual que v1; aguantó el test oculto completo y calificó antes que los blends que se enviaron 2.5 h antes).
- Con ~20 min, el tiempo ya pesa poco (~0.04); para eficiencia la palanca ahora es el AUC (p. ej. un segundo brazo propio cuesta ~+10 min ≈ 0.009 de AUC).

**Más milésimas (2026-10-15 noche).** Objetivo del usuario: mejor posición posible (valor curricular), no los puntos extra.
- Opción 2, R6: Kaggle ignora la versión al adjuntar un dataset (el CLI acepta `dueño/slug/1`, pero el servidor guarda solo `dueño/slug` y monta la última). Se bajó la v1 de `heliosli/rsna-knee-r6-frozen-fe055406` con el CLI (`kaggle datasets download .../1 -f <archivo>`; kagglehub da 404 con versiones viejas), hashes verificados contra su manifest, y se subió como privado `hctorheredia/kad-r6-fe055406` (crédito a heliosli en la descripción). `kaggle/blend_r6/` = fork del 0.9576 con `C_SLUG` apuntando a la copia. Prueba OK (flip TTA activo). Submission 57245875. Score: **0.957** (sube una milésima sobre R7).
- Opción 1, más modelos: S6 de nartaa = `nartaa/rsna-knee-s6fall-solo` (script; blend 0.7 Raptor-0.949 + 0.3 S6 ConvNeXt vía `compact_infer.py`, mismo motor que R6/R7). Dice que Raptor imprime cada 200 estudios "~3 min" → la etapa Raptor 384 tarda ~20 min; da a S6 hasta 4 h de presupuesto. Sin tiempos reales de R6/R7/S6 en el test completo no se sabe si caben R6 + R7 + S6 en 9 h.
- Siguiente: arnés de tiempos: correr el fork con ~100-300 estudios de train disfrazados de test (test.csv/test_series.csv falsos y enlaces a train_series) para medir cada etapa y, de paso, el AUC gold de cada miembro. Con eso decidir qué cabe y con qué pesos.
- **Arnés de tiempos** (`kaggle/timing/build.py`, 200 estudios de train como test, 58 gold incluidos; resultados en `results/timing/`):
  - Fork R6 (0.9576): 3,200 s → **~5.8 h** para 1,300. Raptor 384 301 s, 224 91 s, etapa 2 R6 con flip TTA 2,783 s (GPU 23.7 s/estudio).
  - Fork R7: 2,335 s → **~4.2 h**. Etapa 2 R7 sin flip 1,817 s (GPU 13.3 s/estudio).
  - R6 + R7 juntos no caben (~9 h+).
  - **Nuestra etapa se cortó (`KAD_SKIPPED`)**: solo 100 de 200 estudios antes de la etapa 2. En el envío real seguro igual: el empate 0.956 / 0.956 no midió nuestro modelo.
- AUC gold (58) por miembro: Raptor 384 0.9229 y 224 0.9150 (elegidos con los gold: optimistas), R6 0.9366 (manifest `gold_in_training` 0), R7 0.9312, KAD v2 0.8916.
  Combinaciones por rangos: R6 blend 0.9388 > R7 blend 0.9365 (cuadra con el LB). Agregar R7 al R6: −0.0012 (P>0 = 0.23). Agregar KAD a 0.15: −0.0030 (P>0 = 0.07); a 0.1: −0.0019. Conclusión: nuestro v2 no ayuda a la combinación; la base es R6.
- `notebooks/03_interpretacion.ipynb` (generado con `notebooks/build_interpretacion.py`): atención por corte de los 58 gold con todas las ventanas (62) y Grad-CAM en 2 estudios, de `scripts/attention_maps.py` en Pixilate (~3 min, tope 4 GB de VRAM; en Pixilate `train.csv` está en `data/kagglehub/...`, el script lo busca solo). Hallazgos: sagital FS domina (×1.4 a 2.6 sobre uniforme), coronal no-FS casi ignorado; LCM mira coronal FS y axial; meniscos los más focalizados (~11 ventanas efectivas), OA las más difusas (>30); 8/12 hallazgos concentran distinto en positivos (Mann-Whitney), derrame al revés; traslape de atención no se relaciona con co-ocurrencia (ρ = 0.09). La salida del Grad-CAM (imágenes) ya va borrada.
- **R6 + S6** (`kaggle/blend_r6s6/build.py <rsna-knee-s6fall-solo.py> <W_S6>`): S6 de nartaa como etapa opcional después de R6, con el tiempo que sobre (`KAD_S6_GUARD_S`); su componente A (Raptor 0.949) se lee de `raptor_sub.csv` en vez de correrlo otra vez; si falla queda el blend R6 (`S6_SKIPPED`). Arnés de 200: S6 816 s → total **~7.3 h** para 1,300 (cabe). AUC gold: S6 0.9314 (entrenado sin los gold); blend R6 0.9382 → R6+S6 w=0.3 **0.9402** (+0.002, P>0 ≈ 0.77; plano entre 0.1 y 0.5, el 0.3 se fijó antes de ver). Submission 57254826. Score: PENDIENTE.
- v3 terminó (2026-10-15 22:20): mejor epoch ep8 0.8867, SWA ep8-10 **0.8871** < v2 0.8916. La ponderación por confianza no ayudó. Tarea detenida.

**Día 2026-10-16.** R6 + S6 (57254826) = **0.957**, igual que R6: la mejora en gold (+0.002) no se vio en el público a 3 decimales.
- Nuevo público: `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` (0.958): Raptor + R6 con flip (copia versionada `rostigor/rsna-knee-r6-fe055406`) + pesos por hallazgo de heliosli (Raptor: Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, resto 0.5) + su lector 2.5D a ~1/10 (parte B, sin tope de tiempo, se salta si ya pasaron 8.2 h). `kaggle/blend_r6s6/build.py` ahora acepta una base: `kaggle/kad-knee-gp958s6/` = ese notebook + S6 a 0.3. Prueba de 3 estudios OK. Arnés de 200 (`kad-timing-gp958s6`) corriendo para medir la parte B antes de enviar.
- Etiquetas: mezclar fuentes LLM no mejora (v4 0.8932; v4+pilkwang 0.8935; las 4: 0.8901). Por hallazgo, los modelos ya superan a las etiquetas en casi todo (Fracture: etiquetas 0.793, blend 0.924). En Synovitis y Fracture el reporte no menciona el hallazgo en ~40-60 % de los positivos gold (búsqueda por palabras clave): un mejor lector de texto no puede recuperarlos. Lateral OA sí se menciona (9/11) pero falla el compartimento.
- En gold, el blend empeora Lateral OA (0.834) frente a R6 solo (0.865): cuadra con el 0.2 de heliosli.
- **Destilación** (`kaggle/teacher/`): maestro = fork R6 sin flip TTA sobre los 4,407 de train en 4 partes (`build.py 4` → `kaggle/timing/kad-teacher-<k>`), máx. 2 sesiones de GPU a la vez en Kaggle. `collect.py 4` junta y escribe `data/teacher/teacher_labels.csv` (rango del maestro → cuantiles de las etiquetas LLM) para `train_raptor.py --labels`. Riesgo: Raptor alldata y R6 (4 folds) se entrenaron con esos mismos estudios; el maestro puede repetir sus etiquetas en vez de corregirlas. `collect.py` lo diagnostica (AUC maestro vs LLM dentro de train contra gold).
- Cola en Pixilate (para no depender de Aerilate): `scripts/kaggle_queue.py <carpetas de kernel>` sube kernels con kagglesdk (el CLI `kaggle` 2.x no corre en Python 3.10) conforme se libera una de las 2 sesiones de GPU; marca `.pushed` en cada carpeta, reanudable. Trabajo `knee-queue` (`-AtStartup`) con las partes 1-3 del maestro, copiadas a `kaggle/timing/` de Pixilate. Al terminar: `stop`.
- `knee-convnext-v4` (2026-10-16 11:04): segundo modelo propio para diversidad, receta de v2 con `convnext_small.fb_in22k_ft_in1k_384` (`--tag convnext_v4`, `-AtStartup`), 2.4 GB de VRAM. Al terminar: `stop`, comparar con v2 y probar el ensamble v2 + v4 en gold.

## Siguiente sesión (empezar aquí) — traspaso del 2026-10-16

Estado: mejor público **0.957** (R6 fork, 57245875; R6 + S6, 57254826, empató). Envíos del 16: quedan 4. Fecha límite 2026-10-22 23:59 (UTC). Objetivo: mejor posición posible.

Corriendo al cerrar (nada depende de Aerilate):
- Kaggle: `kad-teacher-0` (parte 0 del maestro) y `kad-timing-gp958s6` (arnés de goodpjw + S6).
- Pixilate: `knee-queue` (sube `kad-teacher-1..3` conforme hay sesión libre; log `~\bakery-jobs\knee-queue.log`) y `knee-convnext-v4` (entrenamiento ~10 h, log `knee-convnext-v4.log`).

Pendientes, en orden:
1. **goodpjw + S6**: leer `timing.json` de `kad-timing-gp958s6` (`kaggle kernels output hctorheredia/kad-timing-gp958s6 -p <dir>`). Si `extrapolated_hours_1300` < ~8 y la parte B ("own leg") no se salta ni falla, enviar `hctorheredia/kad-knee-gp958s6` v1. Si no cabe, quitar la parte B o bajar S6. Comparar AUC gold contra R6 + S6 (0.9402).
2. **Maestro, parte 0**: bajar sus CSV y medir memorización (lo de `kaggle/teacher/collect.py`, pero con una sola parte): AUC maestro vs etiquetas LLM en estudios de train contra lo mismo en gold. Si en train es mucho más alto, el maestro repite sus etiquetas: parar la cola (`job.ps1 stop knee-queue`) y no gastar más cuota. Si no, esperar las 4 partes, `python kaggle/teacher/collect.py 4`, copiar `data/teacher/teacher_labels.csv` a Pixilate y entrenar el alumno (receta de v2 con `--labels data\teacher\teacher_labels.csv`, sin los gold).
3. **convnext_v4**: al terminar, `stop` (tiene `-AtStartup`), matar Python huérfano, comparar con v2 (0.8916) y probar el ensamble v2 + v4 en gold; si suma, segundo brazo en `kaggle/infer_v2` (envío propio y de eficiencia).
4. Al cerrar trabajos con `-AtStartup` (`knee-queue`, `knee-convnext-v4`): `stop`.
5. Al final: elegir los 2 envíos finales (mejor blend + nuestro modelo propio), notebooks de trayecto/definitivo, README con el score.
