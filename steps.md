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

`train_series.csv` tiene 24,371 series; con mediana de 30 cortes por serie y ~1.76 MB por archivo `.dcm` (tamaño observado, consistente entre archivos), el set de entrenamiento completo son **~700,000+ archivos, del orden de 1 TB**. Esto no es viable de descargar ni entrenar en esta máquina local (sin GPU NVIDIA). Consecuencia para el plan:

- El desarrollo de imagen (EDA visual, entrenamiento, validación) debe hacerse **dentro de un Kaggle Notebook**, donde el dataset ya está montado y hay GPU (T4/P100) gratis.
- Localmente solo tiene sentido descargar un subconjunto chico: por ejemplo las imágenes de los 58 estudios con etiqueta gold, para prototipos rápidos antes de subir el notebook completo a Kaggle.
- El trabajo sobre texto (reportes) sí es 100% viable local, porque `train.csv` (5.4 MB) ya está descargado completo.

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

## Siguiente paso inmediato

Fase 1: EDA real. Empezar por el texto (ya está todo local) y por la exploración de imágenes de los 58 casos con etiqueta gold.
