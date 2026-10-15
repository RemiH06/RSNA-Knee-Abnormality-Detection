"""Genera notebooks/02_version_definitiva.ipynb a partir del notebook de Kaggle enviado.

Uso: python build_definitiva.py <notebook de kaggle .ipynb> <salida .ipynb> <peso KAD> <score público> <submission id>
El código de Kaggle no se toca: solo se agregan celdas de texto.
"""
import json
import sys

src, out, w, score, sid = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
nb = json.load(open(src, encoding="utf-8"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


if w > 0:
    formula = r"""$$s_i^{\text{final}}=(1-w)\Big[0.5\,\big(0.65\,r^{384}_i+0.35\,r^{224}_i\big)+0.5\,r^{R7}_i\Big]+w\,r^{KAD}_i,\qquad w=""" + f"{w:g}" + "$$"
    kad_line = f"| KAD v2 (propio) | CoAtNet-384 2.5D + MIL, SWA de los epochs 8 a 10 | 0.919 solo | Corre en paralelo en la segunda GPU; entra con peso {w:g} |"
else:
    formula = r"$$s_i^{\text{final}}=0.5\,\big(0.65\,r^{384}_i+0.35\,r^{224}_i\big)+0.5\,r^{R7}_i$$"
    kad_line = "| KAD v2 (propio) | CoAtNet-384 2.5D + MIL | 0.919 solo | Probado como brazo extra; no mejoró el leaderboard, peso final 0 |"

head = md(rf"""
# ITESO, Universidad Jesuita de Guadalajara

**Materia:** Análisis Estadístico Multivariable

**Profesor:** Luis Alvarado

**Actividad:** Proyecto: RSNA Knee Abnormality Detection (Kaggle). Notebook 2 de 2: versión definitiva

**Integrante:** Lambda Remiel Heredia Pérez (Kaggle: `hctorheredia`, GitHub: `RemiH06`)

**Fecha:** 15 de octubre de 2026

---

## Qué es este notebook

Es el notebook que Kaggle ejecuta para calificar: corre sin internet sobre los ~1,300 estudios ocultos de test, en una máquina con GPU T4 y un límite de 9 horas, y escribe `submission.csv`. No se puede correr fuera de Kaggle (necesita los DICOM de test y los pesos adjuntos como datasets).

El análisis, las pruebas estadísticas y la justificación de cada decisión están en `01_trayecto.ipynb`. Aquí solo se resume la versión final.

**Envío:** {sid}. **Score público (AUC macro):** {score}.

## Cómo llega a la predicción final

Cada modelo produce, para cada estudio $i$ y hallazgo $k$, una puntuación. Como la métrica es el AUC y el AUC solo depende del orden (notebook 1, sección "Métrica"), los modelos se combinan por **rangos percentiles**: $r^m_i=\operatorname{{rank}}(s^m_i)/N$, por hallazgo. La combinación, con pesos fijados antes de enviar y sin ajustarlos a ninguna etiqueta, es

{formula}

| Modelo | Arquitectura | AUC público | Papel |
|---|---|---|---|
| Raptor-384 (nartaa, público) | CoAtNet-384 2.5D + MIL | 0.949 solo | Etapa 1 |
| Raptor-224 (nartaa, público) | El mismo modelo con recorte central a 224 px | 0.945 solo | Etapa 1b |
| R7 (heliosli, público) | ConvNeXt-small, 4 folds | | Etapa 2 |
{kad_line}

Promediar rangos de modelos distintos reduce la varianza del error por la misma razón que el promedio de variables aleatorias: $\operatorname{{Var}}\big(\tfrac{{X_1+X_2}}{{2}}\big)=\tfrac14(\sigma_1^2+\sigma_2^2+2\sigma_{{12}})$ es menor mientras menos covarianza tengan los errores. Modelos con arquitecturas distintas (CoAtNet contra ConvNeXt) se equivocan en estudios distintos.

## Robustez

- Cada etapa corre como proceso aparte, para liberar la memoria de la GPU antes de la siguiente.
- Si una etapa pública falla, el notebook se detiene en vez de enviar en silencio un modelo incompleto.
- La etapa propia (KAD) nunca puede tumbar el envío: si no termina a tiempo o falla en más del 3 % de los estudios, se descarta y la combinación queda igual a la pública.
- Un estudio en el que un modelo falló toma el rango de los otros modelos.

## Créditos

Este notebook es un fork de [heliosli/this-one-is-old-version-0-957-new-one-you-can-use](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-957-new-one-you-can-use), sin cambios en sus etapas; los créditos completos de los modelos públicos están en las celdas siguientes (en inglés, como las publicó su autor). La aportación propia es el modelo KAD y su integración como etapa en paralelo.

---
""")

nb["cells"] = [head] + nb["cells"]
json.dump(nb, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print("ok", out, len(nb["cells"]), "celdas")
