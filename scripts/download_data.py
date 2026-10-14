"""Baja a data/ el corpus preprocesado público, las etiquetas LLM y los CSV de la competencia.

Usa kagglehub (no el CLI de kaggle) porque kaggle 2.x pide Python >= 3.11 y Pixilate tiene 3.10;
kagglehub lee KAGGLE_API_TOKEN del entorno. Reanudable: kagglehub salta lo que ya está en caché.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data" / "kagglehub"))
import kagglehub  # noqa: E402  (KAGGLEHUB_CACHE debe fijarse antes de importar)

DATASETS = [
    "dreaddevelopment/knee-raptor-corpus",
    "dreaddevelopment/knee-raptor-corpus-ext",
    "pilkwang/rsna-knee-llm-labels",
    "stevenleehans/rsna-knee-llm-report-labels",
]
COMP = "rsna-knee-abnormality-detection"

for ds in DATASETS:
    print("dataset", ds, "->", kagglehub.dataset_download(ds), flush=True)
for f in ("train.csv", "train_series.csv"):
    print("competencia", f, "->", kagglehub.competition_download(COMP, path=f), flush=True)
