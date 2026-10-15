"""Sube pesos entrenados a un dataset privado de Kaggle para el notebook de inferencia.

Copia los .pt a una carpeta limpia (kagglehub sube la carpeta entera) y crea el dataset
o una versión nueva si ya existe. kagglehub crea los datasets privados.
Uso: scripts/run_with_token.ps1 scripts/upload_weights.py runs/raptor_ft_coatnet_v2_swa.pt
"""
import argparse, shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("weights", nargs="+")
ap.add_argument("--handle", default="hctorheredia/kad-knee-weights")
ap.add_argument("--notes", default="")
a = ap.parse_args()

up = ROOT / "runs" / "kaggle_upload"
shutil.rmtree(up, ignore_errors=True)
up.mkdir(parents=True)
for w in a.weights:
    shutil.copy2(ROOT / w, up / Path(w).name)
    print("copiado", w, flush=True)
kagglehub.dataset_upload(a.handle, str(up), version_notes=a.notes)
print("subido a", a.handle, flush=True)
