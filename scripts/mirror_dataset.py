"""Copia una versión fija de un dataset público de Kaggle a un dataset privado nuestro.

Kaggle no respeta la versión al adjuntar un dataset a un notebook (siempre monta la última), así que
para usar una versión vieja hay que volver a subirla. Uso:
scripts/run_with_token.ps1 scripts/mirror_dataset.py heliosli/rsna-knee-r6-frozen-fe055406/versions/1 hctorheredia/kad-r6-fe055406
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data" / "kagglehub"))
import kagglehub  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
path = kagglehub.dataset_download(src)
print("bajado", src, "->", path, flush=True)
kagglehub.dataset_upload(dst, path, version_notes=f"copia de {src}")
print("subido a", dst, flush=True)
