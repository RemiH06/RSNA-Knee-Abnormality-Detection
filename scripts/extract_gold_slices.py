"""Extrae del corpus el corte central de cada grupo (slot) para algunos estudios, para mostrarlos en el notebook.

Uso: python scripts/extract_gold_slices.py <StudyInstanceUID> [...] --out results/gold_slices.npz
Lee data/corpus_v4 con memmap: no carga los ~32 GB del corpus.
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SLOTS = [("Sagittal FS", 18), ("Sagittal no-FS", 14), ("Coronal FS", 12), ("Coronal no-FS", 8), ("Axial", 12)]

ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="+")
ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus_v4"))
ap.add_argument("--out", default=str(ROOT / "results" / "gold_slices.npz"))
a = ap.parse_args()

c = Path(a.corpus)
vols = np.load(c / "all_vols.npy", mmap_mode="r")
masks = np.load(c / "all_masks.npy")
ids = list(np.load(c / "all_ids.npy", allow_pickle=True))

out = np.zeros((len(a.ids), len(SLOTS), vols.shape[2], vols.shape[3]), np.uint8)
for i, uid in enumerate(a.ids):
    r, start = ids.index(uid), 0
    for j, (_, k) in enumerate(SLOTS):
        valid = np.flatnonzero(masks[r, start:start + k])
        if len(valid):
            out[i, j] = vols[r, start + valid[len(valid) // 2]]
        start += k
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, slices=out, ids=np.array(a.ids), slots=np.array([s[0] for s in SLOTS]))
print("ok", out.shape, a.out)
