"""Maestro de la destilación: el fork R6 (Raptor 384 + 224 + R6, sin flip TTA para ahorrar cuota de GPU) sobre los
4,407 estudios de train, en K partes que caben cada una en un notebook de 9 h.

Arma la base (kaggle/teacher/kad-teacher-base.ipynb) y, con el arnés de kaggle/timing/build.py, un kernel por parte
(kaggle/timing/kad-teacher-<k>/). Cada parte deja en su salida raptor_raw.csv, raptor224_raw.csv, ours_sub.csv (R6) y
submission.csv (la mezcla por rangos); kaggle/teacher/collect.py los junta.
Uso: python kaggle/teacher/build.py <K>
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = int(sys.argv[1])
src = HERE.parent / "blend_r6"
nb = json.load(open(src / "kad-knee-blend-r6.ipynb", encoding="utf-8"))
meta = json.load(open(src / "kernel-metadata.json", encoding="utf-8"))

old = "FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6"
hits = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    if old in s:
        hits += 1
        c["source"] = s.replace(old, "FLIP_TTA = False   # [teacher] sin flip TTA: ~35 % menos de GPU").splitlines(keepends=True)
assert hits == 1, hits

base = HERE / "kad-teacher-base.ipynb"
json.dump(nb, open(base, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
json.dump(meta, open(HERE / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
for k in range(K):
    subprocess.run([sys.executable, str(HERE.parent / "timing" / "build.py"), str(base), f"kad-teacher-{k}", f"{k}/{K}"],
                   check=True)
