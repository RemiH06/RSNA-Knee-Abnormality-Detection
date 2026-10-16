"""Junta las salidas de las partes del maestro (kad-teacher-<k>) y arma las etiquetas de destilación.

Baja raptor_raw.csv, raptor224_raw.csv, ours_sub.csv (R6) y submission.csv de cada parte, verifica que juntas
cubran los 4,407 estudios una sola vez y escribe data/teacher/teacher_members.csv (miembros crudos, con prefijo) y
data/teacher/teacher_labels.csv: para cada hallazgo, el rango del maestro llevado a la distribución de las
etiquetas LLM (mapeo por cuantiles). Así el alumno recibe el orden del maestro con la prevalencia de siempre, y
train_raptor.py --labels lo usa sin más cambios. Diagnóstico: AUC del maestro contra las etiquetas LLM (umbral 0.5)
fuera y dentro de los gold; si dentro de train es mucho mayor, el maestro memorizó sus etiquetas.
Uso: python kaggle/teacher/collect.py <K> [--skip-download]
"""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]
FILES = {"r384": "raptor_raw.csv", "r224": "raptor224_raw.csv", "R6": "ours_sub.csv", "blend": "submission.csv"}
K = int(sys.argv[1])
OUT = ROOT / "data" / "teacher"
OUT.mkdir(parents=True, exist_ok=True)

parts = []
for k in range(K):
    d = OUT / f"part{k}"
    if "--skip-download" not in sys.argv:
        d.mkdir(exist_ok=True)
        for f in FILES.values():
            subprocess.run(["kaggle", "kernels", "output", f"hctorheredia/kad-teacher-{k}", "-p", str(d), "--file-pattern", f],
                           check=True, capture_output=True, env=os.environ)
    m = None
    for name, f in FILES.items():
        df = pd.read_csv(d / f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB].add_prefix(name + "|")
        m = df if m is None else m.join(df, how="outer")
    parts.append(m)
mem = pd.concat(parts)
assert mem.index.is_unique, "un estudio aparece en dos partes"
tr = pd.read_csv(ROOT / "data" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
assert set(mem.index) == set(tr.index), f"faltan {len(set(tr.index) - set(mem.index))} estudios"
assert mem.notna().all().all(), "hay valores vacíos"
mem.to_csv(OUT / "teacher_members.csv")

llm = pd.read_csv(ROOT / "data" / "labels" / "stevenleehans" / "llm_labels_v4_blend.csv",
                  dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
ids = [u for u in mem.index if u in llm.index]
teach = pd.DataFrame(index=ids)
for c in LAB:
    r = mem.loc[ids, "blend|" + c].rank(pct=True, method="average")
    q = np.sort(llm.loc[ids, c].to_numpy())
    teach[c] = np.quantile(q, r.clip(0, 1))          # rango del maestro -> cuantil de las etiquetas LLM
teach.index.name = "StudyInstanceUID"
teach.to_csv(OUT / "teacher_labels.csv")

gold = tr.dropna(subset=LAB).index
for name, sel in (("train (dentro de muestra)", [u for u in ids if u not in gold]), ("gold", [u for u in ids if u in gold])):
    a = [roc_auc_score((llm.loc[sel, c] > 0.5).astype(int), mem.loc[sel, "blend|" + c]) for c in LAB
         if (llm.loc[sel, c] > 0.5).nunique() > 1]
    print(f"maestro vs etiquetas LLM, {name}: AUC {np.mean(a):.4f} ({len(sel)} estudios)")
print("escrito", OUT / "teacher_labels.csv", teach.shape)
