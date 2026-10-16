"""Arnés de tiempos: toma uno de nuestros forks de blend y lo corre sobre estudios de train disfrazados de test.

Crea /tmp/fakecomp con test.csv, test_series.csv, sample_submission.csv y test_series/<estudio> como enlaces a
train_series (los 58 gold + estudios al azar hasta N), y parcha los tres lugares donde el notebook encuentra la
competencia. Al final mide cada etapa y el AUC gold de cada miembro y del blend.
Uso: python kaggle/timing/build.py <ipynb del fork> <slug del kernel> <N estudios | k/K> [datasets extra...]
     k/K = parte k de K de los 4,407 estudios de train (para correr el maestro de la destilación por partes)
     (lee kernel-metadata.json junto al ipynb para heredar los datasets)
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src, slug, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if "/" in n:
    _k, _K = map(int, n.split("/"))
    SEL, DESC = f"[{_k}::{_K}]", f"parte {_k} de {_K} de todos los estudios de train"
else:
    SEL, DESC = f"[:{int(n)}]", f"{int(n)} estudios de train (los 58 gold primero)"
nb = json.load(open(src, encoding="utf-8"))
meta = json.load(open(src.parent / "kernel-metadata.json", encoding="utf-8"))


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(keepends=True)}


fake = f'''# [timing] test falso: {DESC}, con la misma estructura que test.
import os, numpy as np, pandas as pd
from pathlib import Path
_COMP = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not (_COMP / 'train.csv').is_file():
    _COMP = Path('/kaggle/input/rsna-knee-abnormality-detection')
_FAKE = Path('/tmp/fakecomp'); (_FAKE / 'test_series').mkdir(parents=True, exist_ok=True)
_LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"]
_tr = pd.read_csv(_COMP / 'train.csv', dtype={{'StudyInstanceUID': str}})
_gold = _tr.dropna(subset=_LAB)['StudyInstanceUID'].tolist()
_rest = _tr.loc[~_tr['StudyInstanceUID'].isin(_gold), 'StudyInstanceUID'].sample(frac=1, random_state=0).tolist()
FAKE_IDS = (_gold + _rest){SEL}
pd.DataFrame({{'StudyInstanceUID': FAKE_IDS}}).to_csv(_FAKE / 'test.csv', index=False)
_ser = pd.read_csv(_COMP / 'train_series.csv', dtype={{'StudyInstanceUID': str, 'SeriesInstanceUID': str}})
_ser[_ser['StudyInstanceUID'].isin(FAKE_IDS)].to_csv(_FAKE / 'test_series.csv', index=False)
_ss = pd.DataFrame({{'StudyInstanceUID': FAKE_IDS}})
for _c in _LAB: _ss[_c] = 0.5
_ss.to_csv(_FAKE / 'sample_submission.csv', index=False)
for _s in FAKE_IDS:
    _d = _FAKE / 'test_series' / _s
    if not _d.exists(): os.symlink(_COMP / 'train_series' / _s, _d)
os.environ['KAD_FAKE_ROOT'] = str(_FAKE)
print('FAKE_READY', len(FAKE_IDS), 'estudios,', len(_gold), 'gold', flush=True)
'''

report = '''# [timing] tiempos por etapa y AUC gold de cada miembro
import glob, json, time
from sklearn.metrics import roc_auc_score
_g = _tr[_tr['StudyInstanceUID'].isin(_gold)].set_index('StudyInstanceUID')[_LAB]


def _auc(df):
    df = df.set_index('StudyInstanceUID').reindex(_g.index)[_LAB].astype(float)
    ok = df.notna().all(axis=1)
    return round(float(np.mean([roc_auc_score(_g.loc[ok, c].astype(int), df.loc[ok, c]) for c in _LAB])), 4), int(ok.sum())


rep = {'n_studies': len(FAKE_IDS), 'total_seconds': round(time.time() - NB_T0, 1)}
for k in ('RAPTOR_SECONDS', 'RAPTOR224_SECONDS'):
    if k in globals(): rep[k] = round(globals()[k], 1)
_rc = Path('/kaggle/working/submission_receipt.json')
if _rc.is_file():
    _r = json.loads(_rc.read_text()); rep['stage2_seconds'] = _r.get('seconds'); rep['stage2_receipt'] = {
        k: _r.get(k) for k in ('status', 'scored', 'fallback_studies', 'per_study_mean', 'gpus')}
for f in ('raptor_raw.csv', 'raptor224_raw.csv', 'ours_sub.csv', 'kad_raw.csv', 's6_sub.csv', '_own.csv', '_public_stack.csv', 'submission.csv'):
    p = Path('/kaggle/working') / f
    if p.is_file():
        rep['auc_' + f] = _auc(pd.read_csv(p, dtype={'StudyInstanceUID': str}))
_scale = 1300 / len(FAKE_IDS)
rep['extrapolated_hours_1300'] = round(rep['total_seconds'] * _scale / 3600, 2)
Path('/kaggle/working/timing.json').write_text(json.dumps(rep, indent=1, default=str))
print('TIMING', json.dumps(rep, default=str), flush=True)
'''

PATCHES = [
    ('cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",',
     'cands = [os.environ.get("KAD_FAKE_ROOT", "/nonexistent"), "/kaggle/input/competitions/rsna-knee-abnormality-detection",'),
    ("DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path(os.environ['KAD_FAKE_ROOT'])  # [timing]"),
]
OPTIONAL = [   # solo en algunos forks (lector 2.5D de goodpjw2008)
    ('_o_find("knee.py"), _o_find("test_series.csv")',
     '_o_find("knee.py"), (_o_os.environ.get("KAD_FAKE_ROOT") or _o_find("test_series.csv"))'),
]
hits = {p[0]: 0 for p in PATCHES}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    c["outputs"], c["execution_count"] = [], None
    s = "".join(c["source"])
    for a, b in PATCHES:
        hits[a] += s.count(a)
        s = s.replace(a, b)
    for a, b in OPTIONAL:
        s = s.replace(a, b)
    c["source"] = s.splitlines(keepends=True)
assert all(hits.values()), f"parches sin aplicar: {hits}"
print("parches", hits)
nb["cells"] = [code(fake)] + nb["cells"] + [code(report)]

out = HERE / slug
out.mkdir(exist_ok=True)
json.dump(nb, open(out / f"{slug}.ipynb", "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
meta.update(id=f"hctorheredia/{slug}", title=slug.replace("-", " "), code_file=f"{slug}.ipynb", is_private=True)
meta.pop("id_no", None); meta.pop("docker_image", None)
meta["dataset_sources"] = list(meta["dataset_sources"]) + sys.argv[4:]
json.dump(meta, open(out / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
print("escrito", out)
