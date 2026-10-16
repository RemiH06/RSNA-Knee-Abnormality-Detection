"""Arma kaggle/blend_r6s6/: el fork R6 (kaggle/blend_r6) + S6 de nartaa (nartaa/rsna-knee-s6fall-solo) como etapa opcional.

S6 corre después de R6, con el tiempo que sobre del límite de 9 h. Su componente A (el mismo Raptor 0.949 de nuestra
etapa 1) no se vuelve a correr: se lee de raptor_sub.csv. Si S6 falla o no cabe, la combinación queda igual a la de R6
(S6_SKIPPED). Final = (1 - W_S6) x blend R6 + W_S6 x rank(S6), por hallazgo.
Uso: python kaggle/blend_r6s6/build.py <rsna-knee-s6fall-solo.py> <W_S6>
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
s6_path, w = Path(sys.argv[1]), float(sys.argv[2])
nb = json.load(open(HERE.parent / "blend_r6" / "kad-knee-blend-r6.ipynb", encoding="utf-8"))
meta = json.load(open(HERE.parent / "blend_r6" / "kernel-metadata.json", encoding="utf-8"))
s6 = s6_path.read_text(encoding="utf-8")


def patch(text, a, b):
    assert text.count(a) == 1, a
    return text.replace(a, b)


s6 = patch(s6, 'GUARD_MAX_S = 480 * 60',
           'GUARD_MAX_S = int(os.environ.get("KAD_S6_GUARD_S", 480 * 60))   # [kad] el tiempo que le deja el notebook')
s6 = patch(s6, '    I.comp = find_mount(COMP, "test.csv")',
           '    I.comp = Path(os.environ["KAD_FAKE_ROOT"]) if os.environ.get("KAD_FAKE_ROOT") else find_mount(COMP, "test.csv")')
s6 = patch(s6, '        A, a_info = run_a(I)', '''        # [kad] A = nuestra etapa 1 (el mismo Raptor 0.949), ya calculada: no se vuelve a correr
        _a = pd.read_csv(WORK / "raptor_sub.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        A, a_info = _a.loc[I.test_ids, LAB].to_numpy(np.float64), {"reused": "raptor_sub.csv"}''')


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(keepends=True)}


stage = f'''# [kad] Etapa S6 (nartaa, ConvNeXt-small con todos los datos, 0.950 solo): opcional, con el tiempo que sobra.
import os, sys, time, shutil, subprocess
W_S6 = {w}
S6_OK = False
_left = 9 * 3600 - (time.time() - NB_T0) - 25 * 60          # margen para la mezcla y el cierre
print('S6_BUDGET seconds %.0f' % _left, flush=True)
if W_S6 > 0 and _left > 30 * 60:
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/s6_run.py'], timeout=_left + 300,
                             env=dict(os.environ, KAD_S6_GUARD_S=str(int(_left)))).returncode
    except subprocess.TimeoutExpired:
        _rc = 'timeout'
    _sub = WORK / 'submission.csv'
    if _rc == 0 and _sub.is_file():
        shutil.move(str(_sub), str(WORK / 's6_sub.csv')); S6_OK = True
    elif _sub.exists():
        _sub.unlink()
    print('S6_DONE rc', _rc, 'ok', S6_OK, 'seconds %.0f' % (time.time() - _t), flush=True)
else:
    print('S6_SKIPPED sin tiempo', flush=True)
'''

blend = '''
# [kad] S6 entra con peso W_S6 sobre la combinación R6, por rangos.
if S6_OK:
    _s6 = pd.read_csv(WORK / 's6_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
    assert np.isfinite(_s6.values).all()
    for j, c in enumerate(labels):
        out[:, j] = (1 - W_S6) * out[:, j] + W_S6 * pct(_s6[c].values)
    sub = pd.DataFrame(out.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    sub = sub[cols]
    assert np.isfinite(sub[labels].values).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    print('S6_BLENDED w', W_S6, flush=True)
'''

cells = nb["cells"]
assert "".join(cells[8]["source"]).startswith("# Stage 3"), "estructura del notebook cambió"
cells[8]["source"] = ("".join(cells[8]["source"]) + blend).splitlines(keepends=True)
nb["cells"] = cells[:8] + [code("%%writefile /tmp/s6_run.py\n" + s6), code(stage)] + cells[8:]
json.dump(nb, open(HERE / "kad-knee-blend-r6s6.ipynb", "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
meta.update(id="hctorheredia/kad-knee-blend-r6s6", title="KAD knee blend r6s6", code_file="kad-knee-blend-r6s6.ipynb")
meta["dataset_sources"] = list(meta["dataset_sources"]) + ["nartaa/rsna-knee-s6fall-weights"]
json.dump(meta, open(HERE / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
print("escrito", HERE, "W_S6", w)
