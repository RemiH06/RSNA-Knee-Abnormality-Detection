"""Arma kaggle/blend/kad-knee-blend.ipynb: fork de heliosli/this-one-is-old-version-0-957-new-one-you-can-use
con nuestro modelo (kaggle/infer_v2/kad-knee-infer.py) como etapa extra.

Nuestra etapa corre en segundo plano en la GPU 1 con prioridad de CPU baja mientras las etapas
públicas 1 y 1b usan la GPU 0; si no terminó cuando arranca la etapa 2 se mata y el blend queda
igual al público (KAD_SKIPPED). Así nunca cuesta tiempo a la etapa 2 ni tumba el envío.
Uso: python kaggle/blend/build.py <notebook público .ipynb> <peso de nuestro modelo, 0 = público tal cual>
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src, w = sys.argv[1], float(sys.argv[2])
nb = json.load(open(src, encoding="utf-8"))
for c in nb["cells"]:
    c.pop("outputs", None); c.pop("execution_count", None)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(keepends=True)}

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

ours = (HERE.parent / "infer_v2" / "kad-knee-infer.py").read_text(encoding="utf-8")
launch = f'''# [kad] Nuestro CoAtNet v2 en segundo plano: GPU 1, nice 15, 2 hilos. Se corta al empezar la etapa 2.
import os, sys, subprocess, time
KAD_W = {w}
KAD_PROC = None
if KAD_W > 0:
    import torch
    if torch.cuda.device_count() >= 2:
        _env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", KAD_RAW_OUT="/kaggle/working/kad_raw.csv",
                    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        KAD_PROC = subprocess.Popen(["nice", "-n", "15", sys.executable, "-u", "/tmp/kad_run.py"], env=_env,
                                    stdout=open("/kaggle/working/kad_log.txt", "w"), stderr=subprocess.STDOUT)
        print("KAD_STARTED pid", KAD_PROC.pid, flush=True)
    else:
        print("KAD_SKIPPED una sola GPU", flush=True)
'''
stop = '''# [kad] La etapa 2 tiene la máquina para ella sola: si nuestra etapa no terminó, se mata y no entra al blend.
KAD_OK = False
if KAD_PROC is not None:
    if KAD_PROC.poll() is None:
        KAD_PROC.kill(); KAD_PROC.wait()
        print("KAD_SKIPPED no terminó antes de la etapa 2", flush=True)
    else:
        KAD_OK = KAD_PROC.returncode == 0 and os.path.isfile("/kaggle/working/kad_raw.csv")
        print("KAD_DONE rc", KAD_PROC.returncode, "ok", KAD_OK, flush=True)
    print(open("/kaggle/working/kad_log.txt").read()[-1500:], flush=True)
'''
blend = '''
# [kad] Nuestro modelo entra con peso KAD_W sobre el blend público, por rangos y sobre sus estudios no fallidos.
if KAD_OK:
    kad = pd.read_csv(WORK / 'kad_raw.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
    kad_fail = kad.isna().any(axis=1).values
    if kad_fail.sum() > 0.03 * n:
        print('KAD_SKIPPED fallaron %d de %d estudios' % (kad_fail.sum(), n), flush=True)
    else:
        for j, c in enumerate(labels):
            rk = pct_on(kad[c].values, ~kad_fail)
            out[~kad_fail, j] = (1 - KAD_W) * out[~kad_fail, j] + KAD_W * rk[~kad_fail]
        sub = pd.DataFrame(out.astype(np.float32), columns=labels)
        sub.insert(0, 'StudyInstanceUID', test_ids)
        sub = sub[cols]
        assert np.isfinite(sub[labels].values).all()
        sub.to_csv(WORK / 'submission.csv', index=False)
        print('KAD_BLENDED w', KAD_W, 'failed', int(kad_fail.sum()), flush=True)
'''
cells = nb["cells"]
assert "".join(cells[8]["source"]).startswith("# Stage 3"), "estructura del notebook cambió"
assert "".join(cells[6]["source"]).startswith("ARM = "), "estructura del notebook cambió"
cells[8]["source"] = ("".join(cells[8]["source"]) + blend).splitlines(keepends=True)
intro = md(f"""# KAD: blend público 0.957 + nuestro CoAtNet v2

Fork de [heliosli/this-one-is-old-version-0-957-new-one-you-can-use](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-957-new-one-you-can-use), sin cambios en sus etapas.
Se agrega nuestro modelo (CoAtNet-384 v2, 0.919 solo en el LB público; repo RemiH06/RSNA-Knee-Abnormality-Detection) con peso {w} sobre el blend, por rangos.
Créditos de los modelos públicos en la celda siguiente.
""")
nb["cells"] = [intro] + cells[:2] + [code("%%writefile /tmp/kad_run.py\n" + ours), code(launch)] + cells[2:6] + [code(stop)] + cells[6:]
out = HERE / "kad-knee-blend.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print("escrito", out, len(nb["cells"]), "celdas")
