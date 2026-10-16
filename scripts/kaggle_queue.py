"""Cola de kernels de Kaggle: los sube uno por uno conforme se libera una de las 2 sesiones de GPU.

Hace lo mismo que `kaggle kernels push -p <carpeta>`, pero con kagglesdk (viene con kagglehub), porque el CLI de
kaggle 2.x pide Python >= 3.11 y Pixilate tiene 3.10. Así la cola corre en Pixilate aunque Aerilate esté apagada.
Reanudable: guarda en <carpeta>/.pushed las que ya subió y no las repite.
Uso: scripts/run_with_token.ps1 scripts/kaggle_queue.py kaggle/timing/kad-teacher-1 kaggle/timing/kad-teacher-2 ...
"""
import json, sys, time
from pathlib import Path

from kagglehub.clients import build_kaggle_client
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

WAIT_S = 300


def push(folder: Path):
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    body = (folder / meta["code_file"]).read_text(encoding="utf-8")
    if meta["kernel_type"] == "notebook":
        nb = json.loads(body)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
            if isinstance(c.get("source"), list):
                c["source"] = "".join(c["source"])
        body = json.dumps(nb)
    r = ApiSaveKernelRequest()
    r.slug = meta["id"]
    r.new_title = meta.get("title")
    r.text = body
    r.language = meta["language"]
    r.kernel_type = meta["kernel_type"]
    r.is_private = meta.get("is_private", True)
    r.enable_gpu = meta.get("enable_gpu", False)
    r.enable_tpu = meta.get("enable_tpu", False)
    r.enable_internet = meta.get("enable_internet", True)
    r.dataset_data_sources = meta.get("dataset_sources", [])
    r.competition_data_sources = meta.get("competition_sources", [])
    r.kernel_data_sources = meta.get("kernel_sources", [])
    r.model_data_sources = meta.get("model_sources", [])
    r.category_ids = meta.get("keywords", [])
    r.machine_shape = meta.get("machine_shape")
    with build_kaggle_client() as k:
        return k.kernels.kernels_api_client.save_kernel(r)


for f in map(Path, sys.argv[1:]):
    mark = f / ".pushed"
    if mark.exists():
        print(time.strftime("%H:%M"), "ya subido", f.name, flush=True)
        continue
    while True:
        try:
            resp = push(f)
            err = getattr(resp, "error", None)
        except Exception as e:
            err = repr(e)
        if not err:
            mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(time.strftime("%H:%M"), "subido", f.name, getattr(resp, "url", ""), flush=True)
            time.sleep(60)
            break
        print(time.strftime("%H:%M"), "espera", f.name, "|", str(err)[:160], flush=True)
        time.sleep(WAIT_S)
print("cola vacía", flush=True)
