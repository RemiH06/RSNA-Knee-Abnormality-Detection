"""Atención por corte y Grad-CAM del modelo para interpretar qué mira en cada hallazgo.

Para cada estudio gold evalúa TODAS las ventanas 2.5D posibles (no solo las 24 de la validación) y guarda
el peso de atención de cada ventana para los 12 hallazgos. Para los estudios de --examples además calcula
Grad-CAM en la ventana más atendida de cada hallazgo.

Uso (Pixilate):
  .venv\\Scripts\\python.exe scripts\\attention_maps.py --examples <uid> <uid>
Salida: results/attention_v2.npz (atención, sin imágenes) y results/attention_v2_cam.npz (cortes + Grad-CAM).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "runs" / "raptor_ft_coatnet_v2_swa.pt"))
ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus_v4"))
ap.add_argument("--train_csv", default=next((str(p) for p in (ROOT / "data").rglob("train.csv")), ""))
ap.add_argument("--res", type=int, default=384)
ap.add_argument("--chunk", type=int, default=6)
ap.add_argument("--vram_gb", type=float, default=4)
ap.add_argument("--examples", nargs="*", default=[])
ap.add_argument("--out", default=str(ROOT / "results" / "attention_v2"))
a = ap.parse_args()

os.environ["KNEE_CORPUS"] = os.path.abspath(a.corpus)
sys.path.insert(0, str(ROOT / "scripts"))
import train_raptor as tr  # noqa: E402  (después de fijar KNEE_CORPUS)

dev = "cuda"
if a.vram_gb:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory))

bb = tr.build_backbone(a.arch, pretrained=False)
model = tr.RaptorClassifier(bb, F_dim=bb.num_features)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
model.to(dev).eval()

vols, masks = tr._open_corpus()
ids = list(tr._open_ids())
train = pd.read_csv(a.train_csv)
gold = train.dropna(subset=tr.LAB).StudyInstanceUID.tolist()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
D = vols.shape[1]


def windows(row):
    """Todas las ventanas válidas del estudio, como en StudyWindows pero sin submuestrear."""
    valid = np.flatnonzero(masks[row] > 0)
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, D - 2))]
    vol = vols[row]
    x = torch.from_numpy(np.stack([vol[c - 1:c + 2] for c in cs]).astype(np.float32) / 255.0).to(dev)
    x = F.interpolate(x, size=(a.res, a.res), mode="bilinear", align_corners=False)
    return np.array(cs), (x - mean) / std


@torch.no_grad()
def encode(x):
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.cat([model.backbone(x[i:i + a.chunk]).float() for i in range(0, len(x), a.chunk)])


def head(f):
    h = model.norm(f)
    att = torch.softmax(model.att(h), dim=0)                     # (K,12)
    logits = (torch.einsum("kn,kf->nf", att, h) * model.clsW).sum(-1) + model.clsb
    return att, logits


feat_map = {}
last_stage = model.backbone.stages[-1]
last_stage.register_forward_hook(lambda m, i, o: feat_map.__setitem__("x", o))


def gradcam(x, f_all, k, j):
    """Grad-CAM del hallazgo j en la ventana k: se recalcula solo esa ventana con gradiente."""
    xk = x[k:k + 1].clone().requires_grad_(False)
    with torch.enable_grad():
        fk = model.backbone(xk)
        fmap = feat_map["x"]                                     # (1,C,h,w)
        fmap.retain_grad()
        f = torch.cat([f_all[:k], fk, f_all[k + 1:]])
        _, logits = head(f)
        model.zero_grad(set_to_none=True)
        logits[j].backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1))[0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


KMAX = D - 2
centers = np.full((len(gold), KMAX), -1, np.int16)
attn = np.zeros((len(gold), KMAX, 12), np.float32)
probs = np.zeros((len(gold), 12), np.float32)
cam_slices, cam_maps, cam_ids, cam_centers = [], [], [], []
for i, uid in enumerate(gold):
    row = ids.index(uid)
    cs, x = windows(row)
    f = encode(x)
    with torch.no_grad():
        at, lg = head(f)
    centers[i, :len(cs)] = cs
    attn[i, :len(cs)] = at.cpu().numpy()
    probs[i] = torch.sigmoid(lg).cpu().numpy()
    if uid in a.examples:
        top = at.argmax(0).cpu().numpy()                         # ventana más atendida por hallazgo
        cam_ids.append(uid); cam_centers.append(cs[top])
        cam_slices.append(np.stack([vols[row][cs[k]] for k in top]))
        cam_maps.append(np.stack([gradcam(x, f, int(k), j) for j, k in enumerate(top)]))
        torch.cuda.empty_cache()
    print(f"{i + 1}/{len(gold)} {len(cs)} ventanas", flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out + ".npz", ids=np.array(gold), centers=centers, attn=attn, probs=probs,
                    labels=np.array(tr.LAB))
if cam_ids:
    np.savez_compressed(a.out + "_cam.npz", ids=np.array(cam_ids), centers=np.array(cam_centers),
                        slices=np.array(cam_slices), cams=np.array(cam_maps), labels=np.array(tr.LAB))
print("ok", a.out)
