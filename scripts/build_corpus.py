"""Genera el corpus de entrenamiento de 64 cortes desde data/archive.zip, sin descomprimirlo a disco.

Preprocesamiento copiado de build_study() del notebook público de inferencia
(dreaddevelopment/knee-mri-twelve-findings-from-a-single-model, Apache 2.0): 5 slots por plano y
secuencia, cortes del 6 al 94 % de cada serie ordenados por geometría, ventana de intensidad
2-98 % por serie, recorte de 140 mm y 336 px. Lo único distinto es que lee del zip en vez de
carpetas; si cambias algo aquí, cámbialo igual en el notebook de inferencia.

Salida en data/corpus_v4/: all_vols.npy (N,64,336,336) uint8, all_masks.npy (N,64), all_ids.npy.
Reanudable: done.npy marca los estudios ya escritos.
Uso: python scripts/build_corpus.py [--workers 7] [--limit N] [--out data/corpus_v4]
"""
import argparse, io, os, sys, time, zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "archive.zip"
OUT = ROOT / "data" / "corpus_v4"
IMG, CROP_MM = 336, 140.0
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)  # 64

_zf = None
_members = None


def _init(members):
    global _zf, _members
    _zf = zipfile.ZipFile(ZIP)
    _members = members


def order_and_meta(names):
    recs, ps_list = [], []
    for f in names:
        try:
            with _zf.open(f) as fh:
                h = pydicom.dcmread(fh, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps); recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med_ps


def read_px(f):
    d = pydicom.dcmread(io.BytesIO(_zf.read(f)))
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(job):
    i, sid, rows = job
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set(); bad = 0
    for plane, fluid, k in SLOTS:
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta(_members.get((sid, r["SeriesInstanceUID"]), []))
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps); bad += 1
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return i, vol, mask, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    ser = pd.read_csv(next((ROOT / "data").rglob("train_series.csv")), dtype=str)
    ser["Fluid_Sensitive"] = ser["Fluid_Sensitive"].astype(int)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}
    ids = np.array(sorted(SER))[: a.limit or None]

    members = defaultdict(list)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[0] == "train_series" and name.endswith(".dcm"):
                members[(parts[1], parts[2])].append(name)
    print(f"{len(ids)} estudios | {len(members)} series en el zip", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    N = len(ids)
    vp, mp, dp = out / "all_vols.npy", out / "all_masks.npy", out / "done.npy"
    if vp.exists():
        vols = np.load(vp, mmap_mode="r+"); masks = np.load(mp); done = np.load(dp)
        if vols.shape[0] != N or not np.array_equal(np.load(out / "all_ids.npy"), ids):
            sys.exit("el corpus existente no coincide con estos ids; bórralo o usa el mismo --limit")
    else:
        vols = np.lib.format.open_memmap(vp, mode="w+", dtype=np.uint8, shape=(N, MAXS, IMG, IMG))
        masks = np.zeros((N, MAXS), np.uint8); done = np.zeros(N, bool)
        np.save(out / "all_ids.npy", ids)

    todo = [(i, ids[i], SER[ids[i]]) for i in range(N) if not done[i]]
    print(f"faltan {len(todo)} de {N}", flush=True)
    t0, bad_total = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(dict(members),)) as pool:
        for n, (i, vol, mask, bad) in enumerate(pool.imap_unordered(build_study, todo, chunksize=2), 1):
            vols[i] = vol; masks[i] = mask; done[i] = True; bad_total += bad
            if n % 100 == 0 or n == len(todo):
                vols.flush(); np.save(mp, masks); np.save(dp, done)
                el = time.time() - t0
                print(f"{n}/{len(todo)} | {el / 60:.1f} min | quedan ~{el / n * (len(todo) - n) / 60:.0f} min"
                      f" | cortes ilegibles {bad_total}", flush=True)
    empty = int((masks.sum(1) == 0).sum())
    print(f"listo: {out} | estudios sin ningún corte {empty} | cortes ilegibles {bad_total}", flush=True)


if __name__ == "__main__":
    main()
