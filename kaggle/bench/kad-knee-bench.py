#!/usr/bin/env python3
"""KAD: inferencia del modelo CoAtNet v2 (RSNA Knee Abnormality Detection).

Basado en "Knee MRI: twelve findings from a single model" de dreaddevelopment (Apache 2.0):
https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
Preprocesamiento y modelo sin cambios; solo cambian los pesos, entrenados por nosotros en una
RTX 3060 (scripts/train_raptor.py del repo) sobre un corpus propio con la misma geometría:
64 cortes por estudio en 5 slots, del 6 al 94 % de cada serie, recorte de 140 mm a 336 px.
Etiquetas de entrenamiento: stevenleehans/rsna-knee-llm-report-labels (v4).
Pesos: SWA de los epochs 7-9 de coatnet_v2, AUC macro 0.8916 en los 58 estudios gold.
"""
import os, sys, glob, time, json, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 42   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Un solo modelo: CoAtNet-384 v2 (SWA ep7-9), entrenado sobre data/corpus_v4.
ARMS = [
    {"file": "raptor_ft_coatnet_v2_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
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
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/kad-knee-weights/{fname}",
              f"/kaggle/input/datasets/hctorheredia/kad-knee-weights/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower():
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")


def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
    N = len(test_ids); A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]
    failed = np.zeros(N, bool)

    # SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak system RAM ==
    # one model == the single-arm champion's footprint (which graded fine at 0.879). Holding both
    # arms simultaneously OOM'd system RAM on the full hidden test. Each study is re-preprocessed
    # per arm (build_study is cheap vs inference) and every per-study buffer is freed. Same models,
    # same windowing, same rank-mean blend -> identical 0.8893 result, just serialized.
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        model, res = load_model(wp, arm["arch"], arm["res"], dev)
        print(f"[arm {a}] loaded {arm['file']} | res {res} | {time.time()-t0:.0f}s", flush=True)
        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                arm_probs[a][i] = infer_probs(model, xw, dev)
                del vol, mask, xw
            except Exception as e:
                failed[i] = True
                print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [arm {a}] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)

    # Modo etapa de un blend: KAD_RAW_OUT guarda las probabilidades crudas (un estudio fallido queda vacío,
    # para que el blend lo excluya) en vez de submission.csv.
    raw_out = os.environ.get("KAD_RAW_OUT")
    if raw_out:
        raw = pd.DataFrame(arm_probs[0], columns=LAB)
        raw.loc[failed, LAB] = np.nan
        raw.insert(0, "StudyInstanceUID", test_ids)
        raw.to_csv(raw_out, index=False)
        print("wrote", raw_out, "| failed", int(failed.sum()), f"| DONE {time.time()-t0:.0f}s", flush=True)
        return

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)




# ============================================================================
# KAD bench: tiempo y AUC contra K (ventanas por estudio) sobre los 58 estudios gold de train,
# en el mismo hardware del envío. Fase A: secuencial en una GPU (como el notebook actual), con
# tiempos separados de lectura DICOM, armado de ventanas y GPU; las features de cada ventana se
# calculan una vez y la cabeza se evalúa para cada K. Fase B: tubería paralela (lectura en
# procesos + 2 GPU) con K = 42, solo para medir el tiempo de pared.
# ============================================================================
import multiprocessing as mp
KS = [4, 6, 8, 12, 16, 20, 24, 32, 42, 62]
_W = {}


def _all_centers(mask, D):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    return cs or [max(1, min((lo + hi) // 2, D - 2))]


def _pick(cs, k):                                  # mismos índices que _eval_centers
    return np.linspace(0, len(cs) - 1, k).round().astype(int)


def _windows(vol, centers, res):
    D = vol.shape[0]
    cc = [max(1, min(c, D - 2)) for c in centers]
    tri = np.stack([vol[[c + o for c in cc]] for o in (-1, 0, 1)], 1)
    t = torch.from_numpy(tri.astype(np.float32) / 255.0)
    if t.shape[-1] != res:
        t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return (t - _MEAN) / _STD


def _init_worker(ser, tsdir):
    _W.update(ser=ser, tsdir=tsdir, reader=_make_reader())


def _build(sid):
    t = time.time()
    try:
        vol, mask = build_study(sid, _W["ser"], _W["tsdir"], _W["reader"])
        return sid, vol, mask, time.time() - t
    except Exception:
        return sid, None, None, time.time() - t


def _gpu_half(gpu, sids, ser, tsdir, wp, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_num_threads(1)
    model, res = load_model(wp, ARMS[0]["arch"], 384, "cuda")
    t0 = time.time(); n = 0
    with mp.get_context("fork").Pool(2, _init_worker, (ser, tsdir)) as pool:
        for sid, vol, mask, _ in pool.imap(_build, sids):
            if vol is not None:
                infer_probs(model, eval_windows(vol, mask, k=K_EVAL, res=res), "cuda"); n += 1
    out_q.put((gpu, n, time.time() - t0))


def bench():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    T0 = time.time()
    ROOT = find_test_root()
    tr = pd.read_csv(ROOT + "/train.csv"); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr.dropna(subset=LAB).reset_index(drop=True)
    ids = gold["StudyInstanceUID"].tolist(); Y = gold[LAB].values.astype(int)
    ser = pd.read_csv(ROOT + "/train_series.csv").astype({"StudyInstanceUID": str, "SeriesInstanceUID": str})
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    tsdir = ROOT + "/train_series"
    print(f"gold {len(ids)} | cpus {os.cpu_count()} | gpus {torch.cuda.device_count()}", flush=True)
    wp = find_weight_file(ARMS[0]["file"])
    model, res = load_model(wp, ARMS[0]["arch"], 384, "cuda")
    reader = _make_reader()

    # ---- Fase A: secuencial, como el notebook actual ----
    P = {k: np.full((len(ids), len(LAB)), np.nan) for k in KS}
    rows = []
    for i, sid in enumerate(ids):
        t = time.time(); vol, mask = build_study(sid, SER, tsdir, reader); t_dec = time.time() - t
        # tiempo real del camino actual (K = 42, ventanas en CPU, todo el modelo en fp16)
        torch.cuda.synchronize(); t = time.time()
        xw = eval_windows(vol, mask, k=K_EVAL, res=res); t_prep = time.time() - t
        t = time.time(); infer_probs(model, xw, "cuda"); torch.cuda.synchronize(); t_gpu = time.time() - t
        # features de todas las ventanas posibles, una vez; la cabeza para cada K
        cs = _all_centers(mask, vol.shape[0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x = _windows(vol, cs, res).cuda()
            feats = torch.cat([model.encode(x[None, j:j + 24]) for j in range(0, len(cs), 24)], 1)
            for k in KS:
                sel = torch.as_tensor(_pick(cs, min(k, len(cs))), device="cuda")
                P[k][i] = torch.sigmoid(model.head(feats[:, sel]).float())[0].cpu().numpy()
        rows.append(dict(sid=sid, n_cs=len(cs), t_dec=t_dec, t_prep=t_prep, t_gpu42=t_gpu))
        print(f"  {i+1}/{len(ids)} dec {t_dec:.1f}s prep {t_prep:.2f}s gpu42 {t_gpu:.2f}s cs {len(cs)}", flush=True)
    tm = pd.DataFrame(rows).iloc[1:]                 # el primero lleva el calentamiento de cuDNN
    # tiempo de GPU por ventana, para el modelo de costo T(K)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            model(torch.randn(1, 42, 3, res, res, device="cuda"))
    torch.cuda.synchronize(); per_win = (time.time() - t) / (5 * 42)

    rng = np.random.default_rng(0)

    def macro(Pk, idx):
        return np.mean([roc_auc_score(Y[idx, j], Pk[idx, j]) for j in range(len(LAB)) if len(set(Y[idx, j])) > 1])
    allidx = np.arange(len(ids))
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(500)]
    base = np.array([macro(P[42], b) for b in boots])
    res_k = []
    for k in KS:
        d = np.array([macro(P[k], b) for b in boots]) - base
        res_k.append(dict(K=k, auc=float(macro(P[k], allidx)), d_vs42=float(d.mean()),
                          d_lo=float(np.percentile(d, 2.5)), d_hi=float(np.percentile(d, 97.5))))
    del model; torch.cuda.empty_cache()

    # ---- Fase B: tubería paralela (lectura en 2 procesos por GPU, una mitad de estudios por GPU) ----
    ctx = mp.get_context("spawn"); ng = torch.cuda.device_count(); q = ctx.Queue()
    t = time.time()
    procs = [ctx.Process(target=_gpu_half, args=(g, ids[g::ng], SER, tsdir, wp, q)) for g in range(ng)]
    for p in procs: p.start()
    got = [q.get() for _ in procs]
    for p in procs: p.join()
    wall_par = time.time() - t

    out = dict(n=len(ids), cpus=os.cpu_count(), gpus=ng,
               seq_per_study=dict(dec=float(tm.t_dec.mean()), prep=float(tm.t_prep.mean()), gpu42=float(tm.t_gpu42.mean())),
               seq_wall=float(tm[["t_dec", "t_prep", "t_gpu42"]].sum(1).mean() * len(ids)),
               par_wall=wall_par, par_detail=got, gpu_per_window=per_win, auc_vs_k=res_k)
    json.dump(out, open("/kaggle/working/bench.json", "w"), indent=1, default=float)
    pd.DataFrame(rows).to_csv("/kaggle/working/bench_times.csv", index=False)
    pd.DataFrame(res_k).to_csv("/kaggle/working/bench_auc_k.csv", index=False)
    print(json.dumps(out, indent=1, default=float), flush=True)
    print(f"DONE {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    bench()
