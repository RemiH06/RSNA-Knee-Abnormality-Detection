

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
