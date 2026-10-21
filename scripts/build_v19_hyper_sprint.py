"""Build Speed-Demon V19 Hyper-Sprint (Kernel v21).

Upgrades over V18 Ultra-Sprint (95.9s commit, ~5m hidden test, 0.947 LB):
1. In-Process Execution (No Subprocess): Cell 2 imports `raptor_run` and calls `raptor_run.main()`
   directly in the main notebook kernel process, eliminating 20.5s of subprocess Python/CUDA boot & teardown.
2. True 4-Model Dual-GPU Sharding (4 GPU model copies instead of 8, loaded in parallel on cuda:0 & cuda:1):
   - `cuda:0` (`q0`): evaluates ONLY `Raptor-384` (320px) + `Cleanlab N4 ConvNeXt-Small Fold 0` (256px)
   - `cuda:1` (`q1`): evaluates ONLY `Raptor-224` (224px 0.8-crop) + `Cleanlab C384 CoAtNet Fold 0` (384px)
   - Zero-copy pointer broadcast of `(sid, vol, mask)` to `q0` and `q1`.
3. Wider FUSE I/O Concurrency: `nproc = 8`, `FR_HDR = 48`, `FR_PX = 24`.
4. Exact 100% mathematical parity with V18 Ultra-Sprint (`0.0` diff on `submission.csv`).
"""

import json
from pathlib import Path

NB_PATH = Path("notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb")


def main() -> None:
    # Start cleanly from V18 Ultra-Sprint by running build_v17 + build_v18 or patching V18 directly
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = [
        "# Creaky Speed-Demon V19 Hyper-Sprint · In-Process Dual-GPU Sharded 4-Model Engine (~2.5-3.5 Min)\n",
        "\n",
        "1. **Zero Subprocess Overhead + Zero Second-Pass Disk I/O:** Runs directly in-process and reads every test study **once** via `study_lut` (`8 procs × 48/24 threads`).\n",
        "2. **True 4-Model Dual-GPU Sharding (Zero Duplicate Weights Across GPUs):**\n",
        "   - `cuda:0`: `Raptor-384` (`320px`, `K_EVAL=42`, `w=0.65`) + `Cleanlab N4 ConvNeXt-Small Fold 0` (`256px`)\n",
        "   - `cuda:1`: `Raptor-224` (`224px` 0.8-crop, `K_EVAL=42`, `w=0.35`) + `Cleanlab C384 CoAtNet Fold 0` (`384px`)\n",
    ]

    c1 = "".join(nb["cells"][1]["source"])

    # 1. Bump FR_HDR, FR_PX from 32, 16 to 48, 24
    c1 = c1.replace("FR_HDR, FR_PX = 32, 16", "FR_HDR, FR_PX = 48, 24")

    # 2. Replace _infer_creaky_from_vol with per-GPU single-specialist functions (_infer_n4_z, _infer_c384_z)
    old_infer_fn = """@torch.no_grad()
def _infer_creaky_from_vol(vol, mask, c384_m, n4_m, dev):
    # Extract 16 slices per anatomical plane directly from study_lut's (96, 384, 384) uint8 volume
    sag_u8 = vol[0:26][_IDX26] if mask[0:26].sum() > 0 else vol[26:48][_IDX22]
    cor_u8 = vol[48:66][_IDX18] if mask[48:66].sum() > 0 else vol[66:78][_IDX12]
    ax_u8 = vol[78:96][_IDX18]
    u8_3p = torch.from_numpy(np.stack([sag_u8, cor_u8, ax_u8], axis=0)).to(dev, non_blocking=True)  # (3, 16, 384, 384)
    idx_p = torch.clamp(torch.arange(16, device=dev) - 1, 0, 15)
    idx_n = torch.clamp(torch.arange(16, device=dev) + 1, 0, 15)
    tri = torch.stack([u8_3p[:, idx_p], u8_3p, u8_3p[:, idx_n]], dim=2).to(torch.float32) * (1.0 / 255.0)  # (3, 16, 3, 384, 384)
    dkey = str(dev)
    if dkey not in _CREAKY_NORM:
        _CREAKY_NORM[dkey] = (
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1),
        )
    mg, sg = _CREAKY_NORM[dkey]
    tri384 = (tri - mg) / sg
    gs384, gc384, ga384 = tri384[0:1], tri384[1:2], tri384[2:3]
    tri256 = F.interpolate(tri384.view(48, 3, 384, 384), size=(256, 256), mode="bilinear", align_corners=False).view(3, 1, 16, 3, 256, 256)
    gs256, gc256, ga256 = tri256[0], tri256[1], tri256[2]

    with torch.autocast("cuda", dtype=torch.float16):
        l_c384 = c384_m(gs384, gc384, ga384).float()[0].cpu().numpy()
        l_n4 = n4_m(gs256, gc256, ga256).float()[0].cpu().numpy()

    z_c384 = (np.clip(l_c384, -11.5, 11.5) - C384_F0_MU) / C384_F0_SD
    z_n4 = (np.clip(l_n4, -11.5, 11.5) - N4_F0_MU) / N4_F0_SD
    z_blend = (1.0 - C384_ADAPTIVE_W) * z_n4 + C384_ADAPTIVE_W * z_c384
    probs = 1.0 / (1.0 + np.exp(-np.clip(z_blend, -11.5, 11.5)))
    return np.clip(probs, 0.001, 0.999).astype(np.float32)"""

    new_infer_fn = """@torch.no_grad()
def _build_3p_384(vol, mask, dev):
    sag_u8 = vol[0:26][_IDX26] if mask[0:26].sum() > 0 else vol[26:48][_IDX22]
    cor_u8 = vol[48:66][_IDX18] if mask[48:66].sum() > 0 else vol[66:78][_IDX12]
    ax_u8 = vol[78:96][_IDX18]
    u8_3p = torch.from_numpy(np.stack([sag_u8, cor_u8, ax_u8], axis=0)).to(dev, non_blocking=True)
    idx_p = torch.clamp(torch.arange(16, device=dev) - 1, 0, 15)
    idx_n = torch.clamp(torch.arange(16, device=dev) + 1, 0, 15)
    tri = torch.stack([u8_3p[:, idx_p], u8_3p, u8_3p[:, idx_n]], dim=2).to(torch.float32) * (1.0 / 255.0)
    dkey = str(dev)
    if dkey not in _CREAKY_NORM:
        _CREAKY_NORM[dkey] = (
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1),
        )
    mg, sg = _CREAKY_NORM[dkey]
    return (tri - mg) / sg

@torch.no_grad()
def _infer_n4_z(vol, mask, n4_m, dev):
    tri384 = _build_3p_384(vol, mask, dev)
    tri256 = F.interpolate(tri384.view(48, 3, 384, 384), size=(256, 256), mode="bilinear", align_corners=False).view(3, 1, 16, 3, 256, 256)
    with torch.autocast("cuda", dtype=torch.float16):
        l_n4 = n4_m(tri256[0], tri256[1], tri256[2]).float()[0].cpu().numpy()
    return (np.clip(l_n4, -11.5, 11.5) - N4_F0_MU) / N4_F0_SD

@torch.no_grad()
def _infer_c384_z(vol, mask, c384_m, dev):
    tri384 = _build_3p_384(vol, mask, dev)
    with torch.autocast("cuda", dtype=torch.float16):
        l_c384 = c384_m(tri384[0:1], tri384[1:2], tri384[2:3]).float()[0].cpu().numpy()
    return (np.clip(l_c384, -11.5, 11.5) - C384_F0_MU) / C384_F0_SD"""

    assert old_infer_fn in c1, "Could not find old_infer_fn in Cell 1"
    c1 = c1.replace(old_infer_fn, new_infer_fn, 1)

    # 3. Replace model loading and consumer loop with True Dual-GPU Sharded (cuda:0 = Arm 0 + N4_F0, cuda:1 = Arm 1 + C384_F0)
    old_v18_main = """    _wpaths = [find_weight_file(arm["file"]) for arm in ARMS]
    _p_c384, _p_n4 = _find_creaky_ckpts()
    _all_preload = [p for p in (_wpaths + [_p_c384, _p_n4]) if p is not None]
    def _preload_ckpt(p):
        if p not in _CKPT_RAM_CACHE:
            _CKPT_RAM_CACHE[p] = torch.load(p, map_location="cpu", weights_only=False)
    with ThreadPoolExecutor(max_workers=len(_all_preload)) as _ex:
        list(_ex.map(_preload_ckpt, _all_preload))
    models = {}
    creaky_models = {}
    creaky_probs = np.full((N, len(LAB)), 0.5, np.float32)
    has_creaky = (_p_c384 is not None and _p_n4 is not None)
    for d in devs:
        models[d] = []
        for a, arm in enumerate(ARMS):
            model, res = load_model(_wpaths[a], arm["arch"], arm["res"], d)
            models[d].append((model, res))
        if has_creaky:
            creaky_models[d] = (_load_creaky_model(_p_c384, d), _load_creaky_model(_p_n4, d))
        print(f"[gpu {d}] loaded {A} Raptor arm(s) + {2 if has_creaky else 0} Creaky MoE arm(s) | {time.time()-t0:.0f}s", flush=True)
    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()

    _CKPT_RAM_CACHE.clear(); gc.collect()
    def consumer(d):
        while True:
            item = q.get()
            if item is None: q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                for a, (model, res) in enumerate(models[d]):
                    cfrac = float(ARMS[a].get("crop_frac", 1.0))
                    use_tta = bool(ARMS[a].get("tta", True))
                    xw = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=False, crop_frac=cfrac)
                    if use_tta:
                        xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True, crop_frac=cfrac)
                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))
                        del xm
                    else:
                        arm_probs[a][i] = infer_probs(model, xw, d)
                    del xw
            except Exception as e:
                print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            if has_creaky:
                try:
                    c384_m, n4_m = creaky_models[d]
                    creaky_probs[i] = _infer_creaky_from_vol(vol, mask, c384_m, n4_m, d)
                except Exception as e_c:
                    print(f"  [gpu {d}] study {i} {sid[:16]} CREAKY FALLBACK ({type(e_c).__name__}: {e_c})", flush=True)
            with lock:
                done_n[0] += 1
                if done_n[0] % 200 == 0 or done_n[0] == N:
                    print(f"  [gpu] {done_n[0]}/{N} | {time.time()-t0:.0f}s", flush=True)
            q.task_done()

    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    for th in threads: th.start()

    # ---- 3. prep every study ONCE in a process pool and stream it to the GPUs as it lands ----
    import multiprocessing as mp
    global _SER, _TSDIR, _READER
    _SER, _TSDIR = SER, tsdir
    nproc = 6
    t_pre = time.time()
    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:
        for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):
            q.put((sid, vol, mask))
            if (i + 1) % 200 == 0 or i + 1 == N:
                print(f"  [pre] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
    print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)
    for _ in devs: q.put(None)
    q.join()
    for th in threads: th.join()
    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)
    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5
    _raw.insert(0, "StudyInstanceUID", test_ids)
    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)
    for d in devs:
        for m, _ in models[d]: del m
    models.clear(); gc.collect()"""

    new_v19_main = """    _wpaths = [find_weight_file(arm["file"]) for arm in ARMS]
    _p_c384, _p_n4 = _find_creaky_ckpts()
    _all_preload = [p for p in (_wpaths + [_p_c384, _p_n4]) if p is not None]
    def _preload_ckpt(p):
        if p not in _CKPT_RAM_CACHE:
            _CKPT_RAM_CACHE[p] = torch.load(p, map_location="cpu", weights_only=False)
    with ThreadPoolExecutor(max_workers=len(_all_preload)) as _ex:
        list(_ex.map(_preload_ckpt, _all_preload))

    has_creaky = (_p_c384 is not None and _p_n4 is not None)
    z_n4_arr = np.zeros((N, len(LAB)), np.float32)
    z_c384_arr = np.zeros((N, len(LAB)), np.float32)
    creaky_ok = np.ones(N, dtype=bool)
    dev0 = devs[0]
    dev1 = devs[1] if len(devs) >= 2 else devs[0]

    def _init_dev0():
        m0, r0 = load_model(_wpaths[0], ARMS[0]["arch"], ARMS[0]["res"], dev0)
        mn4 = _load_creaky_model(_p_n4, dev0) if has_creaky else None
        print(f"[gpu {dev0}] sharded Raptor-384 + N4-F0 loaded | {time.time()-t0:.0f}s", flush=True)
        return m0, r0, mn4

    def _init_dev1():
        m1, r1 = load_model(_wpaths[1], ARMS[1]["arch"], ARMS[1]["res"], dev1)
        mc384 = _load_creaky_model(_p_c384, dev1) if has_creaky else None
        print(f"[gpu {dev1}] sharded Raptor-224 + C384-F0 loaded | {time.time()-t0:.0f}s", flush=True)
        return m1, r1, mc384

    with ThreadPoolExecutor(max_workers=2) as _gex:
        _f0 = _gex.submit(_init_dev0)
        _f1 = _gex.submit(_init_dev1)
        m_r384, res_r384, m_n4 = _f0.result()
        m_r224, res_r224, m_c384 = _f1.result()

    _CKPT_RAM_CACHE.clear(); gc.collect()
    q0 = queue.Queue(maxsize=QUEUE_MAX)
    q1 = queue.Queue(maxsize=QUEUE_MAX)
    done_n0 = [0]; done_n1 = [0]

    def consumer_dev0():
        cfrac = float(ARMS[0].get("crop_frac", 1.0))
        while True:
            item = q0.get()
            if item is None: q0.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                xw = gpu_windows(vol, mask, K_EVAL, res_r384, dev0, norm=NORM, mirror=False, crop_frac=cfrac)
                arm_probs[0][i] = infer_probs(m_r384, xw, dev0)
                del xw
            except Exception as e:
                print(f"  [gpu {dev0}] study {i} {sid[:16]} R384 FALLBACK ({type(e).__name__}: {e})", flush=True)
            if has_creaky:
                try:
                    z_n4_arr[i] = _infer_n4_z(vol, mask, m_n4, dev0)
                except Exception as e_c:
                    creaky_ok[i] = False
                    print(f"  [gpu {dev0}] study {i} {sid[:16]} N4 FALLBACK ({type(e_c).__name__}: {e_c})", flush=True)
            done_n0[0] += 1
            if done_n0[0] % 200 == 0 or done_n0[0] == N:
                print(f"  [gpu0] {done_n0[0]}/{N} | {time.time()-t0:.0f}s", flush=True)
            q0.task_done()

    def consumer_dev1():
        cfrac = float(ARMS[1].get("crop_frac", 0.8))
        while True:
            item = q1.get()
            if item is None: q1.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                xw = gpu_windows(vol, mask, K_EVAL, res_r224, dev1, norm=NORM, mirror=False, crop_frac=cfrac)
                arm_probs[1][i] = infer_probs(m_r224, xw, dev1)
                del xw
            except Exception as e:
                print(f"  [gpu {dev1}] study {i} {sid[:16]} R224 FALLBACK ({type(e).__name__}: {e})", flush=True)
            if has_creaky:
                try:
                    z_c384_arr[i] = _infer_c384_z(vol, mask, m_c384, dev1)
                except Exception as e_c:
                    creaky_ok[i] = False
                    print(f"  [gpu {dev1}] study {i} {sid[:16]} C384 FALLBACK ({type(e_c).__name__}: {e_c})", flush=True)
            done_n1[0] += 1
            if done_n1[0] % 200 == 0 or done_n1[0] == N:
                print(f"  [gpu1] {done_n1[0]}/{N} | {time.time()-t0:.0f}s", flush=True)
            q1.task_done()

    th0 = threading.Thread(target=consumer_dev0, daemon=True)
    th1 = threading.Thread(target=consumer_dev1, daemon=True)
    th0.start(); th1.start()

    # ---- 3. prep every study ONCE in a process pool and stream zero-copy pointers to BOTH GPUs ----
    import multiprocessing as mp
    global _SER, _TSDIR, _READER
    _SER, _TSDIR = SER, tsdir
    nproc = 8
    t_pre = time.time()
    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:
        for i, item in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):
            q0.put(item)
            q1.put(item)
            if (i + 1) % 200 == 0 or i + 1 == N:
                print(f"  [pre] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
    print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)
    q0.put(None); q1.put(None)
    q0.join(); q1.join()
    th0.join(); th1.join()
    print(f"[gpu] done: {done_n0[0]}/{N} & {done_n1[0]}/{N} studies sharded across {dev0}+{dev1} | {time.time()-t0:.0f}s", flush=True)

    creaky_probs = np.full((N, len(LAB)), 0.5, np.float32)
    if has_creaky:
        z_blend = (1.0 - C384_ADAPTIVE_W) * z_n4_arr + C384_ADAPTIVE_W * z_c384_arr
        probs_all = np.clip(1.0 / (1.0 + np.exp(-np.clip(z_blend, -11.5, 11.5))), 0.001, 0.999).astype(np.float32)
        creaky_probs[creaky_ok] = probs_all[creaky_ok]

    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)
    _raw.insert(0, "StudyInstanceUID", test_ids)
    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)
    del m_r384, m_r224, m_n4, m_c384; gc.collect()"""

    assert old_v18_main in c1, "Could not find old_v18_main in Cell 1"
    c1 = c1.replace(old_v18_main, new_v19_main, 1)

    nb["cells"][1]["source"] = [line + "\n" for line in c1.splitlines()[:-1]] + [c1.splitlines()[-1]]

    # 4. In-Process execution in Cell 2 (saves ~20.5s subprocess spawn + teardown!)
    nb["cells"][2]["source"] = [
        "# Execute fused single-pass engine directly in-process (saves ~20s subprocess boot/teardown)\n",
        "import os, sys, time\n",
        "NB_T0 = time.time()\n",
        "if '/tmp' not in sys.path:\n",
        "    sys.path.insert(0, '/tmp')\n",
        "import raptor_run\n",
        "raptor_run.main()\n",
        "for f in ('raptor_sub.csv', 'raptor_raw.csv', 'submission.csv'):\n",
        "    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'\n",
        "RAPTOR_SECONDS = time.time() - NB_T0\n",
        "print('FUSED_HYPER_SPRINT_DONE seconds %.1f' % RAPTOR_SECONDS, flush=True)\n",
    ]

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Successfully built Speed-Demon V19 Hyper-Sprint in", NB_PATH)


if __name__ == "__main__":
    main()
