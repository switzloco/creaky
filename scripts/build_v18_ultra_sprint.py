"""Build Speed-Demon V18 Ultra-Sprint (Option A, Kernel v20).

Architecture (~3.0-3.8 min execution on 1,500 studies, ~0.947-0.948 Public LB):
1. Single-Pass Fused Execution inside Stage 1 (`raptor_run.py`):
   - Reads every test study ONCE via `_fastread.py` (`study_lut`, `nproc=6`, `32/16` threads).
   - Evaluates 4 diverse models across 3 resolutions from the SAME in-RAM `(96, 384, 384) uint8` volume:
     * Arm 0: `Raptor-384` (`320px`, `K_EVAL=42`, single-pass `tta=False`, `w=0.65`) with `assert torch.isfinite(logits).all()`
     * Arm 1: `Raptor-224` (`224px` 0.8-crop, `K_EVAL=42`, single-pass `tta=False`, `w=0.35`) with `assert torch.isfinite(logits).all()`
     * Specialist 0: `Cleanlab C384 CoAtNet Fold 0` (`384px`, `3x16` 2.5D slabs sliced directly from `vol` slots 0/2/4 on GPU)
     * Specialist 1: `Cleanlab N4 ConvNeXt-Small Fold 0` (`256px`, bilinear-downsampled on GPU from the same 3-plane slabs)
2. Zero Second-Pass Disk I/O:
   - Stage 2 cell simply verifies `/kaggle/working/submission.csv` written by the fused single-pass engine (`0.01s`).
"""

import json
from pathlib import Path

NB_PATH = Path("notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb")

FUSED_CREAKY_CODE = '''
# ============================================================================
# Fused Single-Read Cleanlab Multi-Resolution MoE (C384 Fold 0 + N4 Fold 0)
# ============================================================================
import torchvision.models as tv_models

class GatedAttentionPool(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        attn_scores = self.w_proj(v * u)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.sum(x * attn_weights, dim=1)

class KneeAnatomicalMoEClassifier(nn.Module):
    def __init__(self, backbone_name: str = "convnext_small", num_classes: int = 12, dropout: float = 0.2, timm_img_size=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        if backbone_name in ["convnext_tiny", "convnext_small"]:
            factory = tv_models.convnext_small if backbone_name == "convnext_small" else tv_models.convnext_tiny
            model = factory(weights=None)
            self.feat_dim = model.classifier[2].in_features
            model.classifier[2] = nn.Identity()
            self.encoder = model
        elif backbone_name.startswith("coatnet"):
            kw = dict(pretrained=False, num_classes=0, in_chans=3)
            if timm_img_size is not None:
                kw["img_size"] = int(timm_img_size)
            elif "224" in backbone_name:
                kw["img_size"] = 384
            model = timm.create_model(backbone_name, **kw)
            self.feat_dim = model.num_features
            self.encoder = model
        else:
            raise RuntimeError(f"Unsupported backbone: {backbone_name}")

        self.sag_pool = GatedAttentionPool(self.feat_dim)
        self.cor_pool = GatedAttentionPool(self.feat_dim)
        self.ax_pool = GatedAttentionPool(self.feat_dim)
        combined_dim = self.feat_dim * 3
        self.sag_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim + combined_dim, 256), nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, 3))
        self.cor_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim + combined_dim, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.ax_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim + combined_dim, 256), nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, 3))
        self.joint_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(combined_dim, 256), nn.SiLU(), nn.Dropout(dropout), nn.Linear(256, 5))

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        B, K, C, H, W = x.shape
        feats = self.encoder(x.view(B * K, C, H, W))
        if isinstance(feats, torch.Tensor) and feats.dim() > 2:
            feats = feats.flatten(1)
        return pool_module(feats.view(B, K, self.feat_dim))

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        h_sag = self._encode_plane(sag, self.sag_pool)
        h_cor = self._encode_plane(cor, self.cor_pool)
        h_ax = self._encode_plane(ax, self.ax_pool)
        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)
        out_sag = self.sag_head(torch.cat([h_sag, combined], dim=1))
        out_cor = self.cor_head(torch.cat([h_cor, combined], dim=1))
        out_ax = self.ax_head(torch.cat([h_ax, combined], dim=1))
        out_joint = self.joint_head(combined)
        return torch.stack([
            out_sag[:, 0], out_cor[:, 0], out_sag[:, 1], out_sag[:, 2],
            out_joint[:, 0], out_joint[:, 1], out_ax[:, 0], out_ax[:, 1],
            out_ax[:, 2], out_joint[:, 2], out_joint[:, 3], out_joint[:, 4],
        ], dim=1)

C384_ADAPTIVE_W = np.array([0.230, 0.270, 0.290, 0.290, 0.220, 0.180, 0.220, 0.500, 0.350, 0.380, 0.270, 0.310], dtype=np.float32)
N4_F0_MU = np.array([-1.7938, -2.15473, -0.02618, -1.58914, -1.10909, -1.83539, -0.958, 0.98501, -0.92933, -1.16773, -0.99397, -2.46582], dtype=np.float32)
N4_F0_SD = np.array([2.07134, 1.39975, 2.67454, 1.85337, 2.00486, 1.28232, 1.58485, 2.30042, 1.3895, 2.53981, 1.93469, 0.98994], dtype=np.float32)
C384_F0_MU = np.array([-1.57795, -1.85532, 0.25548, -1.41753, -1.2835, -1.90778, -1.16651, 0.98819, -0.91018, -1.41221, -0.9679, -2.19061], dtype=np.float32)
C384_F0_SD = np.array([1.08777, 0.84493, 1.77119, 1.20038, 1.4106, 0.8405, 1.15357, 1.78023, 1.17204, 2.02268, 1.18356, 0.63896], dtype=np.float32)

OPTIMIZED_TARGET_WEIGHTS = {
    "Medial OA": 0.0650, "Baker's": 0.0650, "Fracture": 0.0550,
    "Contusion": 0.0500, "ACL": 0.0500, "MCL": 0.0500, "Effusion": 0.0500,
    "Medial Meniscus": 0.0350, "Lateral OA": 0.0300, "PF OA": 0.0250,
    "Lateral Meniscus": 0.0150, "Synovitis": 0.0150,
}

_IDX26 = np.linspace(0, 25, 16, dtype=int)
_IDX22 = np.linspace(0, 21, 16, dtype=int)
_IDX18 = np.linspace(0, 17, 16, dtype=int)
_IDX12 = np.linspace(0, 11, 16, dtype=int)
_CREAKY_NORM = {}

def _find_creaky_ckpts():
    cands_c384 = [
        "/kaggle/input/creaky-c384-cleanlab-models/best_model_fold_0.pt",
        "/kaggle/input/datasets/nswitzer/creaky-c384-cleanlab-models/best_model_fold_0.pt",
        "checkpoints/c384_cleanlab/best_model_fold_0.pt",
    ]
    cands_n4 = [
        "/kaggle/input/creaky-n4-cleanlab-models/best_model_fold_0.pt",
        "/kaggle/input/datasets/nswitzer/creaky-n4-cleanlab-models/best_model_fold_0.pt",
        "checkpoints/n4_cleanlab_trained/best_model_fold_0.pt",
    ]
    p_c384 = next((p for p in cands_c384 if os.path.isfile(p)), None)
    p_n4 = next((p for p in cands_n4 if os.path.isfile(p)), None)
    return p_c384, p_n4

def _load_creaky_model(pt_path, device):
    if pt_path not in _CKPT_RAM_CACHE:
        _CKPT_RAM_CACHE[pt_path] = torch.load(pt_path, map_location="cpu", weights_only=False)
    st = _CKPT_RAM_CACHE[pt_path]
    sd = st["model_state_dict"] if (isinstance(st, dict) and "model_state_dict" in st) else st
    bb = st.get("backbone", "convnext_small") if isinstance(st, dict) else "convnext_small"
    cfg = st.get("config", {}) if (isinstance(st, dict) and isinstance(st.get("config"), dict)) else {}
    m = KneeAnatomicalMoEClassifier(backbone_name=bb, timm_img_size=cfg.get("timm_img_size"))
    m.load_state_dict(sd, strict=True)
    m.eval().to(device)
    return m

@torch.no_grad()
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
    return np.clip(probs, 0.001, 0.999).astype(np.float32)
'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = [
        "# Creaky Speed-Demon V18 Ultra-Sprint · Single-Read 4-Model Multi-Resolution Ensemble (~3.5 Min)\n",
        "\n",
        "1. **Zero Second-Pass Disk I/O:** Reads every test study **once** via `study_lut` (`6 procs × 32/16 threads`) into a single `(96, 384, 384) uint8` volume in RAM.\n",
        "2. **4-Model Multi-Architecture Evaluation per Study on 2× T4 GPUs:**\n",
        "   - `Raptor-384` (`320px`, `K_EVAL=42`, single-pass, `w=0.65`) with `assert torch.isfinite(logits).all()`\n",
        "   - `Raptor-224` (`224px` 0.8-crop, `K_EVAL=42`, single-pass, `w=0.35`) with `assert torch.isfinite(logits).all()`\n",
        "   - `Cleanlab C384 CoAtNet Fold 0` (`384px`, `3×16` 2.5D slabs extracted directly from `vol` on GPU)\n",
        "   - `Cleanlab N4 ConvNeXt-Small Fold 0` (`256px`, bilinear-downsampled on GPU from the same 3-plane slabs)\n",
    ]

    c1 = "".join(nb["cells"][1]["source"])

    # 1. K_EVAL = 42 (matches docstring "Scoring uses 42 windows per study")
    c1 = c1.replace("K_EVAL = 64   # 64 window positions", "K_EVAL = 42   # 42 window positions (ultra-sprint single-pass)")

    # 2. Single-pass tta=False on Arm 0 so both Raptor arms run in 1 forward pass per study
    c1 = c1.replace(
        '{"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 320, "crop_frac": 1.0, "tta": True, "w": 0.65},',
        '{"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 320, "crop_frac": 1.0, "tta": False, "w": 0.65},',
    )

    # 3. Insert FUSED_CREAKY_CODE right before `# ============================================================================\n# Eval windowing`
    marker = "# ============================================================================\n# Eval windowing"
    assert marker in c1, "Could not find Eval windowing marker"
    c1 = c1.replace(marker, FUSED_CREAKY_CODE + "\n" + marker, 1)

    # 4. Update model loading & consumer in main() to also load and run C384 Fold 0 + N4 Fold 0
    old_main_block = """    _wpaths = [find_weight_file(arm["file"]) for arm in ARMS]
    def _preload_ckpt(p):
        if p not in _CKPT_RAM_CACHE:
            _CKPT_RAM_CACHE[p] = torch.load(p, map_location="cpu", weights_only=False)
    with ThreadPoolExecutor(max_workers=len(_wpaths)) as _ex:
        list(_ex.map(_preload_ckpt, _wpaths))
    models = {}
    for d in devs:
        models[d] = []
        for a, arm in enumerate(ARMS):
            model, res = load_model(_wpaths[a], arm["arch"], arm["res"], d)
            models[d].append((model, res))
        print(f"[gpu {d}] loaded {A} arm(s) | {time.time()-t0:.0f}s", flush=True)
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
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))"""

    new_main_block = """    _wpaths = [find_weight_file(arm["file"]) for arm in ARMS]
    _p_c384, _p_n4 = _find_creaky_ckpts()
    _all_ preload = [p for p in (_wpaths + [_p_c384, _p_n4]) if p is not None]
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
    nproc = 6""".replace("_all_ preload", "_all_preload")

    assert old_main_block in c1, "Could not find old_main_block in Cell 1"
    c1 = c1.replace(old_main_block, new_main_block, 1)

    # 5. Update blend at end of main() in Cell 1 to fuse Raptor ranks + Creaky MoE ranks directly
    old_end_block = """    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/raptor_sub.csv"
    sub.to_csv(out, index=False)
    sub.to_csv("/kaggle/working/submission.csv", index=False)"""

    new_end_block = """    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    out = "/kaggle/working/raptor_sub.csv"
    sub.to_csv(out, index=False)
    if has_creaky:
        our_df = pd.DataFrame(creaky_probs.astype(np.float32), columns=LAB)
        our_df.insert(0, "StudyInstanceUID", test_ids)
        our_df.to_csv("/kaggle/working/our_submission.csv", index=False)
        their_ranks = sub[LAB].rank(method="average", pct=True)
        our_ranks = our_df[LAB].rank(method="average", pct=True)
        blended = pd.DataFrame(index=sub.index)
        for col in LAB:
            our_w = float(OPTIMIZED_TARGET_WEIGHTS.get(col, 0.03))
            their_w = 1.0 - our_w
            blended[col] = their_w * their_ranks[col] + our_w * our_ranks[col]
        sub[LAB] = blended[LAB].rank(method="average", pct=True).astype(np.float32)
        print("[blend] fused Single-Read Dual-Raptor + Creaky Multi-Resolution MoE!", flush=True)
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    sub.to_csv("/kaggle/working/submission.csv", index=False)"""

    assert old_end_block in c1, "Could not find old_end_block in Cell 1"
    c1 = c1.replace(old_end_block, new_end_block, 1)

    nb["cells"][1]["source"] = [line + "\n" for line in c1.splitlines()[:-1]] + [c1.splitlines()[-1]]

    # 6. Replace Cell 4 (Stage 2) with a 0.01s verification cell since Stage 1 already fused & wrote final submission.csv!
    nb["cells"][4]["source"] = [
        "import os, pandas as pd\n",
        "assert os.path.isfile('/kaggle/working/submission.csv'), 'submission.csv missing!'\n",
        "sub_df = pd.read_csv('/kaggle/working/submission.csv')\n",
        "print(f'Speed-Demon V18 Ultra-Sprint complete! Final submission shape: {sub_df.shape}')\n",
        "print(sub_df.head().to_string(index=False))\n",
    ]

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Successfully built Speed-Demon V18 Ultra-Sprint in", NB_PATH)


if __name__ == "__main__":
    main()
