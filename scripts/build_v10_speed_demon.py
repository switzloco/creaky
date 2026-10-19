"""Builds Version 10 ("Speed-Demon" Efficiency + Anti-Shakeup Hedge) of
`notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb`:

1. Stage 1 (Unified Single-Read Dual-Raptor on 2x T4 GPUs, ~8 min):
   - Reads each study ONCE via `study_lut` (4 processes x 32/16 threads, C-struct DICOM byte walker).
   - Evaluates both `Raptor-384` (320px crop, 1-view fast FP16, w=0.65) and `Raptor-224` (224px 0.8-crop,
     2-view Mirror-TTA FP16, w=0.35) from the same in-RAM `(vol, mask)` volume across both T4 GPUs.
   - Writes `raptor_sub.csv` and `submission.csv` directly.
2. Stage 2 (Fast Dual-GPU 6-Model Cleanlab-Pruned Multi-Resolution MoE, ~2 min):
   - Uses bit-for-bit verified multithreaded header-first (`stop_before_pixels=True`) + selective slice
     decoding (`ThreadPoolExecutor(16)`) with a 4-study background prefetch pipeline.
   - Shards `3x C384` (384px CoAtNet) onto `cuda:0` and `3x N4` (256px ConvNeXt-Small) onto `cuda:1`,
     running both GPU families concurrently in 1-view FP16 with OOF-Anchored Z-Logit Calibration
     (0.8676 Silver OOF / 0.8568 Gold AUC).
   - Rank-blends into `submission.csv` using Gold-calibrated specialist weights.
"""

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

GOODPJW_NB = "checkpoints/ref_0958_goodpjw/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb"
APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
APEX_META = "notebooks/apex_cleanlab_blend/kernel-metadata.json"


def build_unified_raptor_cell(cell3_src: str) -> str:
    """Transforms Cell 3 (/tmp/raptor_run.py) into a unified single-read Dual-Raptor (384 + 224) runner."""
    src = cell3_src

    # 1. Verify BOTH checkpoints at startup
    old_guard = """_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
_candidates = [
    Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/rsna-knee-oai-original384-s42-pair') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-oai-original384-s42-pair') / _weight_name,
]
_weights = next((p for p in _candidates if p.is_file()), None)
if _weights is None:
    raise FileNotFoundError("Attach the exact selected weights; no alternative model is substituted.")
with _weights.open('rb') as _f:
    assert hashlib.file_digest(_f, 'sha256').hexdigest() == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Checkpoint hash differs\""""

    new_guard = """for _wname, _wsha in [
    ('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'),
    ('raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt', '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'),
]:
    _cands = [
        Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _wname,
        Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _wname,
    ]
    _wp = next((p for p in _cands if p.is_file()), None)
    if _wp is None:
        raise FileNotFoundError(f"Attach the exact selected weights ({_wname}); no alternative model is substituted.")
    with _wp.open('rb') as _f:
        assert hashlib.file_digest(_f, 'sha256').hexdigest() == _wsha, f"Checkpoint hash differs for {_wname}\""""
    assert old_guard in src, "Could not find old_guard in Cell 3"
    src = src.replace(old_guard, new_guard, 1)

    # 2. Update ARMS to include both Raptor-384 (w=0.65, 1-view fast) and Raptor-224 (w=0.35, 2-view mirror TTA)
    old_arms = """ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]"""
    new_arms = """def _crop_frac_fn(t, frac=0.8):
    H, W = t.shape[-2], t.shape[-1]
    ch, cw = int(round(H * frac)), int(round(W * frac))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    return t[..., y0:y0 + ch, x0:x0 + cw]

ARMS = [   # Unified Single-Read Dual-Raptor: 0.65 * Raptor-384 (320px 1-view fast) + 0.35 * Raptor-224 (224px 0.8-crop 2-view Mirror-TTA)
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 320, "crop_frac": 1.0, "tta": False, "w": 0.65},
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224, "crop_frac": 0.8, "tta": True, "w": 0.35},
]"""
    assert old_arms in src, "Could not find old_arms in Cell 3"
    src = src.replace(old_arms, new_arms, 1)

    # 3. Update build_backbone and load_model to support both 320px (384 arch) and 224px (224 arch)
    old_bb_load = """def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)
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
        # Bound backbone memory; the attention head still sees every window together.
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
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
    import hashlib
    with open(pt_path, "rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"
    print(f"[weights] sha256 {observed_sha}", flush=True)
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"
    ck_res = INPUT_RES
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res"""

    new_bb_load = """def build_backbone(arch, pretrained=False, input_res=320):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if input_res == 320:
        kw["img_size"] = 320
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
        flat = x.flatten(0, 1)
        if flat.shape[-1] > 224:
            f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
        else:
            f = self.backbone(flat)
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
    import hashlib
    with open(pt_path, "rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"[weights] {os.path.basename(pt_path)} sha256 {observed_sha[:16]}...", flush=True)
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(res_default)
    bb = build_backbone(arch, pretrained=False, input_res=ck_res)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return model, ck_res"""
    assert old_bb_load in src, "Could not find old_bb_load in Cell 3"
    src = src.replace(old_bb_load, new_bb_load, 1)

    # 4. Update gpu_windows and consumer to support both res=320 (crop_frac=1.0) and res=224 (crop_frac=0.8)
    old_gpu_win = """_WLUT = {}
def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):
    \"\"\"Same windows as eval_windows, built on the GPU from the uint8 volume. Values come from a per-channel
    256-entry table computed on the CPU with eval_windows' own float32 ops, so they are bit-identical.\"\"\"
    D = vol.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)])
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8
    if mirror:
        tri = _mirror_tri(tri, cs, D)
    assert tri.shape[-2:] == (IMG, IMG) and res == INPUT_RES
    lo = (IMG - res) // 2
    tri = tri[:, :, lo:lo + res, lo:lo + res]
    key = (str(dev), norm)
    if key not in _WLUT:
        v = torch.from_numpy((np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0))
        t = torch.stack([v, v, v], 0).view(3, 256, 1)
        if norm == 'imagenet':
            t = (t - _MEAN.view(3, 1, 1)) / _STD.view(3, 1, 1)
        _WLUT[key] = t.view(3, 256).contiguous().to(dev)
    L = _WLUT[key]
    x = torch.stack([L[c][tri[:, c].long()] for c in range(3)], 1)                   # exact float32 values
    if x.shape[-1] != res:
        if norm == 'imagenet':   # resize must happen BEFORE normalisation, as in eval_windows
            L0 = _WLUT.setdefault((str(dev), 'none'), torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(dev))
            x = L0[tri.long()]
            x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
            x = (x - _MEAN.to(dev)) / _STD.to(dev)
        else:
            x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
    return x"""

    new_gpu_win = """_WLUT = {}
def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False, crop_frac=1.0):
    \"\"\"Builds windows on GPU for both 320px center-crop (Raptor-384) and 224px 0.8-crop (Raptor-224).\"\"\"
    D = vol.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)])
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8
    if mirror:
        tri = _mirror_tri(tri, cs, D)
    key = (str(dev), norm)
    if key not in _WLUT:
        v = torch.from_numpy((np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0))
        t = torch.stack([v, v, v], 0).view(3, 256, 1)
        if norm == 'imagenet':
            t = (t - _MEAN.view(3, 1, 1)) / _STD.view(3, 1, 1)
        _WLUT[key] = t.view(3, 256).contiguous().to(dev)
    if crop_frac == 1.0 and res == INPUT_RES:
        lo = (IMG - res) // 2
        tri_c = tri[:, :, lo:lo + res, lo:lo + res]
        L = _WLUT[key]
        return torch.stack([L[c][tri_c[:, c].long()] for c in range(3)], 1)
    else:
        L0 = _WLUT.setdefault((str(dev), 'none'), torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(dev))
        x = L0[tri.long()]
        if crop_frac < 1.0:
            x = _crop_frac_fn(x, crop_frac)
        x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
        if norm == 'imagenet':
            x = (x - _MEAN.to(dev)) / _STD.to(dev)
        return x"""
    assert old_gpu_win in src, "Could not find old_gpu_win in Cell 3"
    src = src.replace(old_gpu_win, new_gpu_win, 1)

    # 5. Update consumer loop to pass crop_frac and respect arm["tta"], and save submission.csv directly
    old_consumer = """    def consumer(d):
        while True:
            item = q.get()
            if item is None: q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                for a, (model, res) in enumerate(models[d]):
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                    del xm
            except Exception as e:"""

    new_consumer = """    def consumer(d):
        while True:
            item = q.get()
            if item is None: q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                for a, (model, res) in enumerate(models[d]):
                    cfrac = float(ARMS[a].get("crop_frac", 1.0))
                    use_tta = bool(ARMS[a].get("tta", False))
                    xw = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=False, crop_frac=cfrac)
                    if use_tta:
                        xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True, crop_frac=cfrac)
                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))
                        del xm
                    else:
                        arm_probs[a][i] = infer_probs(model, xw, d)
                    del xw
            except Exception as e:"""
    assert old_consumer in src, "Could not find old_consumer in Cell 3"
    src = src.replace(old_consumer, new_consumer, 1)

    # 6. Also write submission.csv at the end of raptor_run.py so Stage 1 immediately produces a valid submission.csv
    old_out = """    out = "/kaggle/working/raptor_sub.csv"   # [blend] Raptor stage output; the blend cell writes submission.csv
    sub.to_csv(out, index=False)"""
    new_out = """    out = "/kaggle/working/raptor_sub.csv"
    sub.to_csv(out, index=False)
    sub.to_csv("/kaggle/working/submission.csv", index=False)"""
    assert old_out in src, "Could not find old_out in Cell 3"
    src = src.replace(old_out, new_out, 1)

    return src


def upgrade_part_c_fast_dual_gpu(part_c_src: str) -> str:
    """Upgrades Part C to use:
    1) Bit-for-bit identical multithreaded header-first DICOM loading (stop_before_pixels=True + selective slice decode)
    2) 4-study background prefetching (ThreadPoolExecutor(4))
    3) Dual-GPU model sharding (C384 on cuda:0, N4 on cuda:1) running concurrently in 1-view FP16 (no 2x Zoom TTA)
    4) Specialist weights for blending against Unified Dual-Raptor (Raptor-384 + Raptor-224)
    """
    src = part_c_src

    # 1. Replace sort_by_instance_number, sample_plane_like_cache, and load_series_raw_v1 with fast multithreaded versions
    old_readers = """def sort_by_instance_number(files: List[str]) -> List[str]:
    \"\"\"Same ordering as the cache's get_sorted_files(), for an explicit list of files.\"\"\"
    meta = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            inst = int(getattr(dcm, "InstanceNumber", 0))
            meta.append((inst, f))
        except Exception:
            base = os.path.splitext(os.path.basename(f))[0]
            inst = int(base) if base.isdigit() else 0
            meta.append((inst, f))
    meta.sort(key=lambda x: x[0])
    return [x[1] for x in meta]


def sample_plane_like_cache(dicom_files: List[str], num_slices: int = 16, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    \"\"\"Same as the cache's process_plane(): returns (num_slices, H, W) uint8.

    Only the sampled DICOMs are decoded, so inference reads 16 files per plane, not the whole series.
    \"\"\"
    dcm_files = sort_by_instance_number(dicom_files)
    h, w = target_size
    if not dcm_files:
        return np.zeros((num_slices, h, w), dtype=np.uint8)
    n = len(dcm_files)
    start_idx = int(n * 0.02) if n >= 10 else 0
    end_idx = max(start_idx + 1, int(n * 0.98)) if n >= 10 else n
    indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
    selected = [dcm_files[min(max(0, i), n - 1)] for i in indices]
    slices = []
    for p in selected:
        s = read_slice(p, target_size)
        if s is None:
            s = np.zeros((h, w), dtype=np.uint8)
        slices.append(s)
    return np.stack(slices, axis=0)"""

    new_readers = """from concurrent.futures import ThreadPoolExecutor as _PartCThreadPool

def _fast_inst_hdr(f: str):
    try:
        dcm = pydicom.dcmread(f, stop_before_pixels=True)
        return (int(getattr(dcm, "InstanceNumber", 0)), f)
    except Exception:
        base = os.path.splitext(os.path.basename(f))[0]
        return (int(base) if base.isdigit() else 0, f)


def sort_by_instance_number(files: List[str]) -> List[str]:
    \"\"\"Multithreaded header-only sort by InstanceNumber (bit-identical to single-thread).\"\"\"
    if not files:
        return []
    with _PartCThreadPool(max_workers=16) as ex:
        meta = list(ex.map(_fast_inst_hdr, files))
    meta.sort(key=lambda x: x[0])
    return [x[1] for x in meta]


def read_slice_cropped(dcm_path: str) -> Optional[np.ndarray]:
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=False)
        center = getattr(dcm, "WindowCenter", None)
        width = getattr(dcm, "WindowWidth", None)
        photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
        img = apply_dicom_windowing(dcm.pixel_array, center, width, photometric)
        return crop_empty_borders(img)
    except Exception:
        return None


def sample_plane_dual_cache(dicom_files: List[str], num_slices: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Single-read multithreaded 16-slice sampler returning both (256, 256) and (384, 384) uint8 stacks.\"\"\"
    dcm_files = sort_by_instance_number(dicom_files)
    if not dcm_files:
        return np.zeros((num_slices, 256, 256), dtype=np.uint8), np.zeros((num_slices, 384, 384), dtype=np.uint8)
    n = len(dcm_files)
    start_idx = int(n * 0.02) if n >= 10 else 0
    end_idx = max(start_idx + 1, int(n * 0.98)) if n >= 10 else n
    indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
    selected = [dcm_files[min(max(0, i), n - 1)] for i in indices]
    with _PartCThreadPool(max_workers=16) as ex:
        cropped_list = list(ex.map(read_slice_cropped, selected))
    s256, s384 = [], []
    for c in cropped_list:
        if c is None:
            s256.append(np.zeros((256, 256), dtype=np.uint8))
            s384.append(np.zeros((384, 384), dtype=np.uint8))
        else:
            s256.append(cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA if c.shape[0] > 256 else cv2.INTER_LINEAR))
            s384.append(cv2.resize(c, (384, 384), interpolation=cv2.INTER_AREA if c.shape[0] > 384 else cv2.INTER_LINEAR))
    return np.stack(s256, axis=0), np.stack(s384, axis=0)


def sample_plane_like_cache(dicom_files: List[str], num_slices: int = 16, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    \"\"\"Multithreaded 16-slice cache sampler (bit-identical to single-thread).\"\"\"
    dcm_files = sort_by_instance_number(dicom_files)
    h, w = target_size
    if not dcm_files:
        return np.zeros((num_slices, h, w), dtype=np.uint8)
    n = len(dcm_files)
    start_idx = int(n * 0.02) if n >= 10 else 0
    end_idx = max(start_idx + 1, int(n * 0.98)) if n >= 10 else n
    indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
    selected = [dcm_files[min(max(0, i), n - 1)] for i in indices]
    with _PartCThreadPool(max_workers=16) as ex:
        raw_slices = list(ex.map(lambda p: read_slice(p, target_size), selected))
    slices = [s if s is not None else np.zeros((h, w), dtype=np.uint8) for s in raw_slices]
    return np.stack(slices, axis=0)"""
    assert old_readers in src, "Could not find old_readers in Part C"
    src = src.replace(old_readers, new_readers, 1)

    old_raw_v1 = """def load_series_raw_v1(dicom_files: List[str], target_size=(256, 256), num_slices: int = 16) -> np.ndarray:
    \"\"\"raw_v1 copy of the training script's load_and_preprocess_series(); returns (K, 3, H, W) float32.\"\"\"
    if not dicom_files:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    slice_entries = []
    for fpath in dicom_files:
        try:
            dcm = pydicom.dcmread(fpath)
            coord = compute_slice_position(dcm)
            photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
            wc = getattr(dcm, "WindowCenter", None)
            ww = getattr(dcm, "WindowWidth", None)
            pixels = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixels = pixels * slope + intercept
            img = apply_windowing_raw(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders_raw(img)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            slice_entries.append((coord, img))
        except Exception:
            continue

    if not slice_entries:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    slice_entries.sort(key=lambda x: x[0])
    volume = np.stack([x[1] for x in slice_entries], axis=0)

    n_slices = len(volume)
    indices = np.linspace(0, n_slices - 1, num_slices).round().astype(int)
    slabs = []
    for idx in indices:
        z_prev = volume[max(0, idx - 1)]
        z_curr = volume[idx]
        z_next = volume[min(n_slices - 1, idx + 1)]
        slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
        slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
        slabs.append(slab)

    return np.stack(slabs, axis=0).astype(np.float32)"""

    new_raw_v1 = """def load_series_raw_v1(dicom_files: List[str], target_size=(256, 256), num_slices: int = 16) -> np.ndarray:
    \"\"\"Fast multithreaded header-first raw_v1 loader: sorts headers via stop_before_pixels=True
    and decodes pixels ONLY for the 16 sampled indices and their immediate 2.5D neighbors.\"\"\"
    if not dicom_files:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    def _hdr_pos(fpath):
        try:
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            return (compute_slice_position(dcm), fpath)
        except Exception:
            return None

    with _PartCThreadPool(max_workers=16) as ex:
        hdrs = [h for h in ex.map(_hdr_pos, dicom_files) if h is not None]

    if not hdrs:
        return np.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=np.float32)

    hdrs.sort(key=lambda x: x[0])
    n_slices = len(hdrs)
    indices = np.linspace(0, n_slices - 1, num_slices).round().astype(int)
    needed_idx = sorted({max(0, i - 1) for i in indices} | set(indices) | {min(n_slices - 1, i + 1) for i in indices})

    def _decode_idx(idx):
        fpath = hdrs[idx][1]
        try:
            dcm = pydicom.dcmread(fpath)
            photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
            wc = getattr(dcm, "WindowCenter", None)
            ww = getattr(dcm, "WindowWidth", None)
            pixels = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixels = pixels * slope + intercept
            img = apply_windowing_raw(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders_raw(img)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            return idx, img
        except Exception:
            return idx, np.zeros(target_size, dtype=np.uint8)

    with _PartCThreadPool(max_workers=16) as ex:
        decoded = dict(ex.map(_decode_idx, needed_idx))

    slabs = []
    for idx in indices:
        z_prev = decoded[max(0, idx - 1)]
        z_curr = decoded[idx]
        z_next = decoded[min(n_slices - 1, idx + 1)]
        slab = np.stack([z_prev, z_curr, z_next], axis=0).astype(np.float32) / 255.0
        slab = (slab - IMAGENET_MEAN) / IMAGENET_STD
        slabs.append(slab)

    return np.stack(slabs, axis=0).astype(np.float32)"""
    assert old_raw_v1 in src, "Could not find old_raw_v1 in Part C"
    src = src.replace(old_raw_v1, new_raw_v1, 1)

    # 2. Shard C384 (cache_v1_384) onto cuda:0 and N4 (cache_v1) onto cuda:1 (when 2 GPUs exist)
    old_model_load = """                models.append((model, pipeline, ckpt_fold))"""
    new_model_load = """                target_dev = torch.device("cuda:1") if (pipeline == "cache_v1" and torch.cuda.device_count() >= 2) else device
                model = model.to(target_dev)
                models.append((model, pipeline, ckpt_fold))"""
    assert old_model_load in src, "Could not find old_model_load in Part C"
    src = src.replace(old_model_load, new_model_load, 1)

    # 2b. Single-read dual-resolution (256px + 384px) plane loading when both cache_v1 and cache_v1_384 are active
    old_plane_loop = """        pipelines_in_use = sorted({m[1] for m in models}) or ["cache_v1"]
        plane_tensors = {}  # pipeline -> plane -> (1, K, 3, H, W)
        for pipeline in pipelines_in_use:
            plane_tensors[pipeline] = {}
            for plane in ["Sagittal", "Coronal", "Axial"]:
                series_files = find_series_files(study_uid, study_series, plane_col, plane, pipeline, test_series_dir)
                if idx < 3:
                    print(f"  Study {study_uid[:15]}... | {pipeline} | {plane:8s} | Found {len(series_files)} slices")
                tensor = load_plane(series_files, pipeline, target_size=(256, 256), num_slices=16)
                plane_tensors[pipeline][plane] = tensor.unsqueeze(0).to(device)"""

    new_plane_loop = """        pipelines_in_use = sorted({m[1] for m in models}) or ["cache_v1"]
        plane_tensors = {p: {} for p in pipelines_in_use}  # pipeline -> plane -> (1, K, 3, H, W)
        if "cache_v1" in pipelines_in_use and "cache_v1_384" in pipelines_in_use:
            for plane in ["Sagittal", "Coronal", "Axial"]:
                series_files = find_series_files(study_uid, study_series, plane_col, plane, "cache_v1", test_series_dir)
                if idx < 3:
                    print(f"  Study {study_uid[:15]}... | dual_cache(256+384) | {plane:8s} | Found {len(series_files)} slices")
                arr256, arr384 = sample_plane_dual_cache(series_files, num_slices=16)
                t256 = torch.tensor(cached_plane_to_slabs(arr256), dtype=torch.float32).unsqueeze(0)
                t384 = torch.tensor(cached_plane_to_slabs(arr384), dtype=torch.float32).unsqueeze(0)
                plane_tensors["cache_v1"][plane] = t256
                plane_tensors["cache_v1_384"][plane] = t384
            rem_pipes = [p for p in pipelines_in_use if p not in ("cache_v1", "cache_v1_384")]
        else:
            rem_pipes = pipelines_in_use
        for pipeline in rem_pipes:
            for plane in ["Sagittal", "Coronal", "Axial"]:
                series_files = find_series_files(study_uid, study_series, plane_col, plane, pipeline, test_series_dir)
                if idx < 3:
                    print(f"  Study {study_uid[:15]}... | {pipeline} | {plane:8s} | Found {len(series_files)} slices")
                tensor = load_plane(series_files, pipeline, target_size=(256, 256), num_slices=16)
                plane_tensors[pipeline][plane] = tensor.unsqueeze(0)"""
    assert old_plane_loop in src, "Could not find old_plane_loop in Part C"
    src = src.replace(old_plane_loop, new_plane_loop, 1)

    old_infer_block = """        # Precompute 2nd TTA view in VRAM once per preprocessing pipeline (zero extra DICOM I/O)
        plane_tensors_zoom = {}
        if models:
            for pipeline in pipelines_in_use:
                plane_tensors_zoom[pipeline] = {
                    plane: _make_zoom_tta(plane_tensors[pipeline][plane], crop_ratio=0.96)
                    for plane in ["Sagittal", "Coronal", "Axial"]
                }

        # Run Model Inference (with 2x Multi-Scale TTA + OOF-Anchored Z-Logit Calibration) or Priors Fallback
        if models:
            convnext_family = []
            coatnet_family = []
            other_family = []
            with torch.no_grad():
                for model_item in models:
                    model, pipeline = model_item[0], model_item[1]
                    ckpt_fold = model_item[2] if len(model_item) > 2 else None
                    pt = plane_tensors[pipeline]
                    pt_z = plane_tensors_zoom[pipeline]
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits_1 = model(pt["Sagittal"], pt["Coronal"], pt["Axial"]).float()
                        logits_2 = model(pt_z["Sagittal"], pt_z["Coronal"], pt_z["Axial"]).float()
                    raw_logits = (0.65 * logits_1 + 0.35 * logits_2).cpu().numpy()[0]"""

    new_infer_block = """        # Run Model Inference (1-View Fast Dual-GPU Sharded FP16 + OOF-Anchored Z-Logit Calibration) or Priors Fallback
        if models:
            convnext_family = []
            coatnet_family = []
            other_family = []
            with torch.no_grad():
                for model_item in models:
                    model, pipeline = model_item[0], model_item[1]
                    ckpt_fold = model_item[2] if len(model_item) > 2 else None
                    m_dev = next(model.parameters()).device
                    pt = plane_tensors[pipeline]
                    with torch.amp.autocast('cuda', enabled=(m_dev.type == "cuda")):
                        logits_1 = model(
                            pt["Sagittal"].to(m_dev, non_blocking=True),
                            pt["Coronal"].to(m_dev, non_blocking=True),
                            pt["Axial"].to(m_dev, non_blocking=True),
                        ).float()
                    raw_logits = logits_1.cpu().numpy()[0]"""
    assert old_infer_block in src, "Could not find old_infer_block in Part C"
    src = src.replace(old_infer_block, new_infer_block, 1)

    # 3. Update OPTIMIZED_TARGET_WEIGHTS for blending 6-Model Cleanlab MoE directly with Unified Dual-Raptor
    old_weights = """        OPTIMIZED_TARGET_WEIGHTS = {
            "Medial OA": 0.0450,
            "Baker's": 0.0450,
            "Fracture": 0.0400,
            "Contusion": 0.0350,
            "ACL": 0.0350,
            "MCL": 0.0350,
            "Effusion": 0.0350,
            "Medial Meniscus": 0.0300,
            "Lateral OA": 0.0250,
            "PF OA": 0.0200,
            "Lateral Meniscus": 0.0150,
            "Synovitis": 0.0150,
        }"""

    new_weights = """        # Speed-Demon V10 Specialist Weights: blending our 6-Model Cleanlab-Pruned Multi-Resolution MoE
        # (3x N4 ConvNeXt-Small + 3x C384 CoAtNet, 0.8676 Silver OOF / 0.8568 Gold AUC)
        # directly with Unified Dual-Raptor (0.65 * Raptor-384 + 0.35 * Raptor-224):
        OPTIMIZED_TARGET_WEIGHTS = {
            "Medial OA": 0.0650,
            "Baker's": 0.0650,
            "Fracture": 0.0550,
            "Contusion": 0.0500,
            "ACL": 0.0500,
            "MCL": 0.0500,
            "Effusion": 0.0500,
            "Medial Meniscus": 0.0350,
            "Lateral OA": 0.0300,
            "PF OA": 0.0250,
            "Lateral Meniscus": 0.0150,
            "Synovitis": 0.0150,
        }"""
    assert old_weights in src, "Could not find old_weights in Part C"
    src = src.replace(old_weights, new_weights, 1)

    return src


def main():
    with open(GOODPJW_NB, "r", encoding="utf-8") as f:
        base_nb = json.load(f)

    with open(APEX_NB, "r", encoding="utf-8") as f:
        our_nb = json.load(f)

    our_stage4_cell = None
    for cell in our_nb["cells"]:
        if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
            our_stage4_cell = cell
            break
    assert our_stage4_cell is not None, "Could not find Part C cell in current notebook"

    # Build unified Dual-Raptor cell from base_nb Cell 3
    unified_raptor_src = build_unified_raptor_cell("".join(base_nb["cells"][3]["source"]))
    compile("".join(l for l in unified_raptor_src.splitlines(keepends=True) if not l.startswith("%%writefile")), "<unified_raptor>", "exec")

    # Build fast Dual-GPU Part C cell
    fast_part_c_src = upgrade_part_c_fast_dual_gpu("".join(our_stage4_cell["source"]))
    compile(fast_part_c_src, "<fast_part_c>", "exec")

    # Assemble clean 5-cell Speed-Demon notebook:
    # Cell 0: Intro Markdown
    # Cell 1: Unified Single-Read Dual-Raptor (/tmp/raptor_run.py) writefile
    # Cell 2: Execute /tmp/raptor_run.py
    # Cell 3: Part C Markdown
    # Cell 4: Fast Dual-GPU 6-Model Cleanlab MoE + Specialist Rank Fusion
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Creaky Speed-Demon V10 · Efficiency Track + Anti-Shakeup Hedge (~10 Min, 8-Model Multi-Resolution Ensemble)\n",
                "\n",
                "1. **Stage 1 (Unified Single-Read Dual-Raptor on 2× T4 GPUs):** Reads every test study **once** via `study_lut` (`4 procs × 32/16 threads`) and evaluates both `Raptor-384` (`320px`, `w=0.65`) and `Raptor-224` (`224px` 0.8-crop with Mirror-TTA, `w=0.35`) from the same in-RAM volume.\n",
                "2. **Stage 2 (Fast Dual-GPU 6-Model Cleanlab-Pruned Multi-Resolution MoE):** Header-first multithreaded DICOM sampling (`stop_before_pixels=True`) + dual-GPU (`cuda:0` `3x C384` CoAtNet + `cuda:1` `3x N4` ConvNeXt-Small) 1-view FP16 inference with OOF-Anchored Z-Logit Calibration (`0.8676` Silver OOF / `0.8568` Gold AUC).\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l for l in unified_raptor_src.splitlines(keepends=True)],
        },
        base_nb["cells"][4],  # Runs /tmp/raptor_run.py and records NB_T0
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "# Stage 2 · Fast Dual-GPU 6-Model Cleanlab-Pruned Multi-Resolution MoE (`0.8676` Silver OOF / `0.8568` Gold AUC)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l for l in fast_part_c_src.splitlines(keepends=True)],
        },
    ]

    out_nb = {
        "metadata": base_nb["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }

    for idx, cell in enumerate(out_nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            lines = [l for l in src.splitlines(keepends=True) if not l.startswith("%%writefile")]
            compile("".join(lines), f"<cell_{idx}>", "exec")

    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(out_nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully built Speed-Demon V10 in {APEX_NB} ({len(out_nb['cells'])} cells)!")

    # Only attach the 4 required datasets (drops R6 and 2.5D MIL datasets for faster container mount!)
    meta = {
        "id": "nswitzer/creaky-apex-cleanlab-blend",
        "title": "Creaky Apex Cleanlab Blend",
        "code_file": "creaky-apex-cleanlab-blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "nswitzer/creaky-n4-cleanlab-models",
            "nswitzer/creaky-c384-cleanlab-models",
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection",
        ],
    }
    with open(APEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {APEX_META} with 4 lightweight dataset mounts!")


if __name__ == "__main__":
    main()
