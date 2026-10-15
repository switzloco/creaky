"""Upgrades Stage 4 of creaky-apex-cleanlab-blend.ipynb and submission_notebook.py for E24 (N4 256px ConvNeXt + C384 384px CoAtNet Adaptive MoE)."""

import json
import os

APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
SUB_PY = "scripts/submission/submission_notebook.py"
SUB_NB = "notebooks/submission-notebook/submission-notebook.ipynb"


def patch_stage4_source(src: str, is_apex_nb: bool = True) -> str:
    # 1. Update CHECKPOINT_FILTER in apex_cleanlab_blend
    if is_apex_nb:
        old_filter = 'CHECKPOINT_FILTER: List[str] = ["creaky-n4-cleanlab-models", "best_model_fold_0.pt", "best_model_fold_1.pt", "best_model_fold_2.pt"]'
        new_filter = (
            'CHECKPOINT_FILTER: List[str] = [\n'
            '    "creaky-n4-cleanlab-models",\n'
            '    "creaky-c384-cleanlab-models",\n'
            '    "n4_cleanlab_trained",\n'
            '    "c384_cleanlab",\n'
            ']'
        )
        assert old_filter in src, "Could not find old CHECKPOINT_FILTER"
        src = src.replace(old_filter, new_filter, 1)

    # 2. Update local checkpoint discovery in get_data_paths() to also scan subdirectories n4_cleanlab_trained & c384_cleanlab
    old_local_ckpt = """        local_checkpoints = os.path.abspath("checkpoints")
        if os.path.exists(local_checkpoints):
            for f in os.listdir(local_checkpoints):
                if f.endswith(".pt") or f.endswith(".pth"):
                    ckpt_files.append(os.path.join(local_checkpoints, f))"""

    new_local_ckpt = """        local_checkpoints = os.path.abspath("checkpoints")
        if os.path.exists(local_checkpoints):
            for sub in ["n4_cleanlab_trained", "c384_cleanlab"]:
                sub_dir = os.path.join(local_checkpoints, sub)
                if os.path.isdir(sub_dir):
                    for f in sorted(os.listdir(sub_dir)):
                        if f.endswith(".pt") or f.endswith(".pth"):
                            ckpt_files.append(os.path.join(sub_dir, f))
            if not ckpt_files:
                for f in os.listdir(local_checkpoints):
                    if f.endswith(".pt") or f.endswith(".pth"):
                        ckpt_files.append(os.path.join(local_checkpoints, f))"""

    if old_local_ckpt in src:
        src = src.replace(old_local_ckpt, new_local_ckpt, 1)

    # 3. Add cache_v1_384 to PIPELINES, select_series_uid, load_plane, and checkpoint_pipeline
    src = src.replace('PIPELINES = ("cache_v1", "raw_v1")', 'PIPELINES = ("cache_v1", "cache_v1_384", "raw_v1")')
    src = src.replace('if pipeline == "cache_v1":\n        match = study_series', 'if pipeline in ("cache_v1", "cache_v1_384"):\n        match = study_series')

    old_load_plane = """def load_plane(dicom_files: List[str], pipeline: str, target_size=(256, 256), num_slices: int = 16) -> torch.Tensor:
    \"\"\"DICOM files of one series -> (K, 3, H, W) tensor, preprocessed like the given pipeline.\"\"\"
    if pipeline == "cache_v1":
        slabs = cached_plane_to_slabs(sample_plane_like_cache(dicom_files, num_slices=num_slices, target_size=target_size))
    elif pipeline == "raw_v1":
        slabs = load_series_raw_v1(dicom_files, target_size=target_size, num_slices=num_slices)
    else:
        raise ValueError(f"Unknown preprocessing pipeline: {pipeline}")
    return torch.tensor(slabs, dtype=torch.float32)"""

    new_load_plane = """def load_plane(dicom_files: List[str], pipeline: str, target_size=(256, 256), num_slices: int = 16) -> torch.Tensor:
    \"\"\"DICOM files of one series -> (K, 3, H, W) tensor, preprocessed like the given pipeline.\"\"\"
    if pipeline == "cache_v1":
        slabs = cached_plane_to_slabs(sample_plane_like_cache(dicom_files, num_slices=num_slices, target_size=target_size))
    elif pipeline == "cache_v1_384":
        slabs = cached_plane_to_slabs(sample_plane_like_cache(dicom_files, num_slices=num_slices, target_size=(384, 384)))
    elif pipeline == "raw_v1":
        slabs = load_series_raw_v1(dicom_files, target_size=target_size, num_slices=num_slices)
    else:
        raise ValueError(f"Unknown preprocessing pipeline: {pipeline}")
    return torch.tensor(slabs, dtype=torch.float32)"""

    assert old_load_plane in src, "Could not find old load_plane"
    src = src.replace(old_load_plane, new_load_plane, 1)

    old_ckpt_pipe = """def checkpoint_pipeline(state: dict, backbone: str) -> str:
    \"\"\"Which preprocessing a checkpoint was trained with.

    New checkpoints record it. Older ones don't: the resnet34 run predates the slice cache
    (raw_v1), and the ConvNeXt run was the first trained on the cache (cache_v1).
    \"\"\"
    pipe = state.get("preprocessing") if isinstance(state, dict) else None
    if pipe is None:
        pipe = "raw_v1" if backbone.startswith("resnet") else "cache_v1"
        print(f"    (checkpoint does not record its preprocessing; assuming '{pipe}' for backbone '{backbone}')")
    if pipe not in PIPELINES:
        raise ValueError(f"Checkpoint uses unknown preprocessing '{pipe}'")
    return pipe"""

    new_ckpt_pipe = """def checkpoint_pipeline(state: dict, backbone: str) -> str:
    \"\"\"Which preprocessing a checkpoint was trained with.

    New checkpoints record it. Older ones don't: the resnet34 run predates the slice cache
    (raw_v1), and the ConvNeXt run was the first trained on the cache (cache_v1).
    384px CoAtNet checkpoints trained on the 384px N4 cache map to 'cache_v1_384'.
    \"\"\"
    pipe = state.get("preprocessing") if isinstance(state, dict) else None
    if pipe is None:
        pipe = "raw_v1" if backbone.startswith("resnet") else "cache_v1"
        print(f"    (checkpoint does not record its preprocessing; assuming '{pipe}' for backbone '{backbone}')")
    cfg = state.get("config", {}) if (isinstance(state, dict) and isinstance(state.get("config"), dict)) else {}
    ts = cfg.get("target_size")
    is_384 = (
        (isinstance(ts, (list, tuple)) and len(ts) == 2 and int(ts[0]) == 384)
        or ("384" in backbone)
        or (cfg.get("timm_img_size") == 384)
    )
    if pipe == "cache_v1" and is_384:
        pipe = "cache_v1_384"
    if pipe not in PIPELINES:
        raise ValueError(f"Checkpoint uses unknown preprocessing '{pipe}'")
    return pipe"""

    assert old_ckpt_pipe in src, "Could not find old checkpoint_pipeline"
    src = src.replace(old_ckpt_pipe, new_ckpt_pipe, 1)

    # 4. Add CoAtNet (timm) branch to KneeAnatomicalMoEClassifier.__init__
    old_moe_init = """    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # Vision Backbone (ResNet, ConvNeXt, EfficientNet)
        try:
            if backbone_name == "resnet34":
                model = tv_models.resnet34(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name == "resnet50":
                model = tv_models.resnet50(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name in ["convnext_tiny", "convnext_small"]:
                factory = tv_models.convnext_small if backbone_name == "convnext_small" else tv_models.convnext_tiny
                model = factory(weights=None)
                self.feat_dim = model.classifier[2].in_features
                model.classifier[2] = nn.Identity()
                self.encoder = model
            elif backbone_name in ["efficientnet_v2_s", "effnet"]:
                model = tv_models.efficientnet_v2_s(weights=None)
                self.feat_dim = model.classifier[1].in_features
                model.classifier[1] = nn.Identity()
                self.encoder = model
            else:
                model = getattr(tv_models, backbone_name)(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model"""

    new_moe_init = """    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = 12,
        dropout: float = 0.2,
        timm_img_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # Vision Backbone (ResNet, ConvNeXt, EfficientNet, CoAtNet)
        try:
            if backbone_name == "resnet34":
                model = tv_models.resnet34(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name == "resnet50":
                model = tv_models.resnet50(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model
            elif backbone_name in ["convnext_tiny", "convnext_small"]:
                factory = tv_models.convnext_small if backbone_name == "convnext_small" else tv_models.convnext_tiny
                model = factory(weights=None)
                self.feat_dim = model.classifier[2].in_features
                model.classifier[2] = nn.Identity()
                self.encoder = model
            elif backbone_name in ["efficientnet_v2_s", "effnet"]:
                model = tv_models.efficientnet_v2_s(weights=None)
                self.feat_dim = model.classifier[1].in_features
                model.classifier[1] = nn.Identity()
                self.encoder = model
            elif backbone_name.startswith("coatnet"):
                import timm
                kw = dict(pretrained=False, num_classes=0, in_chans=3)
                if timm_img_size is not None:
                    kw["img_size"] = int(timm_img_size)
                elif "224" in backbone_name:
                    kw["img_size"] = 384
                model = timm.create_model(backbone_name, **kw)
                self.feat_dim = model.num_features
                self.encoder = model
            else:
                model = getattr(tv_models, backbone_name)(weights=None)
                self.feat_dim = model.fc.in_features
                model.fc = nn.Identity()
                self.encoder = model"""

    assert old_moe_init in src, "Could not find old KneeAnatomicalMoEClassifier.__init__"
    src = src.replace(old_moe_init, new_moe_init, 1)

    # 5. Pass timm_img_size when constructing KneeAnatomicalMoEClassifier in run_submission_inference
    old_model_construct = """                is_moe = any("sag_head" in k for k in sd.keys()) or arch == "KneeAnatomicalMoEClassifier"
                if is_moe:
                    model = KneeAnatomicalMoEClassifier(backbone_name=bb, pretrained=False).to(device)"""

    new_model_construct = """                cfg = state.get("config", {}) if (isinstance(state, dict) and isinstance(state.get("config"), dict)) else {}
                timm_img_size = cfg.get("timm_img_size")
                is_moe = any("sag_head" in k for k in sd.keys()) or arch == "KneeAnatomicalMoEClassifier"
                if is_moe:
                    model = KneeAnatomicalMoEClassifier(backbone_name=bb, pretrained=False, timm_img_size=timm_img_size).to(device)"""

    assert old_model_construct in src, "Could not find old model construction block"
    src = src.replace(old_model_construct, new_model_construct, 1)

    # 6. Upgrade model family blending inside run_submission_inference (Silver-OOF-validated C384 + N4 adaptive weights)
    old_infer_loop = """        # Run Model Inference or Priors Fallback
        if models:
            ensemble_probs = []
            weights = []
            with torch.no_grad():
                for model, pipeline in models:
                    pt = plane_tensors[pipeline]
                    logits = model(pt["Sagittal"], pt["Coronal"], pt["Axial"])
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    bb = getattr(model, "backbone_name", "")
                    w = 0.70 if bb in ["convnext_small", "convnext_tiny"] else 0.30
                    ensemble_probs.append(probs * w)
                    weights.append(w)
            avg_probs = np.sum(ensemble_probs, axis=0) / max(1e-6, sum(weights))
            for t_idx, target in enumerate(TARGET_COLS):
                pred_dict[target] = float(np.clip(avg_probs[t_idx], 0.001, 0.999))"""

    new_infer_loop = """        # Silver-OOF-fitted per-target weights for C384 CoAtNet vs N4 ConvNeXt
        # Validated on out-of-sample Gold: lifts 3-fold N4 from 0.8411 -> 0.8549 (+0.0138)
        C384_ADAPTIVE_WEIGHTS = {
            "ACL": 0.325,
            "MCL": 0.150,
            "Medial Meniscus": 0.400,
            "Lateral Meniscus": 0.250,
            "Medial OA": 0.000,
            "Lateral OA": 0.075,
            "PF OA": 0.075,
            "Effusion": 0.400,
            "Synovitis": 0.400,
            "Baker's": 0.425,
            "Contusion": 0.200,
            "Fracture": 0.225,
        }

        # Run Model Inference or Priors Fallback
        if models:
            convnext_family = []
            coatnet_family = []
            other_family = []
            with torch.no_grad():
                for model, pipeline in models:
                    pt = plane_tensors[pipeline]
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        logits = model(pt["Sagittal"], pt["Coronal"], pt["Axial"])
                    probs = torch.sigmoid(logits.float()).cpu().numpy()[0]
                    bb = getattr(model, "backbone_name", "")
                    if bb.startswith("coatnet"):
                        coatnet_family.append(probs)
                    elif bb in ["convnext_small", "convnext_tiny"]:
                        convnext_family.append(probs)
                    else:
                        other_family.append(probs)

            if convnext_family and coatnet_family:
                cn_avg = np.mean(convnext_family, axis=0)
                ca_avg = np.mean(coatnet_family, axis=0)
                avg_probs = np.zeros(len(TARGET_COLS), dtype=np.float32)
                for t_idx, target in enumerate(TARGET_COLS):
                    w_ca = float(C384_ADAPTIVE_WEIGHTS.get(target, 0.25))
                    avg_probs[t_idx] = (1.0 - w_ca) * cn_avg[t_idx] + w_ca * ca_avg[t_idx]
            else:
                all_probs = convnext_family + coatnet_family + other_family
                avg_probs = np.mean(all_probs, axis=0)

            for t_idx, target in enumerate(TARGET_COLS):
                pred_dict[target] = float(np.clip(avg_probs[t_idx], 0.001, 0.999))"""

    assert old_infer_loop in src, "Could not find old inference loop"
    src = src.replace(old_infer_loop, new_infer_loop, 1)

    # 7. In apex_cleanlab_blend, upgrade the final blend block to use continuous 3-way rank fusion anchored on verified 0.950 V1 weights + micro-injected N4+C384 ensemble
    if is_apex_nb:
        old_tail = """import shutil as _sh
_sh.copy('submission.csv', 'public_submission_backup.csv')
try:
    run_submission_inference("our_submission.csv")
    print('Blending Apex Grandmaster V5 (0.952+) with Creaky E18 Cleanlab-Supervised ConvNeXt!')
    import pandas as pd
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        their_sub = pd.read_csv('submission.csv')
        our_sub = pd.read_csv('our_submission.csv')
        their_sub['StudyInstanceUID'] = their_sub['StudyInstanceUID'].astype(str)
        our_sub['StudyInstanceUID'] = our_sub['StudyInstanceUID'].astype(str)
        our_sub = our_sub.set_index('StudyInstanceUID').reindex(their_sub['StudyInstanceUID']).reset_index()
        assert not our_sub.isna().any().any(), 'study ID mismatch between submissions'
        
        # Simple rank blend
        TARGET_COLS = [c for c in their_sub.columns if c != 'StudyInstanceUID']
        
        their_ranks = their_sub[TARGET_COLS].rank(pct=True)
        our_ranks = our_sub[TARGET_COLS].rank(pct=True)
        
                # Conservative Class-Specific Rank Blend Anchored on CoAtNet 0.949 Base
        # Track N4 3-Fold Ensemble (ConvNeXt MoE + Fluid-Sensitive + 2%-98% Wide Slicing)
        OPTIMIZED_TARGET_WEIGHTS = {
            "ACL": 0.1500,
            "MCL": 0.2000,              # Record Fold 1 Gold AUC: 0.9138
            "Medial Meniscus": 0.1500,
            "Lateral Meniscus": 0.2500,  # Solves peripheral clipping (+0.0286 ensemble lift)
            "Medial OA": 0.1500,
            "Lateral OA": 0.1500,
            "PF OA": 0.1500,
            "Effusion": 0.1200,
            "Synovitis": 0.1000,
            "Baker's": 0.1500,
            "Contusion": 0.1500,
            "Fracture": 0.1500,
        }
        blended = pd.DataFrame(index=their_sub.index)
        for col in TARGET_COLS:
            our_w = float(OPTIMIZED_TARGET_WEIGHTS.get(col, 0.15))
            their_w = 1.0 - our_w
            blended[col] = their_w * their_ranks[col] + our_w * our_ranks[col]
            print(f"Target '{col}': weight {our_w:.4f} our model / {their_w:.4f} their model")
        their_sub[TARGET_COLS] = blended
        
        their_sub.to_csv('submission.csv', index=False)
        print('Blend complete! Wrote final submission.csv')
    except Exception as e:
        print('Blend failed! Outputting original submission.csv just to be safe.')
        print(e)
except Exception as _e:
    print('Our ConvNeXt branch failed; restoring untouched public submission:', _e)
    _sh.copy('public_submission_backup.csv', 'submission.csv')"""

        new_tail = """import shutil as _sh
_sh.copy('submission.csv', 'public_submission_backup.csv')
try:
    run_submission_inference("our_submission.csv")
    print('Blending Apex Foundation with Creaky E24 Multi-Resolution MoE (N4 256px ConvNeXt + C384 384px CoAtNet)!')
    import pandas as pd
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        their_sub = pd.read_csv('submission.csv')
        our_sub = pd.read_csv('our_submission.csv')
        their_sub['StudyInstanceUID'] = their_sub['StudyInstanceUID'].astype(str)
        our_sub['StudyInstanceUID'] = our_sub['StudyInstanceUID'].astype(str)
        our_sub = our_sub.set_index('StudyInstanceUID').reindex(their_sub['StudyInstanceUID']).reset_index()
        assert not our_sub.isna().any().any(), 'study ID mismatch between submissions'

        TARGET_COLS = [c for c in their_sub.columns if c != 'StudyInstanceUID']
        our_ranks = our_sub[TARGET_COLS].rank(method="average", pct=True)

        # Micro-injection weights calibrated to Gold AUC of our N4+C384 MoE Ensemble (0.8549 Gold Macro AUC):
        # Top tier (>0.90 Gold AUC): Baker's (0.916), Lateral Meniscus (0.909), MCL (0.903) -> 0.050
        # Strong tier (0.87-0.90 Gold AUC): Medial OA (0.896), Fracture (0.896), ACL (0.869) -> 0.040-0.045
        # Mid tier (0.80-0.86 Gold AUC): Medial Meniscus (0.858), Lateral OA (0.853), Contusion (0.808) -> 0.025-0.035
        # Conservative tier (<0.80 Gold AUC): PF OA (0.796), Synovitis (0.743), Effusion (0.712) -> 0.015
        OPTIMIZED_TARGET_WEIGHTS = {
            "Baker's": 0.0500,
            "Lateral Meniscus": 0.0500,
            "MCL": 0.0500,
            "Medial OA": 0.0450,
            "Fracture": 0.0450,
            "ACL": 0.0400,
            "Medial Meniscus": 0.0350,
            "Lateral OA": 0.0300,
            "Contusion": 0.0250,
            "PF OA": 0.0150,
            "Synovitis": 0.0150,
            "Effusion": 0.0150,
        }

        # Verified 0.950 V1 weights between Stage 1 (_sota_0949.csv) and Stage 2 (_own.csv)
        VERIFIED_V1_OWN_WEIGHTS = {
            "Baker's": 0.35, "Contusion": 0.30, "Medial OA": 0.25, "ACL": 0.25,
            "Medial Meniscus": 0.20, "MCL": 0.20, "Lateral Meniscus": 0.08,
            "Fracture": 0.08, "Lateral OA": 0.08, "PF OA": 0.08,
            "Effusion": 0.04, "Synovitis": 0.03,
        }

        blended = pd.DataFrame(index=their_sub.index)
        if os.path.exists('/kaggle/working/_sota_0949.csv') and os.path.exists('/kaggle/working/_own.csv'):
            print("  --> Performing unquantized 3-way continuous rank fusion (Stage 1 SOTA + Stage 2 MIL + Stage 4 Creaky MoE)")
            sota_df = pd.read_csv('/kaggle/working/_sota_0949.csv', dtype={'StudyInstanceUID': str})
            own_df = pd.read_csv('/kaggle/working/_own.csv', dtype={'StudyInstanceUID': str})
            sota_df = sota_df.set_index('StudyInstanceUID').reindex(their_sub['StudyInstanceUID']).reset_index()
            own_df = own_df.set_index('StudyInstanceUID').reindex(their_sub['StudyInstanceUID']).reset_index()
            rank_sota = sota_df[TARGET_COLS].rank(method="average", pct=True)
            rank_own = own_df[TARGET_COLS].rank(method="average", pct=True)
            for col in TARGET_COLS:
                our_w = float(OPTIMIZED_TARGET_WEIGHTS.get(col, 0.03))
                v1_own_w = float(VERIFIED_V1_OWN_WEIGHTS.get(col, 0.15)) * (1.0 - our_w)
                sota_w = 1.0 - v1_own_w - our_w
                blended[col] = sota_w * rank_sota[col] + v1_own_w * rank_own[col] + our_w * our_ranks[col]
                print(f"Target '{col:16s}': {sota_w:.4f} SOTA_384 + {v1_own_w:.4f} MIL_2.5D + {our_w:.4f} Creaky_N4_C384")
            blended[TARGET_COLS] = blended[TARGET_COLS].rank(method="average", pct=True)
        else:
            their_ranks = their_sub[TARGET_COLS].rank(method="average", pct=True)
            for col in TARGET_COLS:
                our_w = float(OPTIMIZED_TARGET_WEIGHTS.get(col, 0.03))
                their_w = 1.0 - our_w
                blended[col] = their_w * their_ranks[col] + our_w * our_ranks[col]
                print(f"Target '{col:16s}': weight {our_w:.4f} Creaky_N4_C384 / {their_w:.4f} foundation")
            blended[TARGET_COLS] = blended[TARGET_COLS].rank(method="average", pct=True)

        their_sub[TARGET_COLS] = blended[TARGET_COLS]
        their_sub.to_csv('submission.csv', index=False)
        print('Blend complete! Wrote final submission.csv')
    except Exception as e:
        print('Blend failed! Outputting original submission.csv just to be safe.')
        print(e)
except Exception as _e:
    print('Our Creaky branch failed; restoring untouched public submission:', _e)
    _sh.copy('public_submission_backup.csv', 'submission.csv')"""

        assert old_tail in src, "Could not find old tail block in apex_cleanlab_blend"
        src = src.replace(old_tail, new_tail, 1)

    compile(src, "<stage4_patched>", "exec")
    return src


def main():
    # 1. Patch creaky-apex-cleanlab-blend.ipynb
    with open(APEX_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Also update Cell 9 in creaky-apex-cleanlab-blend.ipynb so Tier 3 fallback is anchored on verified 0.950 V1 weights
    cell9_src = "".join(nb["cells"][9]["source"])
    cell9_src = cell9_src.replace('"Lateral OA": 0.32,', '"Lateral OA": 0.08,')
    nb["cells"][9]["source"] = [l for l in cell9_src.splitlines(keepends=True)]

    # Update Cell 10 markdown
    nb["cells"][10]["source"] = [
        "## Stage 4: Creaky E24 Multi-Resolution Anatomical MoE Blend (N4 256px ConvNeXt + C384 384px CoAtNet, 0.8549 Gold AUC)\n\n",
        "Fuses our 3-fold Cleanlab-filtered ConvNeXt-Small MoE (256px) and native 384x384 CoAtNet MoE (`creaky-c384-cleanlab-models`) using Silver-OOF-validated per-pathology weights, then performs continuous 3-way rank fusion (1.5%–5.0% micro-injection) prior to final percentile normalization.\n",
    ]

    # Patch Cell 11 (Stage 4 code)
    cell11_src = "".join(nb["cells"][11]["source"])
    new_cell11 = patch_stage4_source(cell11_src, is_apex_nb=True)
    nb["cells"][11]["source"] = [l for l in new_cell11.splitlines(keepends=True)]

    with open(APEX_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully patched {APEX_NB}")

    # 2. Patch scripts/submission/submission_notebook.py
    with open(SUB_PY, "r", encoding="utf-8") as f:
        sub_src = f.read()
    new_sub_src = patch_stage4_source(sub_src, is_apex_nb=False)
    with open(SUB_PY, "w", encoding="utf-8") as f:
        f.write(new_sub_src)
    print(f"Successfully patched {SUB_PY}")

    # 3. Also sync notebooks/submission-notebook/submission-notebook.ipynb if present
    if os.path.exists(SUB_NB):
        with open(SUB_NB, "r", encoding="utf-8") as f:
            snb = json.load(f)
        for cell in snb["cells"]:
            if cell["cell_type"] == "code" and "def run_submission_inference" in "".join(cell["source"]):
                cell["source"] = [l for l in new_sub_src.splitlines(keepends=True)]
        with open(SUB_NB, "w", encoding="utf-8") as f:
            json.dump(snb, f, indent=1, ensure_ascii=False)
        print(f"Successfully synced {SUB_NB}")


if __name__ == "__main__":
    main()
