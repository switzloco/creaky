import json
import re
from pathlib import Path

def build_tpu_notebook(fold_num: int):
    source_nb_path = Path("notebooks/training-book-n4/training-book-n4.ipynb")
    nb = json.loads(source_nb_path.read_text(encoding="utf-8"))
    
    # 1. Update Markdown header
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                f"# RSNA Knee Abnormality Detection — Track N4 TPU VM (Fold {fold_num})\n",
                "\n",
                f"Trains KneeAnatomicalMoEClassifier (ConvNeXt-Small) on TPU v3-8 for Fold {fold_num}:\n",
                "- Hardware: Google Cloud TPU v3-8 (PyTorch XLA runtime)\n",
                "- Smart Sequence Prioritization: 94.2% Fluid-Sensitive (T2/PD FS) Sagittal & 96.4% Coronal slices.\n",
                "- Wide Depth Sampling: 2% to 98% volume depth (zero clipping of peripheral lateral meniscus).\n",
                "- Cleanlab Confident Learning: 2,634 noisy Silver labels zero-weighted (Abs_Diff >= 0.50).\n",
                "- Report Supervision: 64-D SVD radiology report fingerprint auxiliary head.\n",
                f"- Checkpoint Output: /kaggle/working/best_model_fold_{fold_num}.pt & val_preds_fold_{fold_num}.csv\n"
            ]
            break

    # 2. Update Code Cell for TPU + PyTorch XLA
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code_text = "".join(cell["source"])
            
            # (a) Fold configuration & output filenames
            code_text = re.sub(
                r'"fold":\s*\d+',
                f'"fold": {fold_num}',
                code_text
            )
            code_text = code_text.replace("best_model_fold_0.pt", f"best_model_fold_{fold_num}.pt")
            code_text = code_text.replace("val_preds_fold_0.csv", f"val_preds_fold_{fold_num}.csv")
            
            # (b) Environment setup for TPU XLA
            old_env = 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"'
            new_env = (
                'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n'
                'os.environ["PJRT_DEVICE"] = "TPU"\n'
                'import contextlib\n'
                'try:\n'
                '    import torch_xla.core.xla_model as xm\n'
                '    HAS_XLA = True\n'
                'except ImportError:\n'
                '    HAS_XLA = False'
            )
            if old_env in code_text:
                code_text = code_text.replace(old_env, new_env)

            # (c) Device discovery
            old_device = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    print(f"\\nTraining on Device: {device}")'
            new_device = (
                'if HAS_XLA:\n'
                '        device = xm.xla_device()\n'
                '        is_tpu = True\n'
                '        print(f"\\n[Accelerator] Initialized PyTorch XLA on TPU: {device}")\n'
                '    elif torch.cuda.is_available():\n'
                '        device = torch.device("cuda")\n'
                '        is_tpu = False\n'
                '        print(f"\\n[Accelerator] Initialized CUDA on GPU: {device}")\n'
                '    else:\n'
                '        device = torch.device("cpu")\n'
                '        is_tpu = False\n'
                '        print(f"\\n[Accelerator] Initialized CPU: {device}")'
            )
            if old_device in code_text:
                code_text = code_text.replace(old_device, new_device)

            # (d) Pin memory & DataLoader settings
            old_loader = 'train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)'
            new_loader = (
                'pin_mem = (device.type == "cuda")\n'
                '    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=pin_mem)'
            )
            if old_loader in code_text:
                code_text = code_text.replace(old_loader, new_loader)

            # (e) Scaler and Autocast context
            old_scaler = 'scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))'
            new_scaler = (
                'if device.type == "cuda":\n'
                '        scaler = torch.cuda.amp.GradScaler()\n'
                '        autocast_ctx = torch.cuda.amp.autocast\n'
                '    else:\n'
                '        scaler = None\n'
                '        autocast_ctx = contextlib.nullcontext'
            )
            if old_scaler in code_text:
                code_text = code_text.replace(old_scaler, new_scaler)

            # (f) Training forward & autocast
            old_train_fwd = 'with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):'
            if old_train_fwd in code_text:
                code_text = code_text.replace(old_train_fwd, 'with autocast_ctx():')

            # (g) Backward & Optimizer step with XLA support
            old_backward = (
                '            scaler.scale(loss).backward()\n\n'
                '            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):\n'
                '                scaler.step(optimizer)\n'
                '                scaler.update()\n'
                '                optimizer.zero_grad()'
            )
            new_backward = (
                '            if is_tpu:\n'
                '                loss.backward()\n'
                '                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):\n'
                '                    xm.optimizer_step(optimizer)\n'
                '                    optimizer.zero_grad()\n'
                '                    xm.mark_step()\n'
                '            elif scaler is not None:\n'
                '                scaler.scale(loss).backward()\n'
                '                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):\n'
                '                    scaler.step(optimizer)\n'
                '                    scaler.update()\n'
                '                    optimizer.zero_grad()\n'
                '            else:\n'
                '                loss.backward()\n'
                '                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):\n'
                '                    optimizer.step()\n'
                '                    optimizer.zero_grad()'
            )
            if old_backward in code_text:
                code_text = code_text.replace(old_backward, new_backward)

            # (h) Validation loop XLA sync
            old_val_fwd = (
                '                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):\n'
                '                    logits = model(sag, cor, ax)\n'
                '                    probs = torch.sigmoid(logits).cpu().numpy()'
            )
            new_val_fwd = (
                '                with autocast_ctx():\n'
                '                    logits = model(sag, cor, ax)\n'
                '                    probs = torch.sigmoid(logits)\n'
                '                if is_tpu:\n'
                '                    xm.mark_step()\n'
                '                probs = probs.cpu().numpy()'
            )
            if old_val_fwd in code_text:
                code_text = code_text.replace(old_val_fwd, new_val_fwd)

            # (i) Checkpoint saving with xm.save
            old_save = 'torch.save({'
            new_save = 'save_fn = xm.save if is_tpu else torch.save\n            save_fn({'
            if old_save in code_text:
                code_text = code_text.replace(old_save, new_save, 1)

            # Split back into lines
            cell["source"] = [line + "\n" for line in code_text.split("\n")[:-1]] + ([code_text.split("\n")[-1]] if code_text.split("\n")[-1] else [])
            break

    # 3. Create destination directory
    out_dir = Path(f"notebooks/training-book-n4-tpu-f{fold_num}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ipynb = out_dir / f"training-book-n4-tpu-f{fold_num}.ipynb"
    out_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {out_ipynb}")

    # 4. Generate kernel-metadata.json for TPU
    meta = {
        "id": f"nswitzer/training-book-n4-cleanlab-tpu-fold-{fold_num}",
        "title": f"Training Book N4 Cleanlab TPU (Fold {fold_num})",
        "code_file": f"training-book-n4-tpu-f{fold_num}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [
            "nswitzer/creaky-caching-dataset-n4"
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
        "model_sources": []
    }

    out_meta = out_dir / "kernel-metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Generated {out_meta}")

if __name__ == "__main__":
    for f in [1, 2]:
        build_tpu_notebook(f)
