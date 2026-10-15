import json
import re

GPU_NB_PATH = "notebooks/training-book-c384-gpu-f1/training-book-c384-gpu-f1.ipynb"
TPU_NB_PATH = "notebooks/training-book-c384-tpu-f0/training-book-c384-tpu-f0.ipynb"

NEW_SMOKE_BLOCK = """    # Smoke-test gate check
    import time
    smoke_steps = CONFIG.get("smoke_steps")
    if smoke_steps and isinstance(smoke_steps, int) and smoke_steps > 0:
        print("=======================================================")
        print(f"  RUNNING SMOKE TEST GATE ({smoke_steps} steps)")
        print(f"=======================================================")
        model.train()
        if use_report_aux:
            report_head.train()
        train_step_times = []
        train_iter = iter(train_loader)
        for s in range(smoke_steps):
            t0 = time.time()
            batch = next(train_iter)
            sag = batch["sagittal"].to(device)
            cor = batch["coronal"].to(device)
            ax = batch["axial"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)
            optimizer.zero_grad()
            with autocast_ctx():
                if use_report_aux:
                    logits, feats = model(sag, cor, ax, return_features=True)
                else:
                    logits = model(sag, cor, ax)
                loss = criterion(logits, targets, weights=weights)
                if use_report_aux:
                    mask = batch["has_report"].to(device)
                    if mask.sum() > 0:
                        pred_vec = report_head(feats.float())
                        cos = F.cosine_similarity(pred_vec, batch["report_vec"].to(device), dim=1)
                        aux_loss = ((1.0 - cos) * mask).sum() / mask.sum()
                        loss = loss + aux_w * aux_loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif is_tpu:
                loss.backward()
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                loss.backward()
                optimizer.step()
            dt = time.time() - t0
            if s >= 3:
                train_step_times.append(dt)
            print(f"  [Smoke Train Step {s+1}/{smoke_steps}] step_time={dt:.3f}s | loss={loss.item():.4f}")

        model.eval()
        if use_report_aux:
            report_head.eval()
        val_step_times = []
        val_iter = iter(val_loader)
        val_smoke_steps = min(smoke_steps, len(val_loader))
        with torch.no_grad():
            for s in range(val_smoke_steps):
                t0 = time.time()
                batch = next(val_iter)
                sag = batch["sagittal"].to(device)
                cor = batch["coronal"].to(device)
                ax = batch["axial"].to(device)
                with autocast_ctx():
                    _ = model(sag, cor, ax)
                if is_tpu:
                    xm.mark_step()
                dt = time.time() - t0
                if s >= 1:
                    val_step_times.append(dt)
                print(f"  [Smoke Val Step {s+1}/{val_smoke_steps}] step_time={dt:.3f}s")

        sec_train = float(np.mean(train_step_times)) if train_step_times else 0.0
        sec_val = float(np.mean(val_step_times)) if val_step_times else 0.0
        proj_hours = CONFIG["epochs"] * (len(train_loader) * sec_train + len(val_loader) * sec_val) / 3600.0

        peak_mem = "n/a"
        if torch.cuda.is_available():
            peak_mem = f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"

        print(f"SMOKE sec_per_train_step={sec_train:.3f} sec_per_val_step={sec_val:.3f} peak_mem={peak_mem} device={device}")
        print(f"SMOKE projected_hours={proj_hours:.2f}")
        print(f"=======================================================")
        print(f"SMOKE TEST COMPLETE. Exiting cleanly.")
        print("=======================================================")
        return
"""

def patch_notebook(nb_path):
    print(f"Patching {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the cell containing the smoke test gate
    target_cell_idx = None
    for idx, cell in enumerate(nb["cells"]):
        code = "".join(cell.get("source", []))
        if "RUNNING SMOKE TEST GATE" in code:
            target_cell_idx = idx
            break

    if target_cell_idx is None:
        raise ValueError(f"Could not find smoke test block in {nb_path}")

    cell_source = "".join(nb["cells"][target_cell_idx]["source"])

    # Regex find the entire block from '# Smoke-test gate check' to 'return\n'
    pattern = r"[ \t]*# Smoke-test gate check.*?(?=\n[ \t]*select_metric = CONFIG\.get)"
    match = re.search(pattern, cell_source, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not match smoke test regex pattern!")

    new_source = cell_source[:match.start()] + NEW_SMOKE_BLOCK.strip("\n") + "\n\n" + cell_source[match.end():]

    # Verify python syntax
    compile(new_source, f"<{nb_path}>", "exec")
    print(f"  Successfully compiled python source in {nb_path}!")

    # Format back as lines
    lines = [line + "\n" for line in new_source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    # Ensure correct jupyter format
    nb["cells"][target_cell_idx]["source"] = [l if idx == len(lines)-1 and not l.endswith("\n") else l for idx, l in enumerate(new_source.splitlines(keepends=True))]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved {nb_path} successfully.")

if __name__ == "__main__":
    patch_notebook(GPU_NB_PATH)
    patch_notebook(TPU_NB_PATH)
    print("ALL NOTEBOOKS PATCHED AND COMPILED CLEANLY!")
