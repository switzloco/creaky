# Creaky-CV5 · OpenCV AI Competition 2026 (Powered by AWS)

**THE CMO (Chief MRI Officer) · Autonomous 3-Plane Knee MRI Triage, OpenCV 5 Active-Zoom & Radiologist Second-Opinion Cockpit**

- **Live Zero-Install Browser Cockpit:** [Open Interactive Cockpit](https://switzloco.github.io/creaky/opencv_hackathon/dist/index.html)
- **Live Physician & Radiologist Clinical Brief:** [Open Clinical Brief](https://switzloco.github.io/creaky/physician_pitch.html)
- **Live 3D Knee & UMAP Embedding Explorer ("Galaxy of Knees"):** [Open Galaxy of Knees](https://switzloco.github.io/creaky/galaxy_of_knees.html)
- **Full Technical Report & Devpost Writeup:** [`DEVPOST_SUBMISSION.md`](./DEVPOST_SUBMISSION.md)

---

## Repository Structure (`opencv_hackathon/`)

| File | Description |
| :--- | :--- |
| [`cv5_engine.py`](./cv5_engine.py) | Core OpenCV 5 (`cv2.dnn` ONNX probe, `cv2.MORPH_TOPHAT` HIPAA pixel PHI scrubber, FOV contour auto-crop, Rician bilateral + CLAHE, `cv2.Scharr` cortical bone edge gradient, Articular Surface Contact Rule, and `1.65× cv2.INTER_LANCZOS4` Active Focal Zoom) + AWS Graviton COOL benchmark. |
| [`app.py`](./app.py) | FastAPI server (`:8055`) + dual-viewport HTML5 `<canvas>` PACS Cockpit with Doctor/Patient view toggle, live drag-and-drop MRI scanner (`/api/analyze_upload`), click-to-steer focal zoom (`/api/agent_zoom`), and 16-slice 3D attention volume scrubber. |
| [`dist/index.html`](./dist/index.html) | Pre-rendered 100% standalone static HTML bundle containing all clinical cases, OpenCV 5 multi-modal outputs, and 16-slice volumetric scrubber for zero-install browser evaluation. |
| [`deploy_aws_cool.sh`](./deploy_aws_cool.sh) | 1-command bootstrap script for AWS EC2 Graviton (`c7g.xlarge` / `c8g.xlarge`, Ubuntu 24.04 ARM64) running the official AWS Marketplace OpenCV COOL (`KleidiCV`) AMI. |
| [`requirements.txt`](./requirements.txt) | Pinned Python dependencies for reproducible x86_64 and ARM64 deployment. |
| [`DEVPOST_SUBMISSION.md`](./DEVPOST_SUBMISSION.md) | Complete technical report covering problem, architecture, OpenCV 5 implementation, AWS COOL benchmarks, Agentic Vision traces, RSNA evaluation (`0.957` LB), failure modes, and HIPAA responsible-use governance. |

---

## Build, Deployment & Test Instructions

### 1. Install Pinned Dependencies
```bash
uv pip install -r opencv_hackathon/requirements.txt
```

### 2. Run Automated Verification Tests
```bash
uv run python -m unittest tests/test_opencv_hackathon.py -v
```

### 3. Run Live FastAPI + OpenCV 5 Cockpit (`http://localhost:8055`)
```bash
uv run python -m opencv_hackathon.app
```

### 4. Run Reproducible 144-Slice OpenCV 5 / AWS COOL Benchmark
```bash
uv run python -c "from opencv_hackathon.cv5_engine import run_opencv_cool_benchmark; import json; print(json.dumps(run_opencv_cool_benchmark(144), indent=2))"
```

### 5. Deploy on AWS EC2 Graviton (OpenCV COOL AMI)
```bash
bash opencv_hackathon/deploy_aws_cool.sh
```
