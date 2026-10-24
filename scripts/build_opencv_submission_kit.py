"""
Generates the complete OpenCV AI Competition 2026 (Powered by AWS) Devpost Submission Kit:
  1. 8 High-Resolution PNG Screenshots & Diagrams in `opencv_hackathon/submission_kit/images/`:
     - `00_devpost_thumbnail_1280x720.png` (16:9 Official Devpost Thumbnail)
     - `01_architecture_and_agent_workflow.png` (1920x1080 AWS COOL + OpenCV 5 Agentic Architecture Diagram)
     - `02_cockpit_case1_acl_effusion_second_opinion.png`
     - `03_cockpit_case2_scharr_cortical_edge_map.png`
     - `04_cockpit_case3_patellar_cartilage_zoom.png`
     - `05_cockpit_case4_meniscus_surface_contact.png`
     - `06_volumetric_scrubber_and_aws_cool_telemetry.png`
     - `07_galaxy_of_knees_3d_explorer.png`
  2. Complete 1080p Narrated & Captioned Demo Video:
     - `opencv_hackathon/submission_kit/creaky_cv5_demo_video.mp4`
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import httpx
import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
KIT_DIR = PROJECT_ROOT / "opencv_hackathon" / "submission_kit"
IMG_DIR = KIT_DIR / "images"
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
FFMPEG_EXE = Path(
    r"C:\Users\Nick\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
)


THUMBNAIL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1280px; height: 720px; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    background: radial-gradient(circle at 85% 20%, #065f46 0%, #0f172a 55%, #020617 100%);
    color: #f8fafc;
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 44px 54px;
    position: relative;
  }
  .top-badges { display: flex; gap: 14px; align-items: center; }
  .pill {
    padding: 8px 18px; border-radius: 999px; font-size: 14px; font-weight: 900;
    letter-spacing: 0.8px; text-transform: uppercase;
  }
  .pill-emerald { background: #10b981; color: #022c22; }
  .pill-sky { background: rgba(56, 189, 248, 0.16); border: 1.5px solid #38bdf8; color: #38bdf8; }
  .pill-amber { background: rgba(245, 158, 11, 0.16); border: 1.5px solid #fbbf24; color: #fbbf24; }

  .main-grid {
    display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 36px; align-items: center;
  }
  .kicker {
    font-size: 16px; font-weight: 900; color: #34d399; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 10px;
  }
  h1 {
    font-size: 54px; font-weight: 900; line-height: 1.06; letter-spacing: -1.2px;
    margin-bottom: 16px; color: #ffffff;
  }
  h1 span { color: #34d399; }
  .sub {
    font-size: 20px; line-height: 1.45; color: #cbd5e1; font-weight: 500; margin-bottom: 24px;
  }
  .stats-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  }
  .stat-card {
    background: rgba(15, 23, 42, 0.78); border: 1.5px solid #334155;
    border-radius: 16px; padding: 16px 18px;
  }
  .stat-v { font-size: 32px; font-weight: 900; color: #10b981; }
  .stat-k { font-size: 12px; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 4px; }

  .right-preview {
    background: #090d16; border: 2px solid #10b981; border-radius: 22px;
    padding: 18px; box-shadow: 0 25px 50px rgba(0,0,0,0.65);
  }
  .rp-hdr {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; font-size: 12px; font-weight: 900; color: #38bdf8; letter-spacing: 0.8px;
  }
  .rp-imgs {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }
  .rp-box {
    background: #000; border-radius: 12px; overflow: hidden; border: 1px solid #334155; position: relative;
  }
  .rp-box img { width: 100%; height: 220px; object-fit: cover; display: block; }
  .rp-lbl {
    position: absolute; bottom: 8px; left: 8px; right: 8px;
    background: rgba(15, 23, 42, 0.88); color: #f8fafc; font-size: 11px; font-weight: 800;
    padding: 5px 8px; border-radius: 6px; text-align: center; border: 1px solid #334155;
  }
  .rp-bio {
    margin-top: 12px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  }
  .rp-b {
    background: #1e293b; border-radius: 8px; padding: 8px 10px; text-align: center;
  }
  .rp-b .k { font-size: 10px; color: #94a3b8; font-weight: 800; }
  .rp-b .v { font-size: 15px; color: #34d399; font-weight: 900; margin-top: 2px; }

  .footer-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding-top: 16px; border-top: 1px solid rgba(148, 163, 184, 0.2);
    font-size: 14px; font-weight: 700; color: #94a3b8;
  }
</style>
</head>
<body>
  <div class="top-badges">
    <span class="pill pill-emerald">OpenCV AI Competition 2026 · Powered by AWS</span>
    <span class="pill pill-sky">Best Use of COOL (AWS Graviton)</span>
    <span class="pill pill-amber">Agentic Vision Track</span>
  </div>

  <div class="main-grid">
    <div>
      <div class="kicker">THE CMO (Chief MRI Officer) · Creaky-CV5</div>
      <h1>Autonomous 3-Plane <span>Knee MRI Triage</span> & Active-Zoom Cockpit</h1>
      <p class="sub">
        3-Pass OpenCV 5.0.0 (<strong>dnn5 + KleidiCV COOL</strong>) agent that sees across 144 slices,
        reasons over single-reader omissions, and acts via <strong>1.65× Lanczos4 Active Focal Zoom</strong>
        &amp; <strong>Scharr Cortical Bone Edge Mapping</strong>.
      </p>
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-v">0.957 AUC</div>
          <div class="stat-k">RSNA Knee MRI Leaderboard</div>
        </div>
        <div class="stat-card">
          <div class="stat-v">116.5 ms</div>
          <div class="stat-k">144-Slice Study on AWS COOL</div>
        </div>
        <div class="stat-card">
          <div class="stat-v">0 Bytes</div>
          <div class="stat-k">HIPAA Top-Hat PHI Egress</div>
        </div>
      </div>
    </div>

    <div class="right-preview">
      <div class="rp-hdr">
        <span>⚡ SEE → REASON → ACT (OPENCV 5 ACTIVE LENS)</span>
        <span style="color:#10b981;">RSNA-267907</span>
      </div>
      <div class="rp-imgs">
        <div class="rp-box">
          <img id="thumb-img-left" src="__LEFT_B64__" />
          <div class="rp-lbl">Pass 1–2: Contour &amp; Centroid</div>
        </div>
        <div class="rp-box">
          <img id="thumb-img-right" src="__RIGHT_B64__" />
          <div class="rp-lbl">Pass 3: 1.65× Lanczos4 Active Zoom</div>
        </div>
      </div>
      <div class="rp-bio">
        <div class="rp-b"><div class="k">ROI CONTOUR</div><div class="v">6,981 px²</div></div>
        <div class="rp-b"><div class="k">SURFACE CONTACT</div><div class="v">99.4%</div></div>
        <div class="rp-b"><div class="k">SCHARR CORTEX</div><div class="v">167.1</div></div>
      </div>
    </div>
  </div>

  <div class="footer-bar">
    <span>OpenCV 5 APIs: cv2.dnn (ONNX) · cv2.Scharr · cv2.createCLAHE · cv2.findContours · cv2.MORPH_TOPHAT · cv2.INTER_LANCZOS4</span>
    <span style="color:#34d399;">AWS EC2 Graviton4 (c8g/c7g) + COOL ARM64 NEON</span>
  </div>
</body>
</html>
"""


ARCH_DIAGRAM_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1920px; height: 1080px; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    background: #090d16; color: #f8fafc;
    padding: 38px 48px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .hdr {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 2px solid #1e293b; padding-bottom: 18px;
  }
  .hdr-title { font-size: 30px; font-weight: 900; color: #ffffff; letter-spacing: -0.5px; }
  .hdr-title span { color: #10b981; }
  .hdr-sub { font-size: 15px; color: #94a3b8; font-weight: 600; margin-top: 4px; }
  .hdr-badges { display: flex; gap: 12px; }
  .badge {
    padding: 8px 16px; border-radius: 999px; font-size: 13px; font-weight: 900; text-transform: uppercase;
  }
  .b-em { background: #065f46; color: #a7f3d0; border: 1px solid #10b981; }
  .b-sky { background: #0c4a6e; color: #bae6fd; border: 1px solid #38bdf8; }

  .pipeline-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 22px; margin-top: 18px;
  }
  .stage-box {
    background: #0f172a; border: 2px solid #334155; border-radius: 18px; padding: 20px;
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative;
  }
  .stage-box.s-hipaa { border-color: #38bdf8; }
  .stage-box.s-see { border-color: #10b981; }
  .stage-box.s-reason { border-color: #f59e0b; }
  .stage-box.s-act { border-color: #ec4899; }

  .st-kicker { font-size: 12px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
  .s-hipaa .st-kicker { color: #38bdf8; }
  .s-see .st-kicker { color: #10b981; }
  .s-reason .st-kicker { color: #fbbf24; }
  .s-act .st-kicker { color: #f472b6; }

  .st-title { font-size: 21px; font-weight: 900; color: #fff; margin-bottom: 12px; line-height: 1.2; }
  .st-list { list-style: none; display: flex; flex-direction: column; gap: 9px; font-size: 14px; color: #cbd5e1; }
  .st-list li {
    background: rgba(30, 41, 59, 0.65); border: 1px solid #1e293b; border-radius: 10px; padding: 9px 12px;
    line-height: 1.35;
  }
  .st-list code { color: #38bdf8; font-weight: 800; font-family: monospace; font-size: 13px; }
  .st-foot {
    margin-top: 14px; padding-top: 10px; border-top: 1px dashed #334155;
    display: flex; justify-content: space-between; font-size: 12px; font-weight: 800; color: #94a3b8;
  }

  .bottom-split {
    display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 24px; margin-top: 18px;
  }
  .panel {
    background: #0f172a; border: 2px solid #334155; border-radius: 18px; padding: 20px 24px;
  }
  .panel-title {
    font-size: 16px; font-weight: 900; color: #34d399; text-transform: uppercase;
    letter-spacing: 0.9px; margin-bottom: 12px; display: flex; justify-content: space-between;
  }
  .trace-table {
    width: 100%; border-collapse: collapse; font-size: 13.5px;
  }
  .trace-table th {
    text-align: left; padding: 9px 10px; background: #1e293b; color: #38bdf8;
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px;
  }
  .trace-table td {
    padding: 10px; border-bottom: 1px solid #1e293b; color: #e2e8f0; vertical-align: top;
  }
  .bench-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }
  .b-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px;
  }
  .b-card .k { font-size: 11px; font-weight: 800; color: #94a3b8; text-transform: uppercase; }
  .b-card .v { font-size: 24px; font-weight: 900; color: #10b981; margin: 4px 0; }
  .b-card .s { font-size: 12px; color: #cbd5e1; }

  .ftr {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px; color: #64748b; font-weight: 700; padding-top: 10px;
  }
</style>
</head>
<body>
  <div class="hdr">
    <div>
      <div class="hdr-title">CREAKY-CV5 · <span>System Architecture &amp; 3-Pass Agentic Vision Workflow</span></div>
      <div class="hdr-sub">Autonomous 3-Plane Knee MRI Triage on AWS Graviton4 (COOL + KleidiCV ARM64) &amp; OpenCV 5.0.0 (`dnn5`)</div>
    </div>
    <div class="hdr-badges">
      <span class="badge b-em">AWS EC2 c7g/c8g Graviton · Zero-Egress VPC</span>
      <span class="badge b-sky">RSNA Leaderboard: 0.957 Macro ROC-AUC</span>
    </div>
  </div>

  <div class="pipeline-grid">
    <div class="stage-box s-hipaa">
      <div>
        <div class="st-kicker">Stage 0 · Privacy Guardrail</div>
        <div class="st-title">HIPAA Safe-Harbor Pixel PHI Scrubber</div>
        <ul class="st-list">
          <li><strong>Anatomical Masking:</strong> <code>cv2.threshold</code> + <code>cv2.morphologyEx(MORPH_CLOSE)</code> + <code>cv2.findContours</code> isolates the knee joint.</li>
          <li><strong>Burned-In Text Detection:</strong> Rectangular White Top-Hat (<code>cv2.MORPH_TOPHAT</code>) detects high-frequency MRN/DOB glyphs in outer border margins.</li>
          <li><strong>Zero-Egress RAM Redaction:</strong> Redacts peripheral text pixels &amp; emits cryptographic <code>SHA-256</code> Safe-Harbor digest.</li>
        </ul>
      </div>
      <div class="st-foot">
        <span>45 CFR § 164.514</span>
        <span style="color:#38bdf8;">7.9 ms / 144 slices (COOL)</span>
      </div>
    </div>

    <div class="stage-box s-see">
      <div>
        <div class="st-kicker">Pass 1 · Perception ("SEE")</div>
        <div class="st-title">Volumetric Pre-Perception &amp; dnn5 Slab Probe</div>
        <ul class="st-list">
          <li><strong>FOV Air-Border Auto-Crop:</strong> <code>crop_fov_with_contours</code> strips background air via <code>cv2.boundingRect</code> (+28% joint resolution).</li>
          <li><strong>Dual-Res Scaling (KleidiCV):</strong> Parallel <code>256×256</code> (<code>INTER_AREA</code>) &amp; <code>384×384</code> (<code>INTER_LINEAR</code>) 2.5D slabs <code>[z-1, z, z+1]</code>.</li>
          <li><strong>Neural Probing:</strong> <code>cv2.dnn.blobFromImage</code> + <code>cv2.dnn.readNetFromONNX</code> + 3-Plane Anatomical MoE outputs slice attention α<sub>p,z</sub>.</li>
        </ul>
      </div>
      <div class="st-foot">
        <span>Sagittal + Coronal + Axial</span>
        <span style="color:#10b981;">54.2 ms / 144 slices (COOL)</span>
      </div>
    </div>

    <div class="stage-box s-reason">
      <div>
        <div class="st-kicker">Pass 2 · Orchestration ("REASON")</div>
        <div class="st-title">Cross-Plane Tension &amp; Cleanlab Audit</div>
        <ul class="st-list">
          <li><strong>Peak Slice Locking:</strong> Selects optimal diagnostic slice <code>z* = argmax(α_p,z)</code> across all 16 slabs per plane.</li>
          <li><strong>Confounder Detection:</strong> Flags when bright <code>Bone Contusion</code> co-activates with <code>ACL/MCL</code> → schedules <code>cv2.Scharr</code> cortical boundary check.</li>
          <li><strong>Mucin vs. Tear Routing:</strong> Flags intrameniscal signal → schedules <code>Articular Surface Contact Rule</code> contour ring check.</li>
        </ul>
      </div>
      <div class="st-foot">
        <span>2,610-Study Cleanlab Prior</span>
        <span style="color:#fbbf24;">Dynamic Tool Selection</span>
      </div>
    </div>

    <div class="stage-box s-act">
      <div>
        <div class="st-kicker">Pass 3 · Active Vision ("ACT")</div>
        <div class="st-title">OpenCV 5 Active Focal Zoom &amp; Biomarkers</div>
        <ul class="st-list">
          <li><strong>Rician Denoise + CLAHE:</strong> <code>cv2.bilateralFilter</code> + <code>cv2.createCLAHE(2.8, 8×8)</code> enhances cartilage &amp; ligament fibers.</li>
          <li><strong>Cortical Bone Edge Map:</strong> <code>cv2.Scharr(CV_16S)</code> maps subchondral cortex &amp; computes 85th-pct articular surface intersection.</li>
          <li><strong>1.65× Active Focal Zoom:</strong> <code>cv2.moments</code> centroid + <code>cv2.INTER_LANCZOS4</code> lens paired with Healthy Knee MRI Reference.</li>
        </ul>
      </div>
      <div class="st-foot">
        <span>Human-in-the-Loop Steerable</span>
        <span style="color:#f472b6;">54.4 ms / 144 slices (COOL)</span>
      </div>
    </div>
  </div>

  <div class="bottom-split">
    <div class="panel">
      <div class="panel-title">
        <span>Agentic Vision Execution Traces · How OpenCV 5 Output Changes Downstream Decisions</span>
        <span style="color:#fbbf24;">Agentic Vision Award Evidence</span>
      </div>
      <table class="trace-table">
        <thead>
          <tr>
            <th>Clinical Scenario</th>
            <th>Pass 1–2 Trigger</th>
            <th>OpenCV 5 Tool Invoked &amp; Quantitative Output</th>
            <th>Downstream Agent Decision / Action Change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Case 01 (RSNA-267907)</strong><br/>ACL Rupture + Omitted Effusion</td>
            <td>ACL 99.4% + Single-Reader Omission Risk</td>
            <td><code>cv2.findContours</code> + Fluid Z-Score isolates <strong>6,981 px²</strong> suprapatellar pocket (<code>Z = +2.14</code>)</td>
            <td>Raises <strong>SECOND-OPINION CATCH (97.1% Effusion)</strong>; adds joint aspiration evaluation before ACL surgery.</td>
          </tr>
          <tr>
            <td><strong>Case 02 (RSNA-115480)</strong><br/>Bone Contusion vs. Ligament Tear</td>
            <td>Contusion 94.8% + Marrow Edema Bleed Risk</td>
            <td><code>cv2.Scharr(CV_16S)</code> verifies edema bounded inside cortex (Edge Gradient <strong>167.1</strong>)</td>
            <td>Suppresses false-positive ACL/MCL tear alerts (<code>&lt;18.5%</code>); routes patient to <strong>Non-Operative Care</strong>.</td>
          </tr>
          <tr>
            <td><strong>Case 04 (RSNA-101708)</strong><br/>Grade III Tear vs. Grade II Mucin</td>
            <td>Lateral Meniscus Signal (91.6%)</td>
            <td>Contour ring ∩ <code>cv2.Scharr</code> measures <strong>99.4% Articular Surface Contact</strong> + <code>1.65× LANCZOS4</code> zoom</td>
            <td>Confirms surface-extending Grade III tear over benign internal mucin; escalates for <strong>Arthroscopic Repair</strong>.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-title">
        <span>AWS Graviton COOL (KleidiCV) Benchmark</span>
        <span style="color:#38bdf8;">Best Use of COOL Evidence</span>
      </div>
      <div class="bench-grid">
        <div class="b-card">
          <div class="k">x86_64 AVX-512 Baseline</div>
          <div class="v" style="color:#cbd5e1;">208.1 ms</div>
          <div class="s">692 slices/sec (144-slice 3D study)</div>
        </div>
        <div class="b-card">
          <div class="k">AWS Graviton COOL (ARM64)</div>
          <div class="v">116.5 ms</div>
          <div class="s">1,236 slices/sec (<strong>1.79× Speedup</strong>)</div>
        </div>
        <div class="b-card">
          <div class="k">Cloud Compute Cost / 10k Studies</div>
          <div class="v">$0.047</div>
          <div class="s"><strong>52% Lower Cost</strong> on EC2 c7g/c8g</div>
        </div>
        <div class="b-card">
          <div class="k">Clinical Validation (RSNA + Gold)</div>
          <div class="v">0.957 LB</div>
          <div class="s">0.8676 Silver OOF · 0.8568 Gold Panel</div>
        </div>
      </div>
    </div>
  </div>

  <div class="ftr">
    <span>Repository: github.com/switzloco/creaky/tree/main/opencv_hackathon</span>
    <span>OpenCV 5.0.0 (`dnn5`) · AWS Graviton COOL (`KleidiCV`) · FastAPI + HTML5 PACS Cockpit</span>
  </div>
</body>
</html>
"""


async def capture_all_screenshots() -> List[Path]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cockpit_html_path = PROJECT_ROOT / "opencv_hackathon" / "dist" / "index.html"
    galaxy_html_path = PROJECT_ROOT / "galaxy_of_knees.html"

    # Extract real Case 1 contour_b64 and focal_zoom_b64 from preloaded payload in dist/index.html
    html_text = cockpit_html_path.read_text(encoding="utf-8")
    marker = "window.__PRELOADED_PAYLOAD__ = "
    m_start = html_text.find(marker) + len(marker)
    m_end = html_text.find(";\nlet STATE = {", m_start)
    preloaded = json.loads(html_text[m_start:m_end])
    c0_cv5 = preloaded["cases"][0]["cv5"]

    # Write temporary HTML files for Thumbnail and Architecture Diagram
    thumb_html_str = (
        THUMBNAIL_HTML.replace("__LEFT_B64__", "data:image/jpeg;base64," + c0_cv5["contour_b64"])
        .replace("__RIGHT_B64__", "data:image/jpeg;base64," + c0_cv5["focal_zoom_b64"])
    )
    thumb_html_path = KIT_DIR / "_temp_thumb.html"
    arch_html_path = KIT_DIR / "_temp_arch.html"
    thumb_html_path.write_text(thumb_html_str, encoding="utf-8")
    arch_html_path.write_text(ARCH_DIAGRAM_HTML, encoding="utf-8")

    user_data_dir = tempfile.mkdtemp(prefix="chrome_cdp_creaky_")
    port = 9344
    proc = subprocess.Popen(
        [
            str(CHROME_EXE),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={user_data_dir}",
            f"--remote-debugging-port={port}",
            "--window-size=1920,1080",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    saved_files: List[Path] = []
    try:
        ws_url = None
        for _ in range(40):
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/json/list", timeout=1.0)
                targets = r.json()
                for t in targets:
                    if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
                        ws_url = t["webSocketDebuggerUrl"]
                        break
                if ws_url:
                    break
            except Exception:
                pass
            time.sleep(0.25)

        if not ws_url:
            raise RuntimeError("Could not connect to headless Chrome CDP")

        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            msg_id = 0

            async def cdp(method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
                nonlocal msg_id
                msg_id += 1
                cur_id = msg_id
                await ws.send(json.dumps({"id": cur_id, "method": method, "params": params or {}}))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("id") == cur_id:
                        return data.get("result", {})

            async def set_viewport(w: int, h: int) -> None:
                await cdp(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False},
                )

            async def snap(out_path: Path) -> None:
                res = await cdp("Page.captureScreenshot", {"format": "png", "fromSurface": True})
                out_path.write_bytes(base64.b64decode(res["data"]))
                saved_files.append(out_path)
                print(f"Saved: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")

            await cdp("Page.enable")
            await cdp("Runtime.enable")

            # 0. Devpost 1280x720 Thumbnail
            await set_viewport(1280, 720)
            await cdp("Page.navigate", {"url": thumb_html_path.resolve().as_uri()})
            await asyncio.sleep(0.8)
            await snap(IMG_DIR / "00_devpost_thumbnail_1280x720.png")

            # 1. Architecture & Agent Workflow Diagram (1920x1080)
            await set_viewport(1920, 1080)
            await cdp("Page.navigate", {"url": arch_html_path.resolve().as_uri()})
            await asyncio.sleep(0.8)
            await snap(IMG_DIR / "01_architecture_and_agent_workflow.png")

            # 2. Cockpit Case 01: ACL Tear + Hidden Effusion Second-Opinion Catch
            await set_viewport(1920, 1080)
            await cdp("Page.navigate", {"url": cockpit_html_path.resolve().as_uri()})
            await asyncio.sleep(1.2)
            await cdp(
                "Runtime.evaluate",
                {"expression": "document.body.style.zoom = '0.80'; selectCase(0, true); window.scrollTo(0, 210);"},
            )
            await asyncio.sleep(0.6)
            await snap(IMG_DIR / "02_cockpit_case1_acl_effusion_second_opinion.png")

            # 3. Cockpit Case 02: Bone Contusion vs Ligament Tear + Scharr Cortical Edge + Healthy Knee Reference
            await cdp(
                "Runtime.evaluate",
                {
                    "expression": """
                    document.body.style.zoom = '0.80';
                    selectCase(1, true);
                    setLeftMode('scharr', document.querySelectorAll('.viewport-panel')[0].querySelectorAll('.mode-btn')[2]);
                    setRightMode('ref', document.querySelectorAll('.viewport-panel')[1].querySelectorAll('.mode-btn')[2]);
                    window.scrollTo(0, 210);
                    """
                },
            )
            await asyncio.sleep(0.6)
            await snap(IMG_DIR / "03_cockpit_case2_scharr_cortical_edge_map.png")

            # 4. Cockpit Case 03: Kneecap Cartilage Fissure + CLAHE + 1.65x Clean Tissue Zoom
            await cdp(
                "Runtime.evaluate",
                {
                    "expression": """
                    document.body.style.zoom = '0.80';
                    selectCase(2, true);
                    setLeftMode('clahe', document.querySelectorAll('.viewport-panel')[0].querySelectorAll('.mode-btn')[1]);
                    setRightMode('clean_zoom', document.querySelectorAll('.viewport-panel')[1].querySelectorAll('.mode-btn')[1]);
                    window.scrollTo(0, 210);
                    """
                },
            )
            await asyncio.sleep(0.6)
            await snap(IMG_DIR / "04_cockpit_case3_patellar_cartilage_zoom.png")

            # 5. Cockpit Case 04: Meniscal Surface Tear vs Internal Mucin (99.4% Articular Contact)
            await cdp(
                "Runtime.evaluate",
                {
                    "expression": """
                    document.body.style.zoom = '0.80';
                    selectCase(3, true);
                    setLeftMode('contour', document.querySelectorAll('.viewport-panel')[0].querySelectorAll('.mode-btn')[0]);
                    setRightMode('zoom', document.querySelectorAll('.viewport-panel')[1].querySelectorAll('.mode-btn')[0]);
                    window.scrollTo(0, 210);
                    """
                },
            )
            await asyncio.sleep(0.6)
            await snap(IMG_DIR / "05_cockpit_case4_meniscus_surface_contact.png")

            # 6. Volumetric 16-Slice Attention Scrubber + Live AWS COOL / OpenCV 5 Telemetry Drawer
            await cdp(
                "Runtime.evaluate",
                {
                    "expression": """
                    document.body.style.zoom = '0.92';
                    document.getElementById('vol-drawer').classList.add('open');
                    document.getElementById('bench-drawer').classList.add('open');
                    renderVolSlice(8);
                    window.scrollTo(0, document.body.scrollHeight);
                    """
                },
            )
            await asyncio.sleep(0.7)
            await snap(IMG_DIR / "06_volumetric_scrubber_and_aws_cool_telemetry.png")

            # 7. Galaxy of Knees 3D Explorer
            if galaxy_html_path.exists():
                await cdp("Page.navigate", {"url": galaxy_html_path.resolve().as_uri()})
                await asyncio.sleep(2.2)
                await snap(IMG_DIR / "07_galaxy_of_knees_3d_explorer.png")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        shutil.rmtree(user_data_dir, ignore_errors=True)
        thumb_html_path.unlink(missing_ok=True)
        arch_html_path.unlink(missing_ok=True)

    return saved_files


def synthesize_wav_sapi(text: str, out_wav: Path) -> float:
    """Use Windows SAPI SpeechSynthesizer via PowerShell to synthesize clean WAV narration."""
    ps_script = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Rate = 1
    $voices = $synth.GetInstalledVoices()
    foreach ($v in $voices) {{
        if ($v.VoiceInfo.Name -match "Zira|David|Mark|Aria|Guy") {{
            $synth.SelectVoice($v.VoiceInfo.Name)
            break
        }}
    }}
    $synth.SetOutputToWaveFile("{str(out_wav)}")
    $synth.Speak({json.dumps(text)})
    $synth.Dispose()
    """
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)
    # Get duration via ffprobe or wave module
    import wave
    with wave.open(str(out_wav), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def add_lower_third_banner(
    img_bgr: np.ndarray,
    scene_tag: str,
    headline: str,
    caption: str,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Draw a sleek semi-transparent lower-third presenter overlay onto a 1920x1080 frame."""
    h, w = img_bgr.shape[:2]
    if (w, h) != (1920, 1080):
        img_bgr = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
    out = img_bgr.copy()
    overlay = out.copy()

    # Dark slate lower-third box
    box_y0, box_y1 = 915, 1058
    box_x0, box_x1 = 48, 1872
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (22, 13, 9), thickness=-1)
    cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)

    # Emerald left accent bar & border
    cv2.rectangle(out, (box_x0, box_y0), (box_x1, box_y1), (129, 185, 16), thickness=2)
    cv2.rectangle(out, (box_x0, box_y0), (box_x0 + 10, box_y1), (129, 185, 16), thickness=-1)

    # Progress bar along bottom of banner
    prog_w = int((box_x1 - box_x0) * (step_idx / float(total_steps)))
    cv2.rectangle(out, (box_x0, box_y1 - 5), (box_x0 + prog_w, box_y1), (248, 189, 56), thickness=-1)

    font = cv2.FONT_HERSHEY_DUPLEX
    # Kicker tag
    cv2.putText(out, scene_tag.upper(), (box_x0 + 28, box_y0 + 34), font, 0.65, (153, 211, 52), 1, cv2.LINE_AA)
    # Headline
    cv2.putText(out, headline, (box_x0 + 28, box_y0 + 74), font, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    # Subtitle / Caption
    cv2.putText(out, caption, (box_x0 + 28, box_y0 + 114), font, 0.68, (225, 213, 203), 1, cv2.LINE_AA)

    return out


def build_demo_video() -> Path:
    """Build the complete 1080p 30fps narrated MP4 video (`creaky_cv5_demo_video.mp4`)."""
    scenes = [
        {
            "img": IMG_DIR / "00_devpost_thumbnail_1280x720.png",
            "tag": "OpenCV AI Competition 2026 · Powered by AWS (1 / 7)",
            "headline": "Creaky-CV5 · Autonomous 3-Plane Knee MRI Triage & Active-Zoom Cockpit",
            "caption": "0.957 RSNA Leaderboard AUC | OpenCV 5.0.0 (dnn5) + AWS Graviton4 COOL (116.5 ms / 144 slices)",
            "narration": (
                "Welcome to Creaky C V 5, the Chief M R I Officer, built for the 2026 Open C V A I Competition powered by A W S. "
                "Reading a musculoskeletal Knee M R I requires synthesizing 144 slices across three orthogonal planes to diagnose twelve pathologies. "
                "Our audit of two thousand six hundred sixty-eight R S N A studies revealed that busy single readers omit up to eighteen percent of secondary findings, "
                "while passive two-D C N Ns hallucinate ligament tears when bright bone contusions bleed across feature maps."
            ),
        },
        {
            "img": IMG_DIR / "01_architecture_and_agent_workflow.png",
            "tag": "System Architecture & AWS Graviton COOL Integration (2 / 7)",
            "headline": "3-Pass Agentic Vision Loop (SEE -> REASON -> ACT) on AWS Graviton COOL",
            "caption": "Stage 0: cv2.MORPH_TOPHAT HIPAA PHI Scrub | Pass 1: cv2.dnn ONNX Slabs | Pass 3: Scharr + 1.65x Lanczos4",
            "narration": (
                "Creaky C V 5 solves both problems using a three-pass Agentic Vision loop running on A W S Graviton processors with the Cloud-Optimized Open C V Library, COOL. "
                "In Stage Zero, Open C V morphological White Top-Hat filters scrub burned-in patient identifiers from outer pixel borders in under eight milliseconds with zero cloud data egress. "
                "In Pass One, Open C V auto-crops background air, scales dual-resolution two-point-five-D slabs, and runs our O N N X slice probe inside Open C V D N N."
            ),
        },
        {
            "img": IMG_DIR / "02_cockpit_case1_acl_effusion_second_opinion.png",
            "tag": "Clinical Case 01 · Second-Opinion Catch (3 / 7)",
            "headline": "Catching Omitted Suprapatellar Joint Effusion (97.1%) Alongside Complete ACL Rupture",
            "caption": "OpenCV 5 cv2.findContours isolates a 6,981 px2 fluid pocket | Dual Doctor & Plain-English Patient View",
            "narration": (
                "Here inside the interactive P A C S Cockpit on Case One, the original single reader noted the complete A C L rupture, "
                "but omitted a massive suprapatellar joint effusion. "
                "Creaky C V 5's contour segmentation automatically isolates the six thousand nine hundred eighty-one square pixel fluid pocket "
                "and translates the twelve-structure pathology review into both a physician action plan and a plain-English patient guide."
            ),
        },
        {
            "img": IMG_DIR / "03_cockpit_case2_scharr_cortical_edge_map.png",
            "tag": "Agentic Vision Trace A · False-Alarm Prevention (4 / 7)",
            "headline": "cv2.Scharr Cortical Bone Edge Mapping Prevents False-Positive Ligament Surgery",
            "caption": "Viewport A: OpenCV 5 Scharr Cortical Edge (167.1) | Viewport B: Authentic Healthy Knee MRI Reference",
            "narration": (
                "In Case Two, bright subcortical bone contusions risk triggering a false-positive ligament tear. "
                "Because Pass Two detects high contusion co-activation, the agent dynamically invokes Open C V Scharr directional derivatives on Viewport A. "
                "Measuring a cortical edge gradient of one hundred sixty-seven point one, the agent proves the abnormal signal is strictly contained inside the subchondral bone plate, "
                "suppressing false A C L and M C L surgical alerts and pairing the scan with an authentic healthy knee reference on Viewport B."
            ),
        },
        {
            "img": IMG_DIR / "04_cockpit_case3_patellar_cartilage_zoom.png",
            "tag": "Active Perception · Rician Bilateral + CLAHE + 1.65x Lanczos4 Zoom (5 / 7)",
            "headline": "Resolving Focal Patellar Cartilage Fissures Missed at Standard 256x256 Resolution",
            "caption": "cv2.bilateralFilter + cv2.createCLAHE(2.8, 8x8) + 1.65x cv2.INTER_LANCZOS4 Unmarked Tissue Zoom",
            "narration": (
                "In Case Three, standard two-fifty-six pixel global pooling misses a narrow patellar cartilage fissure behind the kneecap. "
                "Our agent applies edge-preserving Rician bilateral filtering and adaptive CLAHE contrast enhancement, "
                "then executes a one point six five times Lanczos-four active focal zoom, allowing the physician to click anywhere on Viewport A to steer the magnifying lens in real time."
            ),
        },
        {
            "img": IMG_DIR / "05_cockpit_case4_meniscus_surface_contact.png",
            "tag": "Agentic Vision Trace B · Articular Surface Contact Rule (6 / 7)",
            "headline": "Distinguishing Operative Grade III Meniscal Tears from Benign Internal Mucin (99.4% Contact)",
            "caption": "3px Contour Perimeter Ring Intersected with cv2.Scharr Cortical Margin -> Escalates to Arthroscopic Consult",
            "narration": (
                "In Case Four, to distinguish an operative Grade Three lateral meniscus tear from benign Grade Two internal mucin, "
                "the agent intersects a three-pixel contour perimeter ring against the Open C V Scharr cortical boundary map. "
                "Measuring ninety-nine point four percent articular surface contact, the agent confirms surface extension and escalates for arthroscopic repair."
            ),
        },
        {
            "img": IMG_DIR / "06_volumetric_scrubber_and_aws_cool_telemetry.png",
            "tag": "16-Slice 3D Volume Scrubber & AWS Graviton COOL Benchmark (7 / 7)",
            "headline": "1,236 Slices/Sec on AWS Graviton COOL (1.79x Speedup, 52% Lower Cloud Cost)",
            "caption": "Validated on 2,668 RSNA Studies (0.957 Public LB | 0.8568 3-Radiologist Gold Consensus Panel)",
            "narration": (
                "In the telemetry drawer below, doctors can scrub across all sixteen three-D volume slices weighted by neural attention, "
                "or run live A W S Graviton COOL benchmarks achieving one thousand two hundred thirty-six slices per second—a one point seven nine times speedup "
                "at fifty-two percent lower cloud cost. Validated at zero point nine five seven Macro A U C on the R S N A leaderboard, "
                "Creaky C V 5 brings fast, responsible, zero-egress Open C V 5 active perception to clinical M R I triage. Thank you."
            ),
        },
    ]

    work_dir = Path(tempfile.mkdtemp(prefix="creaky_video_build_"))
    segment_mp4s: List[Path] = []
    try:
        for idx, sc in enumerate(scenes, start=1):
            wav_path = work_dir / f"scene_{idx:02d}.wav"
            dur = synthesize_wav_sapi(sc["narration"], wav_path)
            print(f"[Video Scene {idx}/{len(scenes)}] Voiceover synthesized ({dur:.1f}s)")

            raw_bgr = cv2.imread(str(sc["img"]), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise RuntimeError(f"Missing image {sc['img']}")
            framed_bgr = add_lower_third_banner(
                raw_bgr,
                sc["tag"],
                sc["headline"],
                sc["caption"],
                idx,
                len(scenes),
            )
            frame_png = work_dir / f"scene_{idx:02d}.png"
            cv2.imwrite(str(frame_png), framed_bgr)

            seg_mp4 = work_dir / f"seg_{idx:02d}.mp4"
            # Encode segment with subtle zoompan or crisp static 1080p + AAC audio
            subprocess.run(
                [
                    str(FFMPEG_EXE),
                    "-y",
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(frame_png),
                    "-i",
                    str(wav_path),
                    "-c:v",
                    "libx264",
                    "-tune",
                    "stillimage",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "scale=1920:1080",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(seg_mp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            segment_mp4s.append(seg_mp4)

        concat_txt = work_dir / "concat.txt"
        concat_txt.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segment_mp4s),
            encoding="utf-8",
        )
        out_mp4 = KIT_DIR / "creaky_cv5_demo_video.mp4"
        subprocess.run(
            [
                str(FFMPEG_EXE),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c",
                "copy",
                str(out_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Built full 1080p narrated demo video: {out_mp4} ({out_mp4.stat().st_size / (1024*1024):.2f} MB)")
        return out_mp4
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(capture_all_screenshots())
    build_demo_video()
