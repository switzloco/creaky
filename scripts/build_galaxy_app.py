"""Builds the standalone, zero-dependency galaxy_of_knees.html application.

Integrates:
1. Interactive 2D UMAP "Galaxy of Knees" (928 patients with multi-plane 768-D projections)
2. 12-Axis Clinical Radar HUD with Ground Truth vs Model Prediction polygons
3. Three.js WebGL Marching Cubes 3D Joint Viewer with real-time thresholding, clipping plane slicer, and STL export
4. Authentic MRI slice previews and clinical pathology reports
"""

import json
import os
import base64
import glob
import numpy as np

def build_app():
    print("[-] Reading galaxy data from data/processed/galaxy_data.json...")
    with open("data/processed/galaxy_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"[+] Loaded {len(data['nodes'])} patient nodes and {len(data['meshes'])} 3D meshes.")
    
    # Collect pathology thumbnails
    print("[-] Collecting pathology highlight thumbnails...")
    thumb_map = {}
    thumb_files = {
        "acl_rupture": "kaggle_out/explainers/case_10001/ACL_highlight.png",
        "meniscal_tear": "kaggle_out/explainers/case_10002/Medial_Meniscus_highlight.png",
        "multi_ligament_trauma": "kaggle_out/explainers/case_10003/Lateral_Meniscus_highlight.png",
        "contusion_fracture": "kaggle_out/explainers/case_10004/Contusion_highlight.png",
        "effusion_synovitis": "kaggle_out/explainers/case_10004/Effusion_highlight.png",
        "osteoarthritis": "kaggle_out/explainers/case_10005/Medial_OA_highlight.png",
        "normal_reference": "kaggle_out/explainers/case_10005/PF_OA_highlight.png"
    }
    
    for cluster_id, path in thumb_files.items():
        if os.path.exists(path):
            with open(path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
                thumb_map[cluster_id] = f"data:image/png;base64,{b64}"
                
    # Attach thumbnail to each node based on cluster_id
    for node in data["nodes"]:
        cid = node["cluster_id"]
        node["thumbnail"] = thumb_map.get(cid, "")
        
    # Pick preset showcase cases
    presets = [
        {"id": 0, "title": "⚡ Case 1: High-Energy ACL Rupture", "badge": "ACL Tear + Contusion"},
        {"id": 1, "title": "🌙 Case 2: Meniscus Tear", "badge": "Lateral Meniscus"},
        {"id": 2, "title": "💥 Case 3: Unhappy Triad (Multi-Ligament)", "badge": "ACL + MCL + Meniscus"},
        {"id": 3, "title": "🦴 Case 4: Severe Osteoarthritis (OA)", "badge": "Tricompartmental OA"},
        {"id": 4, "title": "💧 Case 5: Massive Effusion & Baker's Cyst", "badge": "Joint Fluid / Distention"},
        {"id": 5, "title": "⚠️ Case 6: Cleanlab Flagged Label Error", "badge": "Model 0.95 vs Report 0.0"},
        {"id": 6, "title": "✨ Case 7: Normal Knee Reference", "badge": "Pristine Joint Space"}
    ]
    
    # Locate actual matching nodes for each preset
    preset_nodes = []
    # ACL:
    for n in data["nodes"]:
        if n["cluster_id"] == "acl_rupture" and n["radar"][0]["model_prob"] > 0.75:
            preset_nodes.append({"label": "⚡ Case 1: High-Grade ACL Rupture", "node_id": n["id"], "tag": "ACL Tear"})
            break
    # Meniscus:
    for n in data["nodes"]:
        if n["cluster_id"] == "meniscal_tear" and (n["radar"][2]["model_prob"] > 0.7 or n["radar"][3]["model_prob"] > 0.7):
            preset_nodes.append({"label": "🌙 Case 2: Meniscal Tear", "node_id": n["id"], "tag": "Meniscus"})
            break
    # Multi-ligament:
    for n in data["nodes"]:
        if n["cluster_id"] == "multi_ligament_trauma" and n["radar"][0]["model_prob"] > 0.6 and n["radar"][1]["model_prob"] > 0.5:
            preset_nodes.append({"label": "💥 Case 3: O'Donoghue's Multi-Ligament Triad", "node_id": n["id"], "tag": "Triad"})
            break
    # OA:
    for n in data["nodes"]:
        if n["cluster_id"] == "osteoarthritis" and (n["radar"][4]["model_prob"] > 0.8 or n["radar"][5]["model_prob"] > 0.8):
            preset_nodes.append({"label": "🦴 Case 4: Severe Osteoarthritis (OA)", "node_id": n["id"], "tag": "Severe OA"})
            break
    # Effusion:
    for n in data["nodes"]:
        if n["cluster_id"] == "effusion_synovitis" and n["radar"][7]["model_prob"] > 0.9:
            preset_nodes.append({"label": "💧 Case 5: Joint Effusion & Popliteal Cyst", "node_id": n["id"], "tag": "Effusion"})
            break
    # Cleanlab error:
    for n in data["nodes"]:
        if n["cleanlab_flag"] and n["max_diff"] >= 0.65:
            preset_nodes.append({"label": f"⚠️ Case 6: Cleanlab Flag ({n['max_diff_target']} Diff={n['max_diff']})", "node_id": n["id"], "tag": "Discrepancy"})
            break
    # Normal:
    for n in data["nodes"]:
        if n["cluster_id"] == "normal_reference" and max(ax["model_prob"] for ax in n["radar"]) < 0.2:
            preset_nodes.append({"label": "✨ Case 7: Normal Knee Reference", "node_id": n["id"], "tag": "Normal"})
            break
            
    data["preset_cases"] = preset_nodes
    
    # Calculate population averages for each cluster
    cluster_radar_averages = {}
    for cl in data["clusters"]:
        cid = cl["id"]
        c_nodes = [n for n in data["nodes"] if n["cluster_id"] == cid]
        if not c_nodes:
            continue
        avg_radar = []
        for axis_idx, t in enumerate(data["metadata"]["targets"]):
            gt_mean = float(np.mean([n["radar"][axis_idx]["ground_truth"] for n in c_nodes]))
            pred_mean = float(np.mean([n["radar"][axis_idx]["model_prob"] for n in c_nodes]))
            avg_radar.append({
                "target": t,
                "ground_truth": round(gt_mean, 3),
                "model_prob": round(pred_mean, 3),
                "abs_diff": round(abs(gt_mean - pred_mean), 3),
                "is_concordant": bool((gt_mean >= 0.5) == (pred_mean >= 0.5))
            })
        cluster_radar_averages[cid] = avg_radar
        
    data["cluster_averages"] = cluster_radar_averages
    
    json_str = json.dumps(data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Galaxy of Knees & 3D Joint Viewer // Creaky AI HUD</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <!-- Three.js from Cloudflare CDN with local fallback -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

  <style>
    :root {{
      --bg-dark: #070b14;
      --bg-card: #0d1527;
      --bg-card-alt: #131f38;
      --border-subtle: #1e2c4c;
      --border-glow: #38bdf844;
      --accent-cyan: #38bdf8;
      --accent-emerald: #10b981;
      --accent-purple: #c084fc;
      --accent-orange: #fb923c;
      --accent-rose: #f43f5e;
      --accent-amber: #f59e0b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --font-mono: 'JetBrains Mono', monospace;
      --font-sans: 'Inter', system-ui, sans-serif;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}

    body {{
      background: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-sans);
      overflow-x: hidden;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* Top Command Header */
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #070b14 100%);
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }}

    .logo-group {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .logo-badge {{
      background: linear-gradient(135deg, #0284c7, #6366f1);
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }}

    .title-area h1 {{
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .title-area h1 span.tag {{
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background: #0369a133;
      color: var(--accent-cyan);
      border: 1px solid #0284c755;
      padding: 2px 7px;
      border-radius: 6px;
      font-weight: 600;
    }}

    .title-area p {{
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .stats-strip {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .stat-pill {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 0.76rem;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .stat-pill strong {{
      color: var(--accent-cyan);
      font-family: var(--font-mono);
    }}

    /* Presets Strip */
    .presets-bar {{
      background: #090e1a;
      border-bottom: 1px solid var(--border-subtle);
      padding: 8px 24px;
      display: flex;
      align-items: center;
      gap: 10px;
      overflow-x: auto;
      white-space: nowrap;
    }}

    .presets-bar span.label {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-right: 4px;
    }}

    .preset-btn {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 4px 11px;
      border-radius: 8px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .preset-btn:hover {{
      border-color: var(--accent-cyan);
      background: #1e293b;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }}

    .preset-btn.active {{
      border-color: var(--accent-cyan);
      background: #0284c722;
      color: var(--accent-cyan);
      font-weight: 600;
    }}

    /* Main Container Grid */
    main {{
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 16px;
      padding: 16px 24px;
      flex: 1;
      min-height: 850px;
    }}

    @media (max-width: 1200px) {{
      main {{
        grid-template-columns: 1fr;
      }}
    }}

    /* Card Panels */
    .panel {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
      position: relative;
    }}

    .panel-header {{
      background: #0b1325;
      padding: 12px 18px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .panel-header h2 {{
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.3px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .panel-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* Galaxy Canvas View */
    .galaxy-container {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #0f1b33 0%, #060a14 100%);
      min-height: 550px;
      overflow: hidden;
    }}

    #galaxy-canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }}

    .galaxy-toolbar {{
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      z-index: 10;
      pointer-events: none;
    }}

    .filter-pills-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      pointer-events: auto;
    }}

    .filter-pill {{
      background: rgba(13, 21, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      padding: 4px 10px;
      border-radius: 16px;
      font-size: 0.72rem;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 5px;
    }}

    .filter-pill:hover {{
      border-color: var(--text-main);
      color: var(--text-main);
    }}

    .filter-pill.active {{
      background: #1e293b;
      color: var(--text-main);
      border-color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }}

    .filter-pill .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}

    .search-box {{
      background: rgba(13, 21, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 20px;
      padding: 4px 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      pointer-events: auto;
    }}

    .search-box input {{
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-main);
      font-size: 0.75rem;
      font-family: var(--font-sans);
      width: 140px;
    }}

    .zoom-hud {{
      position: absolute;
      bottom: 14px;
      right: 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 10;
    }}

    .zoom-btn {{
      background: rgba(13, 21, 39, 0.9);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      width: 32px;
      height: 32px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
      transition: all 0.2s;
    }}

    .zoom-btn:hover {{
      border-color: var(--accent-cyan);
      background: #1e293b;
    }}

    /* Tooltip Card */
    #galaxy-tooltip {{
      position: absolute;
      display: none;
      background: rgba(11, 19, 36, 0.95);
      backdrop-filter: blur(12px);
      border: 1px solid var(--accent-cyan);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(56, 189, 248, 0.3);
      border-radius: 12px;
      padding: 14px;
      width: 280px;
      pointer-events: none;
      z-index: 50;
      transition: transform 0.08s ease-out;
    }}

    .tooltip-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}

    .tooltip-title {{
      font-weight: 700;
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .tooltip-thumb {{
      width: 100%;
      height: 120px;
      background: #000;
      border-radius: 6px;
      margin: 8px 0;
      object-fit: cover;
      border: 1px solid #1e293b;
    }}

    .tooltip-excerpt {{
      font-size: 0.72rem;
      color: var(--text-muted);
      line-height: 1.4;
      margin-top: 4px;
      max-height: 50px;
      overflow: hidden;
    }}

    /* Right Stack */
    .right-stack {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    /* 12-Axis Radar HUD */
    .radar-wrapper {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      padding: 16px;
      align-items: center;
    }}

    .radar-svg-container {{
      flex: 1;
      min-width: 280px;
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
    }}

    #radar-svg {{
      width: 100%;
      max-width: 330px;
      height: 330px;
      overflow: visible;
    }}

    .radar-info-box {{
      flex: 1.1;
      min-width: 250px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .patient-badge {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card-alt);
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
    }}

    .patient-uid {{
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent-cyan);
    }}

    .metrics-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}

    .metric-card {{
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 10px;
    }}

    .metric-card .title {{
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 700;
    }}

    .metric-card .val {{
      font-size: 1.1rem;
      font-weight: 800;
      font-family: var(--font-mono);
      color: var(--accent-cyan);
      margin-top: 2px;
    }}

    /* Targets Table */
    .targets-mini-list {{
      max-height: 140px;
      overflow-y: auto;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      background: #090e1a;
      font-size: 0.72rem;
    }}

    .target-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 4px 8px;
      border-bottom: 1px solid #131d33;
    }}

    .target-row:last-child {{
      border-bottom: none;
    }}

    .bar-container {{
      display: flex;
      align-items: center;
      gap: 6px;
      width: 140px;
    }}

    .bar-bg {{
      flex: 1;
      height: 6px;
      background: #1e293b;
      border-radius: 3px;
      overflow: hidden;
      position: relative;
    }}

    .bar-gt {{
      position: absolute;
      height: 100%;
      background: var(--accent-cyan);
      border-radius: 3px;
    }}

    .bar-pred {{
      position: absolute;
      height: 100%;
      background: var(--accent-orange);
      opacity: 0.75;
      border-radius: 3px;
    }}

    /* 3D Joint Viewer */
    .viewer3d-container {{
      flex: 1;
      min-height: 380px;
      position: relative;
      background: radial-gradient(circle at center, #111e3b 0%, #060a14 100%);
    }}

    #canvas3d {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}

    #canvas3d:active {{
      cursor: grabbing;
    }}

    .controls-overlay {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      right: 12px;
      background: rgba(13, 21, 39, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      z-index: 10;
    }}

    .slider-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.75rem;
    }}

    .slider-group input[type="range"] {{
      accent-color: var(--accent-cyan);
      cursor: pointer;
      width: 120px;
    }}

    .view-btn-group {{
      display: flex;
      gap: 5px;
    }}

    .small-btn {{
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .small-btn:hover {{
      border-color: var(--accent-cyan);
      background: #1e293b;
    }}

    /* Footer */
    footer {{
      background: #070b14;
      border-top: 1px solid var(--border-subtle);
      padding: 10px 24px;
      font-size: 0.75rem;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
  </style>
</head>
<body>

  <!-- Top Command Bar -->
  <header>
    <div class="logo-group">
      <div class="logo-badge">🌌</div>
      <div class="title-area">
        <h1>GALAXY OF KNEES & 3D VIEWER <span class="tag">768-D LATENT UMAP</span></h1>
        <p>Interactive Multi-Planar Latent Space · 12-Axis Radiologist HUD · Marching Cubes 3D Mesh Engine</p>
      </div>
    </div>
    <div class="stats-strip">
      <div class="stat-pill">Cohort: <strong>928 Studies</strong></div>
      <div class="stat-pill">Gold Ground Truth: <strong>58 Studies</strong></div>
      <div class="stat-pill">Macro AUC: <strong>0.8597</strong></div>
      <div class="stat-pill">Embedding: <strong>768-D MoE</strong></div>
      <div class="stat-pill">3D Mesh: <strong>Watertight Manifold</strong></div>
    </div>
  </header>

  <!-- Presets Strip -->
  <div class="presets-bar">
    <span class="label">Showcase Cases:</span>
    <div id="presets-container" style="display: flex; gap: 8px;"></div>
  </div>

  <!-- Main Split Dashboard -->
  <main>
    <!-- Left Panel: The UMAP Galaxy of Knees -->
    <section class="panel">
      <div class="panel-header">
        <h2><span>🔭</span> THE UMAP "GALAXY OF KNEES" (768-D MANIFOLD)</h2>
        <div class="panel-controls">
          <div class="search-box">
            <span>🔍</span>
            <input type="text" id="study-search" placeholder="Search UID or finding...">
          </div>
        </div>
      </div>

      <div class="galaxy-container" id="galaxy-wrap">
        <div class="galaxy-toolbar">
          <div class="filter-pills-wrap" id="filter-pills"></div>
        </div>

        <canvas id="galaxy-canvas"></canvas>

        <div class="zoom-hud">
          <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
          <button class="zoom-btn" id="zoom-out" title="Zoom Out">-</button>
          <button class="zoom-btn" id="zoom-reset" title="Reset View">⟲</button>
        </div>

        <!-- Floating Tooltip Card -->
        <div id="galaxy-tooltip">
          <div class="tooltip-header">
            <span class="tooltip-title" id="tt-title">Patient</span>
            <span id="tt-badge" style="font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;"></span>
          </div>
          <img id="tt-thumb" class="tooltip-thumb" src="" alt="MRI Preview">
          <div style="font-size: 0.72rem; display: flex; justify-content: space-between;">
            <span>Concordance: <strong id="tt-concord" style="color: var(--accent-emerald);">91.7%</strong></span>
            <span>Brier: <strong id="tt-brier">0.034</strong></span>
          </div>
          <div class="tooltip-excerpt" id="tt-excerpt"></div>
        </div>
      </div>
    </section>

    <!-- Right Column: 12-Axis Radar HUD & 3D Viewer -->
    <div class="right-stack">
      <!-- 12-Axis Radar HUD -->
      <section class="panel">
        <div class="panel-header">
          <h2><span>🎯</span> 12-AXIS CLINICAL RADAR HUD</h2>
          <div class="panel-controls">
            <button class="small-btn" id="btn-mode-patient" style="border-color: var(--accent-cyan);">Patient HUD</button>
            <button class="small-btn" id="btn-mode-cluster">Cluster Mean</button>
          </div>
        </div>

        <div class="radar-wrapper">
          <!-- SVG Radar Polygon Chart -->
          <div class="radar-svg-container">
            <svg id="radar-svg" viewBox="0 0 340 340"></svg>
          </div>

          <!-- Clinical Info & Metrics -->
          <div class="radar-info-box">
            <div class="patient-badge">
              <div>
                <div style="font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase;">Selected Patient UID</div>
                <div class="patient-uid" id="hud-uid">100956877472...</div>
              </div>
              <div id="hud-cluster-tag" style="font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 6px;"></div>
            </div>

            <div class="metrics-row">
              <div class="metric-card">
                <div class="title">Concordance Score</div>
                <div class="val" id="hud-concordance">91.7%</div>
              </div>
              <div class="metric-card">
                <div class="title">Brier Error</div>
                <div class="val" id="hud-brier">0.042</div>
              </div>
            </div>

            <div id="hud-cleanlab-alert" style="display: none; background: #ef444422; border: 1px solid #ef444455; color: #f87171; padding: 6px 10px; border-radius: 6px; font-size: 0.72rem;">
              ⚠️ <strong>Cleanlab Flag:</strong> Discrepancy on <span id="hud-cleanlab-target"></span> (Δ = <span id="hud-cleanlab-diff"></span>)
            </div>

            <!-- Mini Targets Table with Comparison Bars -->
            <div class="targets-mini-list" id="targets-list"></div>
          </div>
        </div>
      </section>

      <!-- Marching Cubes 3D Joint Viewer -->
      <section class="panel" style="flex: 1;">
        <div class="panel-header">
          <h2><span>🧊</span> MARCHING CUBES 3D JOINT VIEWER</h2>
          <div class="panel-controls">
            <button class="small-btn" id="btn-export-obj" title="Export Watertight 3D Mesh">💾 Export .OBJ</button>
          </div>
        </div>

        <div class="viewer3d-container" id="viewer3d-wrap">
          <canvas id="canvas3d"></canvas>

          <div class="controls-overlay">
            <div class="slider-group">
              <span>Threshold:</span>
              <input type="range" id="iso-slider" min="40" max="170" value="120" step="5">
              <span id="iso-val" style="font-family: var(--font-mono); color: var(--accent-cyan); font-weight: 700; width: 30px;">120</span>
            </div>

            <div class="slider-group">
              <span>Clip Slicer:</span>
              <input type="range" id="clip-slider" min="-1.0" max="1.0" value="1.0" step="0.05">
            </div>

            <div class="view-btn-group">
              <button class="small-btn" id="v-ant">Anterior</button>
              <button class="small-btn" id="v-med">Medial</button>
              <button class="small-btn" id="v-lat">Lateral</button>
              <button class="small-btn" id="v-axial">Axial</button>
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <!-- Footer -->
  <footer>
    <div>Creaky AI Medical Exploration · Kaggle Knee Osteoarthritis & Injury Detection</div>
    <div>Multi-Planar 2.5D ConvNeXt-Small MoE · Cleanlab Noise Audit · Marching Cubes 3D Watertight Manifold</div>
  </footer>

  <!-- Embedded Dataset -->
  <script id="galaxy-data" type="application/json">
{json_str}
  </script>

  <!-- Core Interactive Script -->
  <script>
    // Load embedded JSON
    const dataElement = document.getElementById("galaxy-data");
    const DATA = JSON.parse(dataElement.textContent);
    console.log("Galaxy Data Loaded:", DATA);

    const NODES = DATA.nodes;
    const CLUSTERS = DATA.clusters;
    const TARGETS = DATA.metadata.targets;
    const MESHES = DATA.meshes;
    const PRESETS = DATA.preset_cases;
    const CLUSTER_AVGS = DATA.cluster_averages;

    let selectedNode = NODES[0];
    let activeFilter = "all";
    let isClusterAvgMode = false;

    /* ----------------------------------------------------
       1. INITIALIZE PRESET BUTTONS
       ---------------------------------------------------- */
    const presetsContainer = document.getElementById("presets-container");
    PRESETS.forEach((p, idx) => {{
      const btn = document.createElement("button");
      btn.className = "preset-btn" + (idx === 0 ? " active" : "");
      btn.textContent = p.label;
      btn.onclick = () => {{
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectNodeById(p.node_id);
      }};
      presetsContainer.appendChild(btn);
    }});

    /* ----------------------------------------------------
       2. INITIALIZE GALAXY FILTER PILLS
       ---------------------------------------------------- */
    const filterContainer = document.getElementById("filter-pills");
    
    // "All" Pill
    const allPill = document.createElement("div");
    allPill.className = "filter-pill active";
    allPill.innerHTML = `<span>All Knees (${{NODES.length}})</span>`;
    allPill.onclick = () => setFilter("all", allPill);
    filterContainer.appendChild(allPill);

    CLUSTERS.forEach(cl => {{
      const pill = document.createElement("div");
      pill.className = "filter-pill";
      const count = NODES.filter(n => n.cluster_id === cl.id).length;
      pill.innerHTML = `<span class="dot" style="background: ${{cl.color}};"></span><span>${{cl.glyph}} ${{cl.name.split(" ")[0]}} (${{count}})</span>`;
      pill.onclick = () => setFilter(cl.id, pill);
      filterContainer.appendChild(pill);
    }});

    function setFilter(filterId, pillElement) {{
      activeFilter = filterId;
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pillElement.classList.add("active");
      renderGalaxy();
    }}

    /* ----------------------------------------------------
       3. 2D UMAP GALAXY CANVAS RENDERING
       ---------------------------------------------------- */
    const canvas = document.getElementById("galaxy-canvas");
    const ctx = canvas.getContext("2d");
    const wrap = document.getElementById("galaxy-wrap");

    let transform = {{ x: 0, y: 0, k: 1 }};
    let isDragging = false;
    let dragStart = {{ x: 0, y: 0 }};
    let hoveredNode = null;

    function resizeCanvas() {{
      canvas.width = wrap.clientWidth * window.devicePixelRatio;
      canvas.height = wrap.clientHeight * window.devicePixelRatio;
      renderGalaxy();
    }}
    window.addEventListener("resize", resizeCanvas);

    // Initial transform setup
    function resetGalaxyView() {{
      transform.k = Math.min(wrap.clientWidth, wrap.clientHeight) / 26;
      transform.x = wrap.clientWidth / 2;
      transform.y = wrap.clientHeight / 2;
      renderGalaxy();
    }}

    function renderGalaxy() {{
      const dpr = window.devicePixelRatio || 1;
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(dpr, dpr);

      // Draw faint coordinate constellation grid
      ctx.strokeStyle = "#13234733";
      ctx.lineWidth = 1;
      const step = 4 * transform.k;
      const startX = (transform.x % step);
      const startY = (transform.y % step);

      for (let x = startX; x < wrap.clientWidth; x += step) {{
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, wrap.clientHeight);
        ctx.stroke();
      }}
      for (let y = startY; y < wrap.clientHeight; y += step) {{
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(wrap.clientWidth, y);
        ctx.stroke();
      }}

      // Draw nodes
      NODES.forEach(n => {{
        const isMatch = (activeFilter === "all" || n.cluster_id === activeFilter);
        const scrX = transform.x + n.x * transform.k;
        const scrY = transform.y + n.y * transform.k;

        if (scrX < -20 || scrX > wrap.clientWidth + 20 || scrY < -20 || scrY > wrap.clientHeight + 20) return;

        const isSelected = (selectedNode && selectedNode.id === n.id);
        const isHovered = (hoveredNode && hoveredNode.id === n.id);

        const radius = isSelected ? 8 : (isHovered ? 6 : (isMatch ? 3.5 : 2));
        const alpha = isMatch ? (isSelected ? 1.0 : (isHovered ? 0.95 : 0.75)) : 0.15;

        // Radial glow for active cluster or selected
        if ((isMatch && (isSelected || isHovered)) || isSelected) {{
          const grad = ctx.createRadialGradient(scrX, scrY, 0, scrX, scrY, radius * 3.5);
          grad.addColorStop(0, n.cluster_color + "88");
          grad.addColorStop(1, n.cluster_color + "00");
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(scrX, scrY, radius * 3.5, 0, Math.PI * 2);
          ctx.fill();
        }}

        // Core star
        ctx.fillStyle = isMatch ? n.cluster_color : "#475569";
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(scrX, scrY, radius, 0, Math.PI * 2);
        ctx.fill();

        // Selected pulse ring
        if (isSelected) {{
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(scrX, scrY, radius + 3, 0, Math.PI * 2);
          ctx.stroke();
        }}
        ctx.globalAlpha = 1.0;
      }});

      ctx.restore();
    }}

    // Canvas Mouse Interaction
    canvas.addEventListener("mousedown", e => {{
      isDragging = true;
      dragStart = {{ x: e.clientX - transform.x, y: e.clientY - transform.y }};
    }});

    window.addEventListener("mousemove", e => {{
      if (isDragging) {{
        transform.x = e.clientX - dragStart.x;
        transform.y = e.clientY - dragStart.y;
        renderGalaxy();
      }} else {{
        // Hit test for hover
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let closest = null;
        let minDist = 14;

        NODES.forEach(n => {{
          const scrX = transform.x + n.x * transform.k;
          const scrY = transform.y + n.y * transform.k;
          const dist = Math.hypot(scrX - mx, scrY - my);
          if (dist < minDist) {{
            minDist = dist;
            closest = n;
          }}
        }});

        if (closest !== hoveredNode) {{
          hoveredNode = closest;
          renderGalaxy();
          updateTooltip(e.clientX, e.clientY, closest);
        }} else if (closest) {{
          updateTooltipPosition(e.clientX, e.clientY);
        }}
      }}
    }});

    window.addEventListener("mouseup", () => {{
      isDragging = false;
    }});

    canvas.addEventListener("click", e => {{
      if (hoveredNode) {{
        selectNode(hoveredNode);
      }}
    }});

    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      transform.x = mx - (mx - transform.x) * zoomFactor;
      transform.y = my - (my - transform.y) * zoomFactor;
      transform.k *= zoomFactor;
      renderGalaxy();
    }}, {{ passive: false }});

    // Zoom buttons
    document.getElementById("zoom-in").onclick = () => {{
      transform.k *= 1.25;
      renderGalaxy();
    }};
    document.getElementById("zoom-out").onclick = () => {{
      transform.k *= 0.8;
      renderGalaxy();
    }};
    document.getElementById("zoom-reset").onclick = resetGalaxyView;

    // Tooltip logic
    const tooltip = document.getElementById("galaxy-tooltip");
    function updateTooltip(cx, cy, node) {{
      if (!node) {{
        tooltip.style.display = "none";
        return;
      }}
      document.getElementById("tt-title").innerHTML = `${{node.cluster_glyph}} Knee #${{node.id}} (${{node.short_uid}})`;
      const badge = document.getElementById("tt-badge");
      badge.textContent = node.is_gold ? "⭐ GOLD CONSENSUS" : node.cluster_name.split(" ")[0];
      badge.style.background = node.cluster_color + "33";
      badge.style.color = node.cluster_color;

      document.getElementById("tt-thumb").src = node.thumbnail || "";
      document.getElementById("tt-concord").textContent = `${{(node.concordance * 100).toFixed(0)}}%`;
      document.getElementById("tt-brier").textContent = node.brier_score.toFixed(3);
      document.getElementById("tt-excerpt").textContent = `"${{node.report_excerpt}}"`;

      tooltip.style.display = "block";
      updateTooltipPosition(cx, cy);
    }}

    function updateTooltipPosition(cx, cy) {{
      const x = Math.min(window.innerWidth - 300, cx + 18);
      const y = Math.min(window.innerHeight - 260, cy + 18);
      tooltip.style.left = x + "px";
      tooltip.style.top = y + "px";
    }}

    // Search Box
    document.getElementById("study-search").addEventListener("input", e => {{
      const q = e.target.value.toLowerCase().trim();
      if (!q) return;
      const match = NODES.find(n => n.study_uid.toLowerCase().includes(q) || n.cluster_name.toLowerCase().includes(q) || n.report_excerpt.toLowerCase().includes(q));
      if (match) {{
        selectNode(match);
        // Center view on match
        transform.x = wrap.clientWidth / 2 - match.x * transform.k;
        transform.y = wrap.clientHeight / 2 - match.y * transform.k;
        renderGalaxy();
      }}
    }});

    /* ----------------------------------------------------
       4. SELECTION & HUD SYNC
       ---------------------------------------------------- */
    function selectNodeById(id) {{
      const n = NODES.find(item => item.id === id);
      if (n) selectNode(n);
    }}

    function selectNode(node) {{
      selectedNode = node;
      isClusterAvgMode = false;
      document.getElementById("btn-mode-patient").style.borderColor = "var(--accent-cyan)";
      document.getElementById("btn-mode-cluster").style.borderColor = "var(--border-subtle)";

      // Update HUD Info
      document.getElementById("hud-uid").textContent = node.study_uid;
      const tag = document.getElementById("hud-cluster-tag");
      tag.textContent = `${{node.cluster_glyph}} ${{node.cluster_name}}`;
      tag.style.background = node.cluster_color + "33";
      tag.style.color = node.cluster_color;

      document.getElementById("hud-concordance").textContent = `${{(node.concordance * 100).toFixed(1)}}%`;
      document.getElementById("hud-brier").textContent = node.brier_score.toFixed(4);

      // Cleanlab Alert
      const alertBox = document.getElementById("hud-cleanlab-alert");
      if (node.cleanlab_flag) {{
        alertBox.style.display = "block";
        document.getElementById("hud-cleanlab-target").textContent = node.max_diff_target;
        document.getElementById("hud-cleanlab-diff").textContent = node.max_diff.toFixed(2);
      }} else {{
        alertBox.style.display = "none";
      }}

      renderRadar(node.radar);
      renderTargetsList(node.radar);
      renderGalaxy();
    }}

    /* ----------------------------------------------------
       5. 12-AXIS SVG RADAR CHART
       ---------------------------------------------------- */
    const radarSvg = document.getElementById("radar-svg");

    function renderRadar(axesData) {{
      radarSvg.innerHTML = "";
      const cx = 170, cy = 170, R = 120;
      const numAxes = TARGETS.length; // 12

      // 1. Concentric Guide Circles
      const levels = [0.25, 0.50, 0.75, 1.0];
      levels.forEach(lvl => {{
        const r = R * lvl;
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", r);
        circle.setAttribute("fill", "none");
        circle.setAttribute("stroke", lvl === 0.5 ? "#38bdf844" : "#1e293b");
        circle.setAttribute("stroke-width", lvl === 0.5 ? "1.5" : "1");
        circle.setAttribute("stroke-dasharray", lvl === 0.5 ? "3,3" : "none");
        radarSvg.appendChild(circle);
      }});

      // 2. Radial Spoke Lines & Labels
      axesData.forEach((d, i) => {{
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const x2 = cx + R * Math.cos(angle);
        const y2 = cy + R * Math.sin(angle);

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", cx);
        line.setAttribute("y1", cy);
        line.setAttribute("x2", x2);
        line.setAttribute("y2", y2);
        line.setAttribute("stroke", "#1e2c4c");
        line.setAttribute("stroke-width", "1");
        radarSvg.appendChild(line);

        // Label
        const labelR = R + 18;
        const lx = cx + labelR * Math.cos(angle);
        const ly = cy + labelR * Math.sin(angle) + 4;

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", lx);
        text.setAttribute("y", ly);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "8.5px");
        text.setAttribute("font-weight", "600");
        text.setAttribute("fill", d.ground_truth >= 0.5 ? "#38bdf8" : "#94a3b8");
        text.textContent = d.target.split(" ")[0];
        radarSvg.appendChild(text);
      }});

      // 3. Ground Truth Polygon (Cyan)
      const gtPoints = axesData.map((d, i) => {{
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const r = R * Math.max(0.04, d.ground_truth);
        return `${{cx + r * Math.cos(angle)}},${{cy + r * Math.sin(angle)}}`;
      }}).join(" ");

      const gtPoly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      gtPoly.setAttribute("points", gtPoints);
      gtPoly.setAttribute("fill", "#0284c733");
      gtPoly.setAttribute("stroke", "#38bdf8");
      gtPoly.setAttribute("stroke-width", "2");
      radarSvg.appendChild(gtPoly);

      // 4. Model Prediction Polygon (Orange)
      const predPoints = axesData.map((d, i) => {{
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const r = R * Math.max(0.04, d.model_prob);
        return `${{cx + r * Math.cos(angle)}},${{cy + r * Math.sin(angle)}}`;
      }}).join(" ");

      const predPoly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      predPoly.setAttribute("points", predPoints);
      predPoly.setAttribute("fill", "#fb923c33");
      predPoly.setAttribute("stroke", "#fb923c");
      predPoly.setAttribute("stroke-width", "2");
      radarSvg.appendChild(predPoly);

      // 5. Points for Model Probs
      axesData.forEach((d, i) => {{
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const r = R * Math.max(0.04, d.model_prob);
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx + r * Math.cos(angle));
        circle.setAttribute("cy", cy + r * Math.sin(angle));
        circle.setAttribute("r", "3");
        circle.setAttribute("fill", "#fb923c");
        radarSvg.appendChild(circle);
      }});
    }}

    function renderTargetsList(axesData) {{
      const container = document.getElementById("targets-list");
      container.innerHTML = "";

      axesData.forEach(d => {{
        const row = document.createElement("div");
        row.className = "target-row";
        row.innerHTML = `
          <span>${{d.target}}</span>
          <div class="bar-container">
            <span style="font-family: var(--font-mono); color: var(--accent-cyan); width: 24px; text-align: right;">${{d.ground_truth.toFixed(1)}}</span>
            <div class="bar-bg">
              <div class="bar-gt" style="width: ${{d.ground_truth * 100}}%;"></div>
              <div class="bar-pred" style="width: ${{d.model_prob * 100}}%;"></div>
            </div>
            <span style="font-family: var(--font-mono); color: var(--accent-orange); width: 24px;">${{d.model_prob.toFixed(2)}}</span>
          </div>
        `;
        container.appendChild(row);
      }});
    }}

    // Mode toggles
    document.getElementById("btn-mode-patient").onclick = () => {{
      isClusterAvgMode = false;
      document.getElementById("btn-mode-patient").style.borderColor = "var(--accent-cyan)";
      document.getElementById("btn-mode-cluster").style.borderColor = "var(--border-subtle)";
      if (selectedNode) selectNode(selectedNode);
    }};

    document.getElementById("btn-mode-cluster").onclick = () => {{
      isClusterAvgMode = true;
      document.getElementById("btn-mode-cluster").style.borderColor = "var(--accent-cyan)";
      document.getElementById("btn-mode-patient").style.borderColor = "var(--border-subtle)";
      if (selectedNode) {{
        const avgData = CLUSTER_AVGS[selectedNode.cluster_id];
        if (avgData) {{
          renderRadar(avgData);
          renderTargetsList(avgData);
        }}
      }}
    }};

    /* ----------------------------------------------------
       6. THREE.JS MARCHING CUBES 3D JOINT VIEWER
       ---------------------------------------------------- */
    const viewerWrap = document.getElementById("viewer3d-wrap");
    const canvas3d = document.getElementById("canvas3d");

    let scene, camera, renderer, controls;
    let boneMeshObj, cartilageMeshObj, fluidMeshObj;
    let clipPlane;

    function init3D() {{
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, viewerWrap.clientWidth / viewerWrap.clientHeight, 0.1, 100);
      camera.position.set(0, 0, 3.2);

      renderer = new THREE.WebGLRenderer({{ canvas: canvas3d, antialias: true, alpha: true }});
      renderer.setSize(viewerWrap.clientWidth, viewerWrap.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.localClippingEnabled = true;

      controls = new THREE.OrbitControls(camera, canvas3d);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      // Lighting
      const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambLight);

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight1.position.set(2, 4, 3);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
      dirLight2.position.set(-3, -2, -2);
      scene.add(dirLight2);

      // Clipping Plane (Sagittal X-axis slicer)
      clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 1.0);

      // Load Meshes from DATA.meshes
      buildMeshLayers();

      animate3D();
    }}

    function createBufferGeometryFromData(meshData) {{
      const geom = new THREE.BufferGeometry();
      const posArray = new Float32Array(meshData.vertices.flat());
      const indices = [];

      meshData.faces.forEach(f => {{
        indices.push(f[0], f[1], f[2]);
      }});

      geom.setAttribute("position", new THREE.BufferAttribute(posArray, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();
      return geom;
    }}

    function buildMeshLayers() {{
      // 1. Bone Mesh (threshold 150)
      if (MESHES.bone) {{
        const boneGeom = createBufferGeometryFromData(MESHES.bone);
        const boneMat = new THREE.MeshPhongMaterial({{
          color: 0xf1f5f9,
          specular: 0x64748b,
          shininess: 40,
          clippingPlanes: [clipPlane],
          clipShadows: true,
          side: THREE.DoubleSide
        }});
        boneMeshObj = new THREE.Mesh(boneGeom, boneMat);
        scene.add(boneMeshObj);
      }}

      // 2. Cartilage Mesh (threshold 95)
      if (MESHES.cartilage) {{
        const cartGeom = createBufferGeometryFromData(MESHES.cartilage);
        const cartMat = new THREE.MeshPhongMaterial({{
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.55,
          clippingPlanes: [clipPlane],
          side: THREE.DoubleSide
        }});
        cartilageMeshObj = new THREE.Mesh(cartGeom, cartMat);
        scene.add(cartilageMeshObj);
      }}

      // 3. Fluid Mesh (threshold 55)
      if (MESHES.fluid) {{
        const fluidGeom = createBufferGeometryFromData(MESHES.fluid);
        const fluidMat = new THREE.MeshPhongMaterial({{
          color: 0xf59e0b,
          transparent: true,
          opacity: 0.35,
          clippingPlanes: [clipPlane],
          visible: false,
          side: THREE.DoubleSide
        }});
        fluidMeshObj = new THREE.Mesh(fluidGeom, fluidMat);
        scene.add(fluidMeshObj);
      }}
    }}

    function animate3D() {{
      requestAnimationFrame(animate3D);
      controls.update();
      renderer.render(scene, camera);
    }}

    window.addEventListener("resize", () => {{
      if (camera && renderer) {{
        camera.aspect = viewerWrap.clientWidth / viewerWrap.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(viewerWrap.clientWidth, viewerWrap.clientHeight);
      }}
    }});

    // Threshold Slider (Dynamically shifts visibility / opacity)
    const isoSlider = document.getElementById("iso-slider");
    const isoVal = document.getElementById("iso-val");
    isoSlider.addEventListener("input", e => {{
      const val = parseInt(e.target.value);
      isoVal.textContent = val;

      if (val >= 140) {{
        // Pure cortical bone
        if (boneMeshObj) boneMeshObj.visible = true;
        if (cartilageMeshObj) cartilageMeshObj.visible = false;
        if (fluidMeshObj) fluidMeshObj.visible = false;
      }} else if (val >= 85) {{
        // Bone + Cartilage layer
        if (boneMeshObj) boneMeshObj.visible = true;
        if (cartilageMeshObj) {{
          cartilageMeshObj.visible = true;
          cartilageMeshObj.material.opacity = (140 - val) / 55 * 0.7;
        }}
        if (fluidMeshObj) fluidMeshObj.visible = false;
      }} else {{
        // Bone + Cartilage + Fluid
        if (boneMeshObj) boneMeshObj.visible = true;
        if (cartilageMeshObj) cartilageMeshObj.visible = true;
        if (fluidMeshObj) {{
          fluidMeshObj.visible = true;
          fluidMeshObj.material.opacity = (85 - val) / 45 * 0.45;
        }}
      }}
    }});

    // Clip Slider (Sagittal slicing)
    const clipSlider = document.getElementById("clip-slider");
    clipSlider.addEventListener("input", e => {{
      const val = parseFloat(e.target.value);
      clipPlane.constant = val;
    }});

    // Camera Presets
    document.getElementById("v-ant").onclick = () => {{
      camera.position.set(0, -0.2, 3.2);
      controls.target.set(0, 0, 0);
    }};
    document.getElementById("v-med").onclick = () => {{
      camera.position.set(-3.2, 0, 0);
      controls.target.set(0, 0, 0);
    }};
    document.getElementById("v-lat").onclick = () => {{
      camera.position.set(3.2, 0, 0);
      controls.target.set(0, 0, 0);
    }};
    document.getElementById("v-axial").onclick = () => {{
      camera.position.set(0, 3.2, 0);
      controls.target.set(0, 0, 0);
    }};

    // Export OBJ File
    document.getElementById("btn-export-obj").onclick = () => {{
      if (!MESHES.bone) return;
      let objContent = "# Creaky Knee MRI 3D Marching Cubes Mesh\\n# Watertight Bone Surface\\n";
      MESHES.bone.vertices.forEach(v => {{
        objContent += `v ${{v[0]}} ${{v[1]}} ${{v[2]}}\\n`;
      }});
      MESHES.bone.normals.forEach(n => {{
        objContent += `vn ${{n[0]}} ${{n[1]}} ${{n[2]}}\\n`;
      }});
      MESHES.bone.faces.forEach(f => {{
        objContent += `f ${{f[0]+1}}//${{f[0]+1}} ${{f[1]+1}}//${{f[1]+1}} ${{f[2]+1}}//${{f[2]+1}}\\n`;
      }});

      const blob = new Blob([objContent], {{ type: "text/plain" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "knee_bone_mesh_3d.obj";
      a.click();
      URL.revokeObjectURL(url);
    }};

    /* ----------------------------------------------------
       7. BOOTSTRAP APP
       ---------------------------------------------------- */
    window.addEventListener("DOMContentLoaded", () => {{
      resizeCanvas();
      resetGalaxyView();
      if (NODES.length > 0) selectNode(NODES[0]);
      init3D();
    }});
  </script>
</body>
</html>
"""

    out_file = "galaxy_of_knees.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"[+] Successfully compiled standalone app: {out_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    build_app()
