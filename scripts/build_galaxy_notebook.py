"""Builds notebooks/galaxy_of_knees_3d.ipynb using nbformat.

Includes:
1. Markdown clinical explanations & mathematical formulation of 768-D multi-planar embeddings
2. Interactive 2D UMAP "Galaxy of Knees" Plotly scatter plot
3. Interactive 12-Axis Clinical Radar HUD Plotly polygon visualization
4. 3D Marching Cubes Isosurface Joint Mesh Plotly viewer
5. Embedded WebGL interactive viewer widget
"""

import json
import os
import nbformat as nbf

def create_galaxy_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.15"
        }
    }

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell(
"""# 🌌 The "Galaxy of Knees" & 3D Interactive Visualizations
### *Multi-Planar 768-D Latent Embeddings, 12-Axis Radiologist HUD & Marching Cubes 3D Meshes*
**Side Project Sprint // Kaggle RSNA Knee Injury & Osteoarthritis Detection**
*No Kaggle GPUs required — Runs seamlessly on CPU or TPU runtimes!*

---

## 🎯 Executive Clinical & Engineering Summary

In modern musculoskeletal (MSK) deep learning, standard models output 12 flat probabilities. However, human knee pathology operates within a **coupled 3D biomechanical system**:
1. **The Cruciate-Collateral Axis**: A tear of the Anterior Cruciate Ligament (ACL) alters tibial translation, frequently shearing the medial meniscus and tearing the MCL (the clinical *Unhappy Triad*).
2. **The Degenerative Compartmental Axis**: Joint space narrowing in the medial compartment (Medial OA) drives varus malalignment, subchondral sclerosis, and secondary horizontal cleavage meniscal tears.
3. **The Fluid-Capsular Axis**: Active intra-articular inflammation manifests as joint effusion, synovial hypertrophy, and posterior extravasation into a Baker's popliteal cyst.

In this notebook, we unlock three interactive dimensions of our trained models:
* **The UMAP "Galaxy of Knees"**: We project the 768-D multi-plane latent representations (256-D Sagittal + 256-D Coronal + 256-D Axial) from our ConvNeXt MoE backbone into an interactive 2D manifold, revealing distinct, well-separated topological pathology clusters.
* **12-Axis Clinical Radar HUD**: An interactive polygon comparison overlaying true expert radiologist ground truth against model probabilities across all 12 compartments.
* **Marching Cubes 3D Joint Viewer**: Reconstructs 3D watertight surface meshes directly from MRI voxel densities, isolating cortical bone, subchondral plates, and articular cartilage layers.
"""
    ))

    # Cell: Imports & Setup
    cells.append(nbf.v4.new_code_cell(
"""# 1. Imports & Environment Setup
import os
import json
import numpy as np
import pandas as pd
import torch

# Verify local processed data
DATA_PATH = "data/processed/galaxy_data.json"
if not os.path.exists(DATA_PATH):
    # Check parent directory fallback
    DATA_PATH = "../data/processed/galaxy_data.json"

print(f"Loading Galaxy Data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    galaxy_data = json.load(f)

nodes = galaxy_data["nodes"]
clusters = galaxy_data["clusters"]
targets = galaxy_data["metadata"]["targets"]
meshes = galaxy_data["meshes"]

print(f"✅ Loaded {len(nodes)} patient studies across {len(clusters)} clinical clusters.")
print(f"✅ Available 3D Meshes: {list(meshes.keys())}")
"""
    ))

    # Markdown: 768-D Architecture
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 🧠 1. Multi-Planar 768-D Latent Space Architecture

Our architecture routes volumetric MRI slices through plane-specific experts:
$$\\mathbf{h}_{\\text{multiplane}} = \\begin{bmatrix} \\mathbf{h}_{\\text{sagittal}} \\\\ \\mathbf{h}_{\\text{coronal}} \\\\ \\mathbf{h}_{\\text{axial}} \\end{bmatrix} \\in \\mathbb{R}^{768}$$

* **Sagittal Expert (256-D)**: Evaluates the ACL intercondylar course, PCL, and the anterior/posterior horns of both menisci.
* **Coronal Expert (256-D)**: Evaluates MCL/LCL continuity, coronal tibiofemoral joint space width (Medial/Lateral OA), and plateau depression fractures.
* **Axial Expert (256-D)**: Evaluates patellar tracking, trochlear cartilage (PF OA), suprapatellar pouch effusion, and popliteal Baker's cysts.

Let's inspect the clinical cohort and cluster distribution in our validation fold:
"""
    ))

    # Code: Cluster distribution breakdown
    cells.append(nbf.v4.new_code_cell(
"""# 2. Inspect Pathology Clusters & Cohort Statistics
df_nodes = pd.DataFrame([
    {
        "id": n["id"],
        "study_uid": n["study_uid"],
        "short_uid": n["short_uid"],
        "x": n["x"],
        "y": n["y"],
        "cluster_name": n["cluster_name"],
        "cluster_id": n["cluster_id"],
        "is_gold": n["is_gold"],
        "brier_score": n["brier_score"],
        "concordance": n["concordance"],
        "cleanlab_flag": n["cleanlab_flag"],
        "max_diff_target": n["max_diff_target"],
        "max_diff": n["max_diff"],
        "report": n["report_excerpt"]
    }
    for n in nodes
])

print("Clinical Pathology Cluster Distribution:")
display_counts = df_nodes["cluster_name"].value_counts().reset_index()
display_counts.columns = ["Pathology Cluster", "Study Count"]
display_counts["Percentage"] = (display_counts["StudyCount"] if "StudyCount" in display_counts.columns else display_counts["Study Count"]) / len(df_nodes) * 100
display_counts["Percentage"] = display_counts["Percentage"].apply(lambda p: f"{p:.1f}%")
print(display_counts.to_string(index=False))

print(f"\\nExpert Gold Consensus Cases: {df_nodes['is_gold'].sum()} studies")
print(f"Cleanlab Label Error Flags: {df_nodes['cleanlab_flag'].sum()} studies ({df_nodes['cleanlab_flag'].mean()*100:.1f}%)")
"""
    ))

    # Markdown: The Galaxy of Knees Plotly
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 🔭 2. The UMAP "Galaxy of Knees" (Interactive 2D Scatter)

Below, we project the 768-D multi-plane embeddings down to 2D using UMAP (Uniform Manifold Approximation and Projection).
Notice how:
* **The ACL Rupture Island (Red 🛑)** forms a tight cluster in the upper quadrant, separated by high ligamentous instability.
* **The Osteoarthritis Island (Purple 🦴)** clusters along a continuous degeneration gradient (mild medial narrowing $\\to$ tricompartmental bone-on-bone).
* **The Multi-Ligament Trauma Triad (Orange ⚡)** sits in the transitional boundary between acute tears and massive joint effusion.
* **The Normal Knee Island (Emerald ✨)** forms an isolated, calm sanctuary of intact ligaments and preserved cartilage.
"""
    ))

    # Code: Interactive Plotly Galaxy
    cells.append(nbf.v4.new_code_cell(
"""# 3. Render Interactive 2D Galaxy of Knees
try:
    import plotly.graph_objects as go
    
    fig = go.Figure()

    # Add each cluster as an interactive scatter trace
    cluster_color_map = {c["id"]: c["color"] for c in clusters}
    
    for cl in clusters:
        sub = df_nodes[df_nodes["cluster_id"] == cl["id"]]
        if len(sub) == 0:
            continue
            
        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=f"{cl['glyph']} {cl['name'].split(' ')[0]} ({len(sub)})",
            marker=dict(
                size=sub["is_gold"].apply(lambda g: 11 if g else 7),
                color=cl["color"],
                line=dict(width=1.5, color=sub["is_gold"].apply(lambda g: "#ffffff" if g else cl["color"])),
                opacity=0.88
            ),
            text=sub.apply(lambda r: (
                f"<b>Patient:</b> {r['short_uid']}<br>"
                f"<b>Cluster:</b> {r['cluster_name']}<br>"
                f"<b>Gold Consensus:</b> {'⭐ YES' if r['is_gold'] else 'Silver Jev'}<br>"
                f"<b>Concordance:</b> {r['concordance']*100:.0f}%<br>"
                f"<b>Brier Error:</b> {r['brier_score']:.3f}<br>"
                f"<b>Report:</b> <i>{r['report'][:90]}...</i>"
            ), axis=1),
            hoverinfo="text"
        ))

    fig.update_layout(
        title=dict(
            text="<b>THE UMAP GALAXY OF KNEES</b> // 768-D Multi-Plane Latent Projections",
            font=dict(size=18, color="#f8fafc")
        ),
        plot_bgcolor="#070b14",
        paper_bgcolor="#070b14",
        xaxis=dict(
            title="UMAP Component 1",
            gridcolor="#1e2c4c",
            zerolinecolor="#38bdf8",
            showgrid=True,
            color="#94a3b8"
        ),
        yaxis=dict(
            title="UMAP Component 2",
            gridcolor="#1e2c4c",
            zerolinecolor="#38bdf8",
            showgrid=True,
            color="#94a3b8"
        ),
        legend=dict(
            font=dict(color="#f8fafc", size=11),
            bgcolor="rgba(13, 21, 39, 0.8)",
            bordercolor="#1e2c4c",
            borderwidth=1
        ),
        width=950,
        height=620,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.show()
except ImportError:
    print("Plotly not installed. Generating Matplotlib fallback...")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 7), facecolor='#070b14')
    ax = plt.gca()
    ax.set_facecolor('#070b14')
    
    for cl in clusters:
        sub = df_nodes[df_nodes['cluster_id'] == cl['id']]
        plt.scatter(sub['x'], sub['y'], c=cl['color'], label=cl['name'].split(' ')[0], s=35, alpha=0.85)
        
    plt.title('UMAP Galaxy of Knees (768-D Multi-Plane Space)', color='#f8fafc', fontsize=14)
    plt.legend(facecolor='#0d1527', edgecolor='#1e2c4c', labelcolor='#f8fafc')
    plt.tick_params(colors='#94a3b8')
    plt.show()
"""
    ))

    # Markdown: 12-Axis Radar HUD
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 🎯 3. 12-Axis Clinical Radar HUD

The Clinical Radar HUD compares the **true radiologist ground truth** (Cyan polygon) against **model predicted probabilities** (Orange polygon) across all 12 anatomical compartments:
1. `ACL` | 2. `MCL` | 3. `Medial Meniscus` | 4. `Lateral Meniscus`
5. `Medial OA` | 6. `Lateral OA` | 7. `PF OA` | 8. `Effusion`
9. `Synovitis` | 10. `Baker's` | 11. `Contusion` | 12. `Fracture`

When the polygons overlap closely, the model exhibits perfect clinical concordance. When the polygons diverge, it immediately flags subtle clinical misses or false alarms (such as Cleanlab-audited effusion over-calling).
"""
    ))

    # Code: Plotly Radar Chart
    cells.append(nbf.v4.new_code_cell(
"""# 4. Interactive 12-Axis Clinical Radar HUD
def plot_clinical_radar(patient_id=0):
    node = nodes[patient_id]
    axes_data = node["radar"]
    
    categories = [d["target"] for d in axes_data]
    gt_vals = [d["ground_truth"] for d in axes_data]
    pred_vals = [d["model_prob"] for d in axes_data]
    
    # Close the radar loop
    categories_closed = categories + [categories[0]]
    gt_closed = gt_vals + [gt_vals[0]]
    pred_closed = pred_vals + [pred_vals[0]]
    
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Ground Truth Polygon
        fig.add_trace(go.Scatterpolar(
            r=gt_closed,
            theta=categories_closed,
            fill='toself',
            name='Radiologist Ground Truth',
            line=dict(color='#38bdf8', width=2.5),
            fillcolor='rgba(56, 189, 248, 0.25)'
        ))
        
        # Model Prediction Polygon
        fig.add_trace(go.Scatterpolar(
            r=pred_closed,
            theta=categories_closed,
            fill='toself',
            name='Model Probabilities',
            line=dict(color='#fb923c', width=2.5),
            fillcolor='rgba(251, 146, 60, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.0],
                    gridcolor='#1e2c4c',
                    color='#94a3b8',
                    tickvals=[0.25, 0.50, 0.75, 1.0]
                ),
                angularaxis=dict(
                    gridcolor='#1e2c4c',
                    color='#f8fafc',
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='#0d1527'
            ),
            paper_bgcolor='#070b14',
            title=dict(
                text=f"<b>12-AXIS CLINICAL RADAR HUD</b> // Patient {node['short_uid']} ({node['cluster_name']})",
                font=dict(color='#f8fafc', size=16)
            ),
            legend=dict(font=dict(color='#f8fafc'), bgcolor='rgba(13, 21, 39, 0.8)'),
            width=700,
            height=500
        )
        
        fig.show()
    except ImportError:
        print("Plotly not available. Printing text radar representation:")
        print(f"Patient {node['short_uid']} Concordance: {node['concordance']*100:.1f}%, Brier: {node['brier_score']:.4f}")
        for ax in axes_data:
            print(f"  {ax['target']:<18}: GT={ax['ground_truth']:.1f} | Pred={ax['model_prob']:.2f} | Diff={ax['abs_diff']:.2f}")

# Display radar for Case 0
plot_clinical_radar(patient_id=0)
"""
    ))

    # Markdown: Marching Cubes 3D Joint Viewer
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 🧊 4. Marching Cubes 3D Joint Viewer

Using the Marching Cubes algorithm, we threshold the 3D MRI voxel densities to generate watertight surface meshes:
* **Threshold = 150 (Cortical & Subchondral Bone)**: Reconstructs the distal femur, proximal tibia, fibular head, and patella.
* **Threshold = 95 (Articular Cartilage & Menisci)**: Captures the articular lining and meniscal wedges in the tibiofemoral joint space.
* **Threshold = 55 (Joint Capsule & Effusion)**: Models the surrounding synovial fluid envelope.

Let's render the 3D isosurface mesh using Plotly's interactive 3D WebGL engine:
"""
    ))

    # Code: Plotly 3D Mesh
    cells.append(nbf.v4.new_code_cell(
"""# 5. Interactive 3D Marching Cubes Knee Mesh
try:
    import plotly.graph_objects as go
    
    bone_mesh = meshes["bone"]
    cart_mesh = meshes["cartilage"]
    
    verts = np.array(bone_mesh["vertices"])
    faces = np.array(bone_mesh["faces"])
    
    fig3d = go.Figure()
    
    # Bone Isosurface Layer
    fig3d.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name="Bone Isosurface (Femur/Tibia)",
        color="#f1f5f9",
        opacity=1.0,
        lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.4),
        lightposition=dict(x=100, y=200, z=150)
    ))
    
    # Cartilage Isosurface Layer (Translucent Cyan)
    c_verts = np.array(cart_mesh["vertices"])
    c_faces = np.array(cart_mesh["faces"])
    
    fig3d.add_trace(go.Mesh3d(
        x=c_verts[:, 0],
        y=c_verts[:, 1],
        z=c_verts[:, 2],
        i=c_faces[:, 0],
        j=c_faces[:, 1],
        k=c_faces[:, 2],
        name="Cartilage & Meniscus Layer",
        color="#38bdf8",
        opacity=0.45,
        lighting=dict(ambient=0.5, diffuse=0.7)
    ))
    
    fig3d.update_layout(
        title=dict(
            text="<b>MARCHING CUBES 3D JOINT VIEWER</b> // Dual-Tissue Isosurface (Bone + Cartilage)",
            font=dict(color='#f8fafc', size=16)
        ),
        paper_bgcolor="#070b14",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#070b14",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=-1.5, z=1.2)
            )
        ),
        width=850,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    fig3d.show()
except ImportError:
    print("Plotly not available. Mesh statistics:")
    print(f"Bone Mesh: {bone_mesh['vertex_count']} vertices, {bone_mesh['triangle_count']} triangles.")
    print(f"Cartilage Mesh: {cart_mesh['vertex_count']} vertices, {cart_mesh['triangle_count']} triangles.")
"""
    ))

    # Markdown: Standalone Application Embedding
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 🚀 5. Launch the Full Standalone 60 FPS Web Application

For the complete cyber-clinical experience — with real-time threshold sliders, 3D Sagittal cutting plane slicer, instant search, and STL export for 3D printing — open `galaxy_of_knees.html` directly in your browser or run the cell below to embed it directly in your notebook!
"""
    ))

    # Code: IFrame display of galaxy_of_knees.html
    cells.append(nbf.v4.new_code_cell(
"""# 6. Render Full WebGL App in Notebook IFrame
from IPython.display import IFrame, display

html_path = "galaxy_of_knees.html"
if os.path.exists(html_path):
    print(f"✅ Found standalone application at: {html_path}")
    print("Tip: You can also double-click 'galaxy_of_knees.html' to open full-screen in any web browser!")
    display(IFrame(src=html_path, width="100%", height="850"))
else:
    print("galaxy_of_knees.html is located in the project root directory.")
"""
    ))

    nb["cells"] = cells

    out_nb_path = "notebooks/galaxy_of_knees_3d.ipynb"
    os.makedirs("notebooks", exist_ok=True)
    with open(out_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[+] Successfully generated Jupyter Notebook: {out_nb_path}")

if __name__ == "__main__":
    create_galaxy_notebook()
