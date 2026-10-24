"""
Builds the interactive Devpost Auto-Filler & Control Center:
  - `opencv_hackathon/submission_kit/devpost_autofill.js`
  - `opencv_hackathon/submission_kit/DEVPOST_CONTROL_CENTER.html`
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = PROJECT_ROOT / "opencv_hackathon" / "submission_kit"
SUBMISSION_MD = PROJECT_ROOT / "opencv_hackathon" / "DEVPOST_SUBMISSION.md"


def build_control_center() -> Path:
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    md_text = SUBMISSION_MD.read_text(encoding="utf-8")

    # Extract Story body starting from "## 1. Problem Statement"
    story_idx = md_text.find("## 1. Problem Statement")
    story_body = md_text[story_idx:].strip() if story_idx != -1 else md_text

    submission_data = {
        "title": "THE CMO (Chief MRI Officer) · Creaky-CV5: Autonomous 3-Plane Knee MRI Triage, OpenCV 5 Active-Zoom & Radiologist Second-Opinion Cockpit",
        "short_title": "Creaky-CV5: Autonomous OpenCV 5 + AWS COOL Knee MRI Agent",
        "elevator_pitch": "An autonomous OpenCV 5 (dnn5 + AWS COOL) 3-plane Knee MRI agent (0.957 RSNA LB) that sees across 144 slices, reasons over Cleanlab radiologist blind spots, and acts via 1.65× Lanczos4 focal zoom.",
        "story_markdown": story_body,
        "built_with": [
            "opencv",
            "aws",
            "aws-graviton",
            "python",
            "onnx",
            "pytorch",
            "fastapi",
            "html5",
            "medical-imaging",
            "computer-vision",
        ],
        "try_it_out_urls": [
            "https://raw.githack.com/switzloco/creaky/main/opencv_hackathon/dist/index.html",
            "https://raw.githack.com/switzloco/creaky/main/physician_pitch.html",
            "https://raw.githack.com/switzloco/creaky/main/galaxy_of_knees.html",
        ],
        "github_url": "https://github.com/switzloco/creaky",
        "physician_pitch_url": "https://raw.githack.com/switzloco/creaky/main/physician_pitch.html",
        "brother_message": (
            "Hey bro — we just built an AI + computer-vision second-opinion tool for 3-plane Knee MRIs (trained on 2,668 RSNA studies, 0.957 AUC). "
            "In our audit against a 3-radiologist consensus panel, busy single readers omitted 12–18% of secondary findings (like effusions/synovitis/subtle meniscal tears) due to satisfaction of search. "
            "It runs in ~116 ms inside a hospital VPC with zero PHI egress and shows an unmarked 1.65x tissue zoom + cortical bone edge map next to a matched normal knee reference.\n\n"
            "Here's the 2-minute interactive clinical brief & live PACS demo if you or any radiologist/ortho friends want to play with it:\n"
            "https://raw.githack.com/switzloco/creaky/main/physician_pitch.html"
        ),
        "youtube_title": "Creaky-CV5 · Autonomous OpenCV 5 + AWS Graviton COOL Knee MRI Agent (OpenCV AI Competition 2026)",
        "youtube_description": (
            "Official submission for the OpenCV AI Competition 2026, powered by AWS (Overall + Best Use of COOL + Agentic Vision tracks).\n\n"
            "• Live Interactive Web Cockpit: https://raw.githack.com/switzloco/creaky/main/opencv_hackathon/dist/index.html\n"
            "• 3D Knee & UMAP Explorer: https://raw.githack.com/switzloco/creaky/main/galaxy_of_knees.html\n"
            "• GitHub Repository: https://github.com/switzloco/creaky/tree/main/opencv_hackathon\n\n"
            "Highlights:\n"
            "- 0.957 Macro ROC-AUC on the RSNA Knee MRI Challenge Leaderboard (0.8568 on 3-Radiologist Gold Consensus Panel)\n"
            "- 3-Pass OpenCV 5.0.0 (dnn5 + AWS Graviton COOL KleidiCV) Agentic Vision Loop (See -> Reason -> Act)\n"
            "- 116.5 ms per 144-slice 3-plane study on AWS Graviton COOL (1,236 slices/sec, 1.79x speedup, 52% lower cloud cost)\n"
            "- Automated HIPAA Safe-Harbor Burned-In Pixel PHI Scrubber (cv2.MORPH_TOPHAT) with zero cloud PHI egress"
        ),
    }

    # Create the Devpost DOM Auto-Fill Script
    autofill_js = f"""(function() {{
  const DATA = {json.dumps(submission_data)};

  function setNativeValue(el, val) {{
    if (!el) return false;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set
      || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
    if (setter) {{
      setter.call(el, val);
    }} else {{
      el.value = val;
    }}
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
    el.dispatchEvent(new Event('blur', {{ bubbles: true }}));
    return true;
  }}

  let filledCount = 0;

  // 1. Project Name / Title
  const titleSelectors = [
    '#software_name',
    'input[name="software[name]"]',
    'input[name*="title" i]',
    'input[placeholder*="project name" i]',
    'input[id*="name" i]'
  ];
  for (const sel of titleSelectors) {{
    const el = document.querySelector(sel);
    if (el && el.type !== 'hidden') {{
      const maxLen = el.maxLength && el.maxLength > 0 ? el.maxLength : 200;
      setNativeValue(el, maxLen < 80 ? DATA.short_title : DATA.title.slice(0, maxLen));
      filledCount++;
      break;
    }}
  }}

  // 2. Elevator Pitch / Tagline
  const pitchSelectors = [
    '#software_tagline',
    'input[name="software[tagline]"]',
    'textarea[name="software[tagline]"]',
    'input[name*="pitch" i]',
    'textarea[name*="pitch" i]',
    'input[placeholder*="elevator pitch" i]',
    'input[placeholder*="short" i]'
  ];
  for (const sel of pitchSelectors) {{
    const el = document.querySelector(sel);
    if (el) {{
      setNativeValue(el, DATA.elevator_pitch);
      filledCount++;
      break;
    }}
  }}

  // 3. Project Story / Description (Markdown)
  const descSelectors = [
    '#software_description',
    'textarea[name="software[description]"]',
    'textarea[id*="description" i]',
    'textarea.markdown-editor',
    '.CodeMirror'
  ];
  for (const sel of descSelectors) {{
    const el = document.querySelector(sel);
    if (el) {{
      if (el.CodeMirror) {{
        el.CodeMirror.setValue(DATA.story_markdown);
      }} else {{
        setNativeValue(el, DATA.story_markdown);
      }}
      filledCount++;
      break;
    }}
  }}

  // 4. Try It Out URLs & GitHub Repo URL
  const urlInputs = Array.from(document.querySelectorAll('input[type="url"], input[name*="url" i], input[placeholder*="http" i], input[name*="website" i], input[name*="github" i]'));
  let tryIdx = 0;
  urlInputs.forEach((inp) => {{
    const nm = (inp.name + ' ' + inp.id + ' ' + inp.placeholder).toLowerCase();
    if (nm.includes('video') || nm.includes('youtube')) {{
      // Leave video URL for user's YouTube link unless empty
    }} else if (nm.includes('github') || nm.includes('repo') || nm.includes('source')) {{
      setNativeValue(inp, DATA.github_url);
      filledCount++;
    }} else if (tryIdx < DATA.try_it_out_urls.length) {{
      setNativeValue(inp, DATA.try_it_out_urls[tryIdx++]);
      filledCount++;
    }}
  }});

  // 5. Opt-in Checkboxes (Best Use of COOL, Agentic Vision, Rules)
  const checkboxes = Array.from(document.querySelectorAll('input[type="checkbox"]'));
  checkboxes.forEach((cb) => {{
    const lbl = (cb.closest('label')?.innerText || cb.parentElement?.innerText || '').toLowerCase();
    if (lbl.includes('cool') || lbl.includes('agentic') || lbl.includes('rule') || lbl.includes('eligible') || lbl.includes('terms') || lbl.includes('overall')) {{
      if (!cb.checked) {{
        cb.click();
        cb.checked = true;
        cb.dispatchEvent(new Event('change', {{ bubbles: true }}));
        filledCount++;
      }}
    }}
  }});

  // Show confirmation banner inside Devpost
  const banner = document.createElement('div');
  banner.style.cssText = 'position:fixed;top:16px;right:16px;z-index:999999;background:#065f46;color:#fff;padding:16px 22px;border-radius:12px;border:2px solid #10b981;font-family:sans-serif;box-shadow:0 15px 35px rgba(0,0,0,0.4);max-width:420px;';
  banner.innerHTML = '<div style="font-weight:900;font-size:15px;margin-bottom:4px;">⚡ Creaky-CV5 Devpost Auto-Filler Executed!</div>' +
    '<div style="font-size:13px;color:#d1fae5;">Auto-populated <strong>' + filledCount + '</strong> fields on this step. Click Save & Continue to move to the next Devpost tab and click the bookmarklet again!</div>';
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 7000);
}})();"""

    autofill_path = KIT_DIR / "devpost_autofill.js"
    autofill_path.write_text(autofill_js, encoding="utf-8")

    # Bookmarklet URL-encoded string
    import urllib.parse
    bookmarklet_href = "javascript:" + urllib.parse.quote(autofill_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>⚡ Creaky-CV5 · 1-Click Devpost Submission Control Center</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    background: #090d16; color: #f8fafc; line-height: 1.5; padding-bottom: 60px;
  }}
  .topbar {{
    background: #0f172a; border-bottom: 2px solid #10b981; padding: 16px 32px;
    display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100;
  }}
  .tb-title {{ font-size: 20px; font-weight: 900; color: #fff; }}
  .tb-title span {{ color: #10b981; }}
  .tb-links {{ display: flex; gap: 12px; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px;
    background: #10b981; color: #022c22; font-weight: 900; font-size: 13px;
    padding: 10px 18px; border-radius: 10px; text-decoration: none; border: none; cursor: pointer;
    transition: all 0.15s;
  }}
  .btn:hover {{ background: #34d399; transform: translateY(-1px); }}
  .btn-sky {{ background: #38bdf8; color: #082f49; }}
  .btn-amber {{ background: #f59e0b; color: #451a03; }}
  .btn-outline {{ background: #1e293b; color: #e2e8f0; border: 1px solid #475569; }}
  .btn-outline:hover {{ border-color: #10b981; color: #10b981; background: #0f172a; }}

  .container {{ max-width: 1380px; margin: 24px auto; padding: 0 28px; }}

  .hero-box {{
    background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
    border: 2px solid #10b981; border-radius: 18px; padding: 24px 28px; margin-bottom: 24px;
    display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 24px; align-items: center;
  }}
  .bookmarklet-pill {{
    display: inline-block; background: #fbbf24; color: #451a03; font-weight: 900;
    font-size: 15px; padding: 14px 24px; border-radius: 12px; text-decoration: none;
    border: 2px dashed #78350f; box-shadow: 0 10px 25px rgba(245, 158, 11, 0.3);
    cursor: grab; margin-top: 12px;
  }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  @media (max-width: 1000px) {{ .grid-2, .hero-box {{ grid-template-columns: 1fr; }} }}

  .card {{
    background: #0f172a; border: 1px solid #334155; border-radius: 16px; padding: 20px 24px;
    margin-bottom: 20px;
  }}
  .card-hdr {{
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
    padding-bottom: 10px; border-bottom: 1px solid #1e293b;
  }}
  .card-title {{ font-size: 14px; font-weight: 900; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px; }}
  .field-box {{
    width: 100%; background: #020617; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 10px; padding: 12px; font-family: monospace; font-size: 13px; resize: vertical;
  }}

  .img-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 12px;
  }}
  .img-card {{
    background: #020617; border: 1px solid #334155; border-radius: 10px; overflow: hidden; padding: 8px;
  }}
  .img-card img {{ width: 100%; height: 140px; object-fit: cover; border-radius: 6px; }}
  .img-card .lbl {{ font-size: 11px; font-weight: 800; color: #cbd5e1; margin-top: 6px; word-break: break-all; }}
</style>
</head>
<body>

  <header class="topbar">
    <div class="tb-title">⚡ CREAKY-CV5 · <span>Devpost 5-Minute Submission Control Center</span></div>
    <div class="tb-links">
      <a class="btn" href="https://opencv26.devpost.com/" target="_blank">1. Open OpenCV 2026 Devpost ↗</a>
      <a class="btn btn-sky" href="https://studio.youtube.com/" target="_blank">2. Open YouTube Studio (Upload MP4) ↗</a>
      <a class="btn btn-amber" href="../../physician_pitch.html" target="_blank">👨‍⚕️ Preview Doctor/Radiologist Pitch Page ↗</a>
      <a class="btn btn-outline" href="../dist/index.html" target="_blank">Preview Live Cockpit ↗</a>
    </div>
  </header>

  <main class="container">

    <!-- DOCTOR / RADIOLOGIST PITCH CARD FOR YOUR BROTHER -->
    <section class="card" style="border:2px solid #38bdf8;background:linear-gradient(135deg, #0c4a6e 0%, #0f172a 100%);">
      <div class="card-hdr" style="border-bottom-color:rgba(56,189,248,0.3);">
        <span class="card-title" style="color:#38bdf8;">👨‍⚕️ Pitch Link &amp; Text Message for Your Brother (Doctor / Radiologist Contacts)</span>
        <div style="display:flex;gap:10px;">
          <button class="btn btn-sky" onclick="copyField('bro-link', this)">🔗 Copy Pitch Link Only</button>
          <button class="btn" onclick="copyField('bro-msg', this)">💬 Copy Ready-to-Send Text / Email</button>
        </div>
      </div>
      <div class="grid-2" style="margin-bottom:0;">
        <div>
          <div style="font-size:12px;font-weight:800;color:#bae6fd;margin-bottom:6px;">Direct Interactive Physician &amp; Radiologist Brief URL (Mobile &amp; Desktop Ready):</div>
          <input id="bro-link" class="field-box" style="margin-bottom:10px;font-weight:800;color:#34d399;" value="{submission_data['physician_pitch_url']}" readonly />
          <div style="font-size:12.5px;color:#cbd5e1;">
            Stripped of all hackathon jargon—speaks directly to radiologists &amp; orthopedists about catching <strong>12–18% "satisfaction of search" single-reader omissions</strong>, unmarked <code>1.65×</code> tissue zoom, subchondral cortical edge containment, and matched normal knee references.
          </div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:800;color:#bae6fd;margin-bottom:6px;">Ready-to-Send Text / WhatsApp / Email to Your Brother:</div>
          <textarea id="bro-msg" class="field-box" rows="5" readonly>{submission_data['brother_message']}</textarea>
        </div>
      </div>
    </section>

    <section class="hero-box">
      <div>
        <div style="font-size:12px;font-weight:900;color:#34d399;text-transform:uppercase;letter-spacing:1px;">Option A · Instant Browser Auto-Fill</div>
        <h2 style="font-size:24px;font-weight:900;margin:6px 0;">Drag This Button to Your Bookmarks Bar — Or Use the 1-Click Copy Buttons Below</h2>
        <p style="font-size:14px;color:#cbd5e1;">
          1. Drag the yellow <strong>⚡ Auto-Fill Creaky-CV5 Devpost</strong> button below onto your Chrome Bookmarks Bar (<code>Ctrl+Shift+B</code> to show bar).<br/>
          2. Open your <a href="https://opencv26.devpost.com/" target="_blank" style="color:#38bdf8;font-weight:700;">OpenCV 2026 Devpost Submission Form</a> and click the bookmarklet on each tab to auto-fill Title, Pitch, Markdown Story, Links, and Prize Track checkboxes!
        </p>
        <a class="bookmarklet-pill" href="{bookmarklet_href}">⚡ Auto-Fill Creaky-CV5 Devpost (Drag Me to Bookmarks Bar)</a>
      </div>
      <div style="background:#090d16;padding:18px;border-radius:14px;border:1px solid #334155;">
        <div style="font-size:12px;font-weight:900;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;">Or Paste in DevTools Console (F12 on Devpost)</div>
        <p style="font-size:12.5px;color:#94a3b8;margin-bottom:10px;">
          If you prefer not to use bookmarks: press <code>F12</code> → <code>Console</code> on the Devpost tab, click the button below, and press <code>Ctrl+V</code> + <code>Enter</code>:
        </p>
        <button class="btn btn-amber" style="width:100%;justify-content:center;" onclick="copyField('console-script', this)">📋 Copy Devpost Console Auto-Fill Script</button>
        <textarea id="console-script" style="display:none;">{autofill_js}</textarea>
      </div>
    </section>

    <!-- VIDEO & YOUTUBE SECTION -->
    <section class="card">
      <div class="card-hdr">
        <div class="card-title">🎬 Step 1 · Pre-Rendered 1080p Narrated Demo Video (`creaky_cv5_demo_video.mp4`)</div>
        <span style="font-size:12px;color:#10b981;font-weight:800;">Ready to Upload to YouTube (Unlisted or Public)</span>
      </div>
      <div class="grid-2" style="margin-bottom:0;">
        <div>
          <video controls style="width:100%;border-radius:12px;border:2px solid #334155;background:#000;" src="./creaky_cv5_demo_video.mp4"></video>
          <div style="margin-top:8px;font-size:12px;color:#94a3b8;">
            File path: <code>c:\\dev\\creaky\\opencv_hackathon\\submission_kit\\creaky_cv5_demo_video.mp4</code>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:12px;">
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
              <strong style="font-size:12px;color:#cbd5e1;">YouTube Video Title</strong>
              <button class="btn btn-outline" style="padding:4px 10px;font-size:11px;" onclick="copyField('yt-title', this)">Copy Title</button>
            </div>
            <input id="yt-title" class="field-box" value="{submission_data['youtube_title']}" readonly />
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
              <strong style="font-size:12px;color:#cbd5e1;">YouTube Video Description</strong>
              <button class="btn btn-outline" style="padding:4px 10px;font-size:11px;" onclick="copyField('yt-desc', this)">Copy Description</button>
            </div>
            <textarea id="yt-desc" class="field-box" rows="7" readonly>{submission_data['youtube_description']}</textarea>
          </div>
        </div>
      </div>
    </section>

    <!-- DEVPOST FORM FIELDS (1-CLICK COPY) -->
    <div class="grid-2">
      <div class="card" style="margin-bottom:0;">
        <div class="card-hdr">
          <span class="card-title">📝 Field 1 · Project Name / Title</span>
          <button class="btn" onclick="copyField('f-title', this)">📋 Copy Project Title</button>
        </div>
        <textarea id="f-title" class="field-box" rows="2" readonly>{submission_data['title']}</textarea>
      </div>

      <div class="card" style="margin-bottom:0;">
        <div class="card-hdr">
          <span class="card-title">⚡ Field 2 · Elevator Pitch (198 / 200 chars)</span>
          <button class="btn" onclick="copyField('f-pitch', this)">📋 Copy Elevator Pitch</button>
        </div>
        <textarea id="f-pitch" class="field-box" rows="2" readonly>{submission_data['elevator_pitch']}</textarea>
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <div class="card-hdr">
        <span class="card-title">📖 Field 3 · Full Project Story / Technical Report (Markdown — Meets All OpenCV 2026 Rubric Criteria)</span>
        <button class="btn" onclick="copyField('f-story', this)">📋 Copy Full Markdown Story</button>
      </div>
      <textarea id="f-story" class="field-box" rows="14" readonly>{submission_data['story_markdown']}</textarea>
    </div>

    <div class="grid-2">
      <div class="card" style="margin-bottom:0;">
        <div class="card-hdr">
          <span class="card-title">🛠️ Field 4 · "Built With" Tags</span>
          <button class="btn" onclick="copyField('f-tags', this)">📋 Copy Comma-Separated Tags</button>
        </div>
        <input id="f-tags" class="field-box" value="{', '.join(submission_data['built_with'])}" readonly />
      </div>

      <div class="card" style="margin-bottom:0;">
        <div class="card-hdr">
          <span class="card-title">🔗 Field 5 · "Try It Out" &amp; GitHub Links</span>
          <button class="btn" onclick="copyField('f-links', this)">📋 Copy Links</button>
        </div>
        <textarea id="f-links" class="field-box" rows="4" readonly>{chr(10).join(submission_data['try_it_out_urls'] + [submission_data['github_url']])}</textarea>
      </div>
    </div>

    <!-- SCREENSHOTS & THUMBNAIL GALLERY -->
    <section class="card" style="margin-top:20px;">
      <div class="card-hdr">
        <span class="card-title">🖼️ Step 3 · Generated Thumbnail (1280×720) &amp; 1080p Gallery Screenshots (`opencv_hackathon/submission_kit/images/`)</span>
        <span style="font-size:12px;color:#38bdf8;font-weight:800;">Drag &amp; Drop into Devpost</span>
      </div>
      <div class="img-grid">
        <div class="img-card"><img src="./images/00_devpost_thumbnail_1280x720.png"/><div class="lbl">00_devpost_thumbnail_1280x720.png (Thumbnail)</div></div>
        <div class="img-card"><img src="./images/01_architecture_and_agent_workflow.png"/><div class="lbl">01_architecture_and_agent_workflow.png</div></div>
        <div class="img-card"><img src="./images/02_cockpit_case1_acl_effusion_second_opinion.png"/><div class="lbl">02_cockpit_case1_acl_effusion_second_opinion.png</div></div>
        <div class="img-card"><img src="./images/03_cockpit_case2_scharr_cortical_edge_map.png"/><div class="lbl">03_cockpit_case2_scharr_cortical_edge_map.png</div></div>
        <div class="img-card"><img src="./images/04_cockpit_case3_patellar_cartilage_zoom.png"/><div class="lbl">04_cockpit_case3_patellar_cartilage_zoom.png</div></div>
        <div class="img-card"><img src="./images/05_cockpit_case4_meniscus_surface_contact.png"/><div class="lbl">05_cockpit_case4_meniscus_surface_contact.png</div></div>
        <div class="img-card"><img src="./images/06_volumetric_scrubber_and_aws_cool_telemetry.png"/><div class="lbl">06_volumetric_scrubber_and_aws_cool_telemetry.png</div></div>
        <div class="img-card"><img src="./images/07_galaxy_of_knees_3d_explorer.png"/><div class="lbl">07_galaxy_of_knees_3d_explorer.png</div></div>
      </div>
    </section>

  </main>

<script>
function copyField(id, btn) {{
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.value);
  const orig = btn.textContent;
  btn.textContent = '✅ Copied!';
  setTimeout(() => btn.textContent = orig, 2000);
}}
</script>
</body>
</html>
"""

    out_html = KIT_DIR / "DEVPOST_CONTROL_CENTER.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Built Devpost Control Center: {out_html}")
    return out_html


if __name__ == "__main__":
    build_control_center()
