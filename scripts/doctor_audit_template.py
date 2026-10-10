# ==============================================================================
# 2. EMBEDDED HIGH-FIDELITY MEDICAL CASE REPOSITORY & DASHBOARD RENDERER
# ==============================================================================

import json
from IPython.display import HTML, display

IMAGE_PAYLOAD = __IMAGE_PAYLOAD__

CASES_DATA = [
    {
        "id": "case-1",
        "case_num": 1,
        "title": "Case 1: Massive Effusion Overcall in Acute ACL Rupture",
        "study_uid": "1.2.826.0.1.3680043.8.498.26790702379506190936834447203448882465",
        "badge": "label-omission",
        "badge_label": "Probable Label Omission (Ground Truth Missed)",
        "badge_color": "#8b5cf6",
        "plane": "Sagittal (Fluid-Sensitive T2/PD)",
        "target": "Joint Effusion (Secondary: ACL)",
        "model_prob": 97.1,
        "gold_label": 0.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["c1_raw"],
        "heat_img": IMAGE_PAYLOAD["c1_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_sag"],
        "report_quote": "ACL is ruptured in its mid substance. There is mild residual bone contusion in the weightbearing aspect of the lateral femoral condyle and posterior lateral tibial condyle. Mild anterior tibial translation noted.",
        "dilemma": "In an acute complete ACL rupture with kissing bone contusions and anterior tibial translation, hemarthrosis / joint effusion is virtually always present. The algorithm predicted 97.1% effusion confidence based on fluid pooling in the anterior and suprapatellar pouch. Yet the Gold Standard label marked Effusion = 0.0! The algorithm is being penalized for detecting real pathology that the expert annotator omitted.",
        "questions": [
            "Looking at the fluid distension in the anterior suprapatellar space: would you classify this volume as a clinically significant joint effusion (hemarthrosis)?",
            "In your practice, do you ever encounter complete mid-substance ACL ruptures with zero joint effusion?",
            "Should this case be re-labeled as Effusion = 1.0 in the training set?"
        ]
    },
    {
        "id": "case-2",
        "case_num": 2,
        "title": "Case 2: Subtle Lateral Meniscal Tear with Multi-Ligament Hallucinations",
        "study_uid": "1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269",
        "badge": "false-positive",
        "badge_label": "False Positive Cascade (Cross-Compartment Signal Leak)",
        "badge_color": "#f59e0b",
        "plane": "Sagittal (T2/PD-FS)",
        "target": "Lateral Meniscus (Confounders: ACL, Medial Meniscus)",
        "model_prob": 88.6,
        "gold_label": 1.0,
        "prob_color": "#16a34a",
        "raw_img": IMAGE_PAYLOAD["c2_raw"],
        "heat_img": IMAGE_PAYLOAD["c2_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_sag"],
        "report_quote": "Horizontal tear at the anterior horn of the lateral meniscus communicating with the inferior articular surface. Cruciate ligaments and collateral ligaments remain intact.",
        "dilemma": "While the model correctly caught the lateral meniscal tear (88.6%), it hallucinated tears across the intact ACL (76.2%) and Medial Meniscus (61.8%). When an algorithm detects one primary tear, deep feature representations can bleed across compartments, triggering multi-ligament false alarms.",
        "questions": [
            "Do you observe any subtle partial-volume averaging or degenerative fraying in the medial compartment that could explain the model's medial overcall?",
            "What specific anatomical landmarks would you inspect to confirm that the ACL is completely intact despite the algorithm's 76% alarm?"
        ]
    },
    {
        "id": "case-3",
        "case_num": 3,
        "title": "Case 3: Complex Pivot-Shift Injury & Contralateral Meniscus Dilemma",
        "study_uid": "1.2.826.0.1.3680043.8.498.11382021393803389951964005983002209238",
        "badge": "false-positive",
        "badge_label": "High-Confidence False Positive (Lateral Meniscus)",
        "badge_color": "#f59e0b",
        "plane": "Sagittal (T2-Weighted)",
        "target": "Lateral Meniscus (True: ACL + Medial Meniscus)",
        "model_prob": 79.1,
        "gold_label": 0.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["c3_raw"],
        "heat_img": IMAGE_PAYLOAD["c3_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_sag"],
        "report_quote": "In the medial compartment, there is a longitudinal vertical tear at the body and posterior horn of the medial meniscus. Complete disruption of the anterior cruciate ligament with associated pivot-shift contusion.",
        "dilemma": "The model achieved near 100% confidence on the medial tear (96.2%) and ACL rupture (99.7%), but predicted a 79.1% probability of a lateral meniscus tear where the gold label was 0.0. Does the lateral meniscus show subtle peripheral detachment or meniscofemoral ligament overlap that tricked the network?",
        "questions": [
            "In high-energy pivot-shift injuries with complete ACL rupture, how frequently do subtle lateral meniscal ramps or posterior root sprains go un-annotated on single-slice reviews?",
            "Does this lateral meniscus slice look completely pristine to your trained eye, or is there equivocal signal near the capsule?"
        ]
    },
    {
        "id": "case-4",
        "case_num": 4,
        "title": "Case 4: Bone Marrow Contusion Mimicry & Ligament False Alarm Storm",
        "study_uid": "1.2.826.0.1.3680043.8.498.11548045715264151632153040089882701935",
        "badge": "breakdown",
        "badge_label": "Catastrophic Confounder (Marrow Edema False Alarm Storm)",
        "badge_color": "#ef4444",
        "plane": "Axial (Fluid-Sensitive PD-FS)",
        "target": "Ligament Overcalls (True: Bone Contusion & Effusion)",
        "model_prob": 96.1,
        "gold_label": 0.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["c4_raw"],
        "heat_img": IMAGE_PAYLOAD["c4_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_ax"],
        "report_quote": "Large distention of the suprapatellar bursa compatible with extensive joint effusion. Extensive subcortical bone marrow contusion pattern identified in the lateral compartment. Cruciate and collateral ligaments intact.",
        "dilemma": "This is our model's most dangerous blind spot: high-intensity T2 bone marrow contusion edema caused the model to fire 90%+ confidence across EVERY ligament and meniscus (ACL 96.1%, MCL 90.9%, MM 86.0%, LM 90.3%). The convolutional filters conflated marrow edema brightness with soft-tissue tears.",
        "questions": [
            "How do radiologists train their eyes to ignore overwhelming bright marrow edema when assessing ligament margins?",
            "Would masking bone tissue or adding explicit cortical bone boundary supervision help AI models avoid this edema confounder?"
        ]
    },
    {
        "id": "case-5",
        "case_num": 5,
        "title": "Case 5: Full-Thickness Patellar Chondral Loss (Critical False Negative)",
        "study_uid": "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149",
        "badge": "false-negative",
        "badge_label": "Critical False Negative (Missed Severe Osteoarthritis)",
        "badge_color": "#ef4444",
        "plane": "Axial (Patellofemoral Joint Space)",
        "target": "Patellofemoral Osteoarthritis (PF OA)",
        "model_prob": 15.4,
        "gold_label": 1.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["c5_raw"],
        "heat_img": IMAGE_PAYLOAD["c5_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_ax"],
        "report_quote": "Severe patellofemoral osteoarthritis with full-thickness chondral loss across the median ridge of the patella, subchondral cyst formation, and lateral patellar tilt with marginal osteophyte formation.",
        "dilemma": "Despite full-thickness cartilage delamination, subchondral sclerosis, and marginal osteophytes on the patella, the model output only a 15.4% probability for PF OA! Standard 2D pooling failed to register focal cartilage ulceration because the overall joint space contour appeared intact.",
        "questions": [
            "What specific spatial feature (trochlear groove depth, median ridge subchondral cysts, or lateral tilt) is most diagnostic of severe PF OA?",
            "Why do you think computer vision models detect tibiofemoral joint space narrowing much more easily than patellofemoral cartilage erosion?"
        ]
    },
    {
        "id": "case-6",
        "case_num": 6,
        "title": "Case 6: Intrameniscal Mucoid Degeneration vs Operative Tear",
        "study_uid": "1.2.826.0.1.3680043.8.498.56512564464301544960446419143135447363",
        "badge": "false-positive",
        "badge_label": "False Positive (Fooled by Grade II Intrameniscal Mucin)",
        "badge_color": "#f59e0b",
        "plane": "Coronal / Sagittal (Medial Compartment)",
        "target": "Medial Meniscus",
        "model_prob": 97.2,
        "gold_label": 0.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["ref_sag"],
        "heat_img": IMAGE_PAYLOAD["c6_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_cor"],
        "report_quote": "Medial meniscus mucoid degeneration without discrete surface-extending tear. Lateral meniscus normal. Moderate OA of medial compartment. Effusion with synovitis.",
        "dilemma": "Grade II intrameniscal mucoid degeneration produces increased intrameniscal T2/PD signal that does NOT reach the superior or inferior articular surface. The algorithm sees high pixel brightness and confidently flags a tear (97.2%), failing the critical radiologic 'articular surface contact rule'.",
        "questions": [
            "Does this signal unambiguously terminate before the articular surface (true Grade II), or is there a subtle inferior surface fissure that could cause symptoms?",
            "How can we teach an algorithm to verify that hyperintense signal reaches the articular cortex before declaring a tear?"
        ]
    },
    {
        "id": "case-7",
        "case_num": 7,
        "title": "Case 7: Chronic Mucoid Degeneration of the ACL vs Traumatic Tear",
        "study_uid": "1.2.826.0.1.3680043.8.498.13335129881737731410081002627729200903",
        "badge": "false-negative",
        "badge_label": "False Negative (The 'Celery Stalk' Mucoid ACL Blind Spot)",
        "badge_color": "#ef4444",
        "plane": "Sagittal (Intercondylar Notch)",
        "target": "ACL Abnormality",
        "model_prob": 5.6,
        "gold_label": 1.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["ref_sag"],
        "heat_img": IMAGE_PAYLOAD["c7_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_cor"],
        "report_quote": "Gevorderde mucoïde degeneratie van de voorste kruisband, met opgezet voorkomen en gestegen T2-signaal (Advanced mucoid degeneration of ACL with swollen 'celery stalk' appearance and increased T2 signal)... no traumatic disruption.",
        "dilemma": "The model was trained heavily to detect traumatic fiber discontinuity (empty notch sign). Here, the ACL is anatomically present and continuous, but thickened with diffuse mucinous matrix. The model scored it 5.6% (intact), while the gold radiologist flagged it as abnormal (1.0).",
        "questions": [
            "In your clinical practice, do you classify mucoid degeneration under the umbrella of 'ACL Abnormality', or do you treat it as a distinct non-surgical entity?",
            "What clinical signs differentiate an intact mucoid ACL from a chronic scarred partial rupture?"
        ]
    },
    {
        "id": "case-8",
        "case_num": 8,
        "title": "Case 8: Tibial Plateau Split Fracture with Omitted Multi-Ligament Labels",
        "study_uid": "1.2.826.0.1.3680043.8.498.13267780356245120052411517053322874891",
        "badge": "label-omission",
        "badge_label": "Massive Ground Truth Omission (Model Found Complete Injury)",
        "badge_color": "#8b5cf6",
        "plane": "Coronal & Sagittal (Tibial Plateau)",
        "target": "Fracture, ACL, Contusion, Effusion",
        "model_prob": 99.9,
        "gold_label": 0.0,
        "prob_color": "#dc2626",
        "raw_img": IMAGE_PAYLOAD["ref_cor"],
        "heat_img": IMAGE_PAYLOAD["c8_heat"],
        "ref_img": IMAGE_PAYLOAD["ref_ax"],
        "report_quote": "Split fractures at tibial plateau with depression of the lateral articular surface... Bone contusion with neglected fracture line at fibular head. The contour of ACL is not in a straight line with increased signal intensity inside, in favor of grade 2 injury.",
        "dilemma": "The text report explicitly describes a depressed tibial plateau fracture, bone contusion, and ACL Grade 2 injury! The vision model detected ALL of them at >99% confidence. However, the Gold annotator only checked Fracture and Lateral Meniscus, leaving ACL=0 and Contusion=0! The algorithm is being punished for being MORE thorough than the human labeler!",
        "questions": [
            "Does the scan support the algorithm's prediction of a concomitant ACL sprain and severe bone contusion alongside the tibial plateau fracture?",
            "Is this a clear case of ground truth annotation fatigue where secondary findings were omitted?"
        ]
    }
]

def render_case_card_html(case):
    """Generates clean, medical-grade HTML card for a failure case."""
    badge_bg = case['badge_color']
    
    questions_items = []
    for i, q in enumerate(case['questions']):
        questions_items.append(f"<li style='margin-bottom: 8px; color: #334155;'><b>Q{i+1}:</b> {q}</li>")
    questions_html = "".join(questions_items)
    
    html = f'''
    <div id="{case['id']}" style="background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 36px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <!-- Card Header -->
        <div style="background: #f8fafc; padding: 18px 24px; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="background: {badge_bg}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                    {case['badge_label']}
                </span>
                <h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 20px; font-weight: 700;">{case['title']}</h3>
                <span style="color: #64748b; font-size: 13px;">UID: <code>{case['study_uid'][:28]}...</code> | Plane: <b>{case['plane']}</b></span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Primary Metric</div>
                <div style="font-size: 22px; font-weight: 800; color: {case['prob_color']};">
                    Pred: {case['model_prob']:.1f}% <span style="font-size: 14px; font-weight: 500; color: #64748b;">(Gold: {case['gold_label']})</span>
                </div>
            </div>
        </div>

        <div style="padding: 24px;">
            <!-- MRI Viewer Comparison Row -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div style="background: #0f172a; border-radius: 8px; padding: 8px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 12px; font-weight: 600; margin-bottom: 6px;">DIAGNOSTIC MRI SCAN (HIGH CONTRAST)</div>
                    <img src="data:image/jpeg;base64,{case['raw_img']}" style="width: 100%; max-width: 340px; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" alt="Raw Scan">
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Authentic Windowed Slice</div>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 8px; text-align: center;">
                    <div style="color: #f59e0b; font-size: 12px; font-weight: 600; margin-bottom: 6px;">ALGORITHMIC ATTENTION / ROI OVERLAY</div>
                    <img src="data:image/jpeg;base64,{case['heat_img']}" style="width: 100%; max-width: 340px; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" alt="Heatmap">
                    <div style="color: #f59e0b; font-size: 11px; margin-top: 4px;">Region Triggering Model Activation</div>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 8px; text-align: center;">
                    <div style="color: #10b981; font-size: 12px; font-weight: 600; margin-bottom: 6px;">HEALTHY ANATOMICAL REFERENCE</div>
                    <img src="data:image/jpeg;base64,{case['ref_img']}" style="width: 100%; max-width: 340px; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" alt="Normal Benchmark">
                    <div style="color: #10b981; font-size: 11px; margin-top: 4px;">Verified Normal Benchmark (Same Plane)</div>
                </div>
            </div>

            <!-- Radiologist Report Box -->
            <div style="background: #f1f5f9; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <div style="font-size: 12px; font-weight: 700; color: #1e3a8a; text-transform: uppercase; margin-bottom: 4px;">
                    📄 Hospital Radiologist Diagnostic Report:
                </div>
                <div style="font-style: italic; color: #334155; font-size: 14px; line-height: 1.5;">
                    "{case['report_quote']}"
                </div>
            </div>

            <!-- The Diagnostic Dilemma -->
            <div style="background: #fffbeb; border: 1px solid #fef3c7; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <div style="font-size: 13px; font-weight: 700; color: #92400e; text-transform: uppercase; margin-bottom: 4px;">
                    🔍 Why Our Algorithm Struggled Here:
                </div>
                <div style="color: #78350f; font-size: 14px; line-height: 1.5;">
                    {case['dilemma']}
                </div>
            </div>

            <!-- Consultation Questions & Interactive Feedback -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px;">
                <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 10px;">
                    👨‍⚕️ Clinical Insights for the Reviewing Physician:
                </div>
                <ul style="padding-left: 20px; margin: 0 0 16px 0;">
                    {questions_html}
                </ul>

                <!-- Physician Feedback Form -->
                <div style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 12px;">
                    <span style="font-size: 13px; font-weight: 700; color: #1e293b;">Your Clinical Verdict:</span>
                    <div style="display: flex; gap: 16px; margin: 8px 0 12px 0; flex-wrap: wrap;">
                        <label style="font-size: 13px; color: #334155; cursor: pointer; display: flex; align-items: center; gap: 6px;">
                            <input type="radio" name="verdict_{case['id']}" value="algo_correct"> <b>Algorithm is Correct</b> (Label error in dataset)
                        </label>
                        <label style="font-size: 13px; color: #334155; cursor: pointer; display: flex; align-items: center; gap: 6px;">
                            <input type="radio" name="verdict_{case['id']}" value="algo_wrong"> <b>Algorithm is Wrong</b> (Fooled by variant/artifact)
                        </label>
                        <label style="font-size: 13px; color: #334155; cursor: pointer; display: flex; align-items: center; gap: 6px;">
                            <input type="radio" name="verdict_{case['id']}" value="borderline"> <b>Equivocal / Borderline</b> (Inter-rater gray zone)
                        </label>
                    </div>
                    <textarea placeholder="Doctor's notes & anatomical observations for this case..." style="width: 100%; height: 50px; padding: 8px; font-size: 13px; border-radius: 6px; border: 1px solid #cbd5e1; box-sizing: border-box; font-family: inherit; resize: vertical;"></textarea>
                </div>
            </div>
        </div>
    </div>
    '''
    return html

# Generate all 8 case cards
all_cards_html = "".join([render_case_card_html(c) for c in CASES_DATA])

# Render full interactive clinical dashboard
full_dashboard_html = f'''
<div style="max-width: 1100px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <div style="margin-bottom: 24px; padding: 16px 20px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; color: #166534; font-size: 14px;">
        ✅ <b>Interactive Clinical Review Mode Ready:</b> Displaying all 8 failure cases across 16 authentic MRI images. Use the scorecard below or scroll down to examine each patient case in detail.
    </div>
    {all_cards_html}
</div>
'''

display(HTML(full_dashboard_html))
