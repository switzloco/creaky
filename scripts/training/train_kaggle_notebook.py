"""Self-Contained Kaggle Training Script & Notebook Generator.

Trains KneeAbnormalityClassifier on RSNA Knee Abnormality Detection dataset.
Zero external imports or GitHub cloning required.
Outputs best_model_fold_0.pt to /kaggle/working/
"""

import os
import sys
import re
import glob
import unicodedata
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

CONFIG = {
    "epochs": 5,
    "batch_size": 4,
    "num_slices": 16,
    "target_size": (256, 256),
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "label_smoothing": 0.05,
    "num_workers": 2,
    "max_train_studies": None,  # Set e.g. 500 for quick test, or None for full dataset
}


# ==============================================================================
# 2. PATH AUTODETECTION
# ==============================================================================

def get_train_paths() -> Tuple[str, str, str]:
    """Locate train.csv, train_series.csv, and train_series directory."""
    train_csv, series_csv, series_dir = None, None, None

    # 1. Search Kaggle input
    if os.path.exists("/kaggle/input"):
        print("Scanning /kaggle/input for competition data...")
        try:
            for entry in sorted(os.listdir("/kaggle/input")):
                print(f"  /kaggle/input/{entry}")
                ep = os.path.join("/kaggle/input", entry)
                if os.path.isdir(ep):
                    for sub in sorted(os.listdir(ep))[:10]:
                        print(f"    {entry}/{sub}")
        except OSError as e:
            print(f"  Error reading /kaggle/input: {e}")

        for root, dirs, files in os.walk("/kaggle/input"):
            # Prune DICOM/image folders to keep search instant
            dirs[:] = [d for d in dirs if not any(x in d.lower() for x in ["series", "image", "dicom", "dcm", "studies"])]
            if "train.csv" in files:
                train_csv = os.path.join(root, "train.csv")
                print(f"  --> Found train.csv at: {train_csv}")
                if "train_series.csv" in files:
                    series_csv = os.path.join(root, "train_series.csv")
                    print(f"  --> Found train_series.csv at: {series_csv}")
                for cand in ["train_series", "train_images", "train"]:
                    cand_p = os.path.join(root, cand)
                    if os.path.isdir(cand_p):
                        series_dir = cand_p
                        print(f"  --> Found train series dir at: {series_dir}")
                        break
                break

    # 2. Local fallback
    if train_csv is None or not os.path.exists(train_csv):
        local_root = os.path.abspath("data/raw")
        if os.path.exists(os.path.join(local_root, "train.csv")):
            train_csv = os.path.join(local_root, "train.csv")
            series_csv = os.path.join(local_root, "train_series.csv")
            series_dir = os.path.join(local_root, "sample_dicom")

    print(f"\nFinal Paths:")
    print(f"  Train CSV : {train_csv}")
    print(f"  Series CSV: {series_csv}")
    print(f"  Series Dir: {series_dir}")
    return train_csv, series_csv, series_dir


# ==============================================================================
# 3. MULTILINGUAL REGEX LABEL MINING (PURE PYTHON, 0 DEPENDENCIES)
# ==============================================================================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\r\n\t]+", " . ", text)
    text = re.sub(r"[;:]+", " . ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_accents(text: str) -> str:
    text_nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text_nfkd if not unicodedata.combining(c))

NEGATION_WORDS = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bfree of\b", r"\bnegative for\b",
    r"\babsence of\b", r"\bintact\b", r"\bunremarkable\b", r"\bnormal\b",
    r"\bpreserved\b", r"\buninjured\b", r"\bcontinuous\b", r"\bno evidence of\b",
    r"\bsin\b", r"\bausencia de\b", r"\bintacto\b", r"\bintacta\b", r"\bconservado\b",
    r"\bconservada\b", r"\brespetado\b", r"\brespetada\b", r"\bno se observa\b",
    r"\bno se aprecian\b", r"\bno hay\b", r"\bsin signos de\b", r"\bsans\b",
    r"\babsence de\b", r"\bconserve\b", r"\bconservee\b", r"\bpas de\b",
    r"\bgeen\b", r"\bzonder\b", r"\bnormaal\b", r"\bonopvallend\b",
    r"\byoktur\b", r"\bnormaldir\b", r"\byok\b", r"\bizlenmedi\b", r"\bsaptanmadi\b",
    r"\bχωρίς\b", r"\bδεν\b", r"\bφυσιολογικ", r"\bбез\b", r"\bняма\b", r"\bсъхранен"
]

HEDGE_WORDS = [
    r"cannot exclude", r"could not exclude", r"possible", r"equivocal",
    r"indeterminate", r"questionable", r"borderline", r"no descartable",
    r"dudoso", r"no excluible", r"pequena duda", r"non exclu", r"şüpheli",
    r"suspect", r"suspected", r"sospecha"
]

class MultilingualRegexLabeler:
    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        acl_terms = r"(acl|lca|voorste kruisband|croise anterieur|anterior cruciate|ligamento cruzado anterior|anterior capraz|χιαστό|кръстни връзки)"
        tear_terms = r"(tear|ruptur|rotur|rupture|torn|disrupt|injury|avulsion|scheur|lesion|lacerat|ρήξη|разкъсване|фрактура)"
        intact_terms = r"(intact|normal|conservad|unremarkable|sin alteracion|sans particularite|behoud|integrite|normaldir|φυσιολογικ|съхранен)"

        self.acl_pos = re.compile(rf"\b{acl_terms}\b.{{0,60}}?{tear_terms}", re.IGNORECASE)
        self.acl_pos_rev = re.compile(rf"{tear_terms}.{{0,60}}?\b{acl_terms}\b", re.IGNORECASE)
        self.acl_neg = re.compile(rf"\b{acl_terms}\b.{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.acl_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b{acl_terms}\b", re.IGNORECASE)

        mcl_terms = r"(mcl|lcm|medial collateral|ligamento colateral medial|mediale band|collat[eé]ral m[eé]dial|medial kollateral|πλάγιοι|коллатерал)"
        sprain_terms = r"(tear|ruptur|rotur|rupture|torn|disrupt|sprain|lesion|strain|esguince|entorse|injury|scheur|incelme|ödem|ρήξη)"
        self.mcl_pos = re.compile(rf"\b{mcl_terms}\b.{{0,60}}?{sprain_terms}", re.IGNORECASE)
        self.mcl_pos_rev = re.compile(rf"{sprain_terms}.{{0,60}}?\b{mcl_terms}\b", re.IGNORECASE)
        self.mcl_neg = re.compile(rf"\b{mcl_terms}\b.{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.mcl_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b{mcl_terms}\b", re.IGNORECASE)

        mm_terms = r"(medial meniscus|menisco medial|menisco interno|binnenmeniscus|menisque medial|menisque interne|medial menisk|έσω διαμέρισμα|медиалния менискус|mm\b)"
        meniscus_tear = r"(tear|ruptur|rotur|rupture|torn|fissur|lesion|amputat|flap|scheur|cleavage|meniskopati|ρήξη|дегенерация|разкъсване|rotura)"
        self.mm_pos = re.compile(rf"\b{mm_terms}.{{0,60}}?{meniscus_tear}", re.IGNORECASE)
        self.mm_pos_rev = re.compile(rf"{meniscus_tear}.{{0,60}}?\b{mm_terms}", re.IGNORECASE)
        self.mm_neg = re.compile(rf"\b({mm_terms}|meniscos? (medial|interno|ambos|los dos)).{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.mm_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b({mm_terms}|meniscos?)", re.IGNORECASE)

        lm_terms = r"(lateral meniscus|menisco lateral|menisco externo|buitenmeniscus|menisque lateral|menisque externe|lateral menisk|έξω διαμέρισμα|латералния менискус|lm\b)"
        self.lm_pos = re.compile(rf"\b{lm_terms}.{{0,60}}?{meniscus_tear}", re.IGNORECASE)
        self.lm_pos_rev = re.compile(rf"{meniscus_tear}.{{0,60}}?\b{lm_terms}", re.IGNORECASE)
        self.lm_neg = re.compile(rf"\b({lm_terms}|meniscos? (lateral|externo|ambos|los dos)|medial y lateral).{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.lm_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b({lm_terms}|meniscos?|medial y lateral)", re.IGNORECASE)

        oa_signs = r"(osteoarthritis|oa|arthrosis|artrosis|gonartrosis|gonarthrose|chondromalacia|chondropathy|condropatia|chondrosis|cartilage loss|cartilage defect|ulcer|pinzamiento|thinning|wear|οστεοαρθρίτιδα|хрущял)"
        moa_terms = r"(medial|femorotibial medial|compartimento medial|compartiment medial|binnenste compartiment|condilo femoral medial|plateau tibial medial|έσω κνημιαίο)"
        self.moa_pos = re.compile(rf"\b{moa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.moa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{moa_terms}", re.IGNORECASE)
        self.moa_neg = re.compile(rf"\b{moa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin artrosis)", re.IGNORECASE)

        loa_terms = r"(lateral|femorotibial lateral|compartimento lateral|compartiment lateral|buitenste compartiment|condilo femoral lateral|plateau tibial lateral|έξω κνημιαίο)"
        self.loa_pos = re.compile(rf"\b{loa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.loa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{loa_terms}", re.IGNORECASE)
        self.loa_neg = re.compile(rf"\b{loa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin artrosis)", re.IGNORECASE)

        pfoa_terms = r"(patellofemoral|femoropatellar|femororrotuliana|femoropatelar|patellar cartilage|trochlear cartilage|rotuliana|rotula|patella|retropatellaire|patelofemoral|патела|trochlea)"
        self.pfoa_pos = re.compile(rf"\b{pfoa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.pfoa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{pfoa_terms}", re.IGNORECASE)
        self.pfoa_neg = re.compile(rf"\b{pfoa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin alteracion)", re.IGNORECASE)

        self.eff_pos = re.compile(r"\b(joint effusion|effusion|derrame|derrame articular|epanchement|hydrops|vochtuitstorting|fluid accumulation|fluid in the joint|liquid articular|sıvı artışı|υγρού|излив|bursitis|hidrartrosis)\b", re.IGNORECASE)
        self.eff_neg = re.compile(r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(joint effusion|effusion|derrame|epanchement|hydrops|fluid|sıvı|υγρού|излив)\b", re.IGNORECASE)

        self.syn_pos = re.compile(r"\b(synovitis|sinovitis|synoviale|synovial thickening|synovial hypertrophy|synovial enhancement|engrosamiento sinovial|hipertrofia sinovial|epaississement synovial|hoffitis|sinovit|συνοβιακ)\b", re.IGNORECASE)
        self.syn_neg = re.compile(r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(synovitis|sinovitis|synovial thickening|synovial hypertrophy|sinovit)\b", re.IGNORECASE)

        self.baker_pos = re.compile(r"\b(baker|bakers|popliteal cyst|quiste de baker|quiste popliteo|kyste poplite|kyste de baker|bakerse cyste|popliteale cyste|baker kisti|κύστη baker)\b", re.IGNORECASE)
        self.baker_neg = re.compile(r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(baker|bakers|popliteal cyst|quiste de baker|quiste popliteo|kyste poplite)\b", re.IGNORECASE)

        self.cont_pos = re.compile(r"\b(bone contusion|bone bruise|contusion osea|contusion osseuse|botcontusie|marrow edema|edema oseo|edema medular|bone marrow edema|oedeme osseux|osseous contusion|osteocondral contusion|kemik iliği ödem|οστεομυελικό οίδημα|костномозъчен едем|kissing contusion|contusion|contusiones)\b", re.IGNORECASE)
        self.cont_neg = re.compile(r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(bone contusion|bone bruise|contusion osea|marrow edema|edema oseo|edema medular|bone marrow edema)\b", re.IGNORECASE)

        self.frac_pos = re.compile(r"\b(fracture|fractura|fractur|fraktür|break|insufficiency fracture|subchondral fracture|trabecular fracture|fracture de fatigue|κατάγματα|фрактура)\b", re.IGNORECASE)
        self.frac_neg = re.compile(r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(fracture|fractura|fractur|fraktür|break|κατάγματα|фрактура)\b", re.IGNORECASE)

    def extract_single(self, text: str, pos_re, neg_re, pos_rev_re=None, neg_rev_re=None) -> Optional[float]:
        sentences = re.split(r"[.\n;•\->]+", text)
        has_pos, has_neg = False, False
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            clean_s = strip_accents(s)
            if neg_re and neg_re.search(clean_s):
                has_neg = True
            elif neg_rev_re and neg_rev_re.search(clean_s):
                has_neg = True

            matched_pos = False
            if pos_re and pos_re.search(clean_s):
                matched_pos = True
            elif pos_rev_re and pos_rev_re.search(clean_s):
                matched_pos = True

            if matched_pos:
                hedge_p = r"(" + "|".join(HEDGE_WORDS) + r")"
                neg_p = r"(" + "|".join(NEGATION_WORDS) + r")"
                if re.search(hedge_p, clean_s, re.IGNORECASE) or re.search(neg_p, clean_s, re.IGNORECASE):
                    continue
                has_pos = True

        if has_pos:
            return 1.0
        elif has_neg:
            return 0.0
        return None

    def extract_from_report(self, report_text: str) -> Dict[str, Optional[float]]:
        norm = normalize_text(report_text)
        if not norm:
            return {t: None for t in TARGET_COLS}
        return {
            "ACL": self.extract_single(norm, self.acl_pos, self.acl_neg, self.acl_pos_rev, self.acl_neg_rev),
            "MCL": self.extract_single(norm, self.mcl_pos, self.mcl_neg, self.mcl_pos_rev, self.mcl_neg_rev),
            "Medial Meniscus": self.extract_single(norm, self.mm_pos, self.mm_neg, self.mm_pos_rev, self.mm_neg_rev),
            "Lateral Meniscus": self.extract_single(norm, self.lm_pos, self.lm_neg, self.lm_pos_rev, self.lm_neg_rev),
            "Medial OA": self.extract_single(norm, self.moa_pos, self.moa_neg, self.moa_pos_rev),
            "Lateral OA": self.extract_single(norm, self.loa_pos, self.loa_neg, self.loa_pos_rev),
            "PF OA": self.extract_single(norm, self.pfoa_pos, self.pfoa_neg, self.pfoa_pos_rev),
            "Effusion": self.extract_single(norm, self.eff_pos, self.eff_neg),
            "Synovitis": self.extract_single(norm, self.syn_pos, self.syn_neg),
            "Baker's": self.extract_single(norm, self.baker_pos, self.baker_neg),
            "Contusion": self.extract_single(norm, self.cont_pos, self.cont_neg),
            "Fracture": self.extract_single(norm, self.frac_pos, self.frac_neg),
        }


def assemble_labels(train_csv_path: str) -> pd.DataFrame:
    """Build train labels DataFrame with Gold prioritized, then Regex, then soft 0.5."""
    df = pd.read_csv(train_csv_path)
    labeler = MultilingualRegexLabeler()
    print(f"Assembling labels for {len(df)} studies...")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mining Labels"):
        uid = str(row["StudyInstanceUID"])
        out_row = {"StudyInstanceUID": uid}

        # Check if expert gold annotated
        is_gold = not pd.isna(row["ACL"])
        reg_preds = labeler.extract_from_report(row.get("Report", "")) if not is_gold else {}

        gold_or_regex = False
        for t in TARGET_COLS:
            if is_gold and not pd.isna(row.get(t)):
                out_row[t] = float(row[t])
                gold_or_regex = True
            elif t in reg_preds and reg_preds[t] is not None:
                out_row[t] = float(reg_preds[t])
                gold_or_regex = True
            else:
                out_row[t] = 0.5

        out_row["is_gold"] = is_gold
        out_row["label_source"] = "gold" if is_gold else ("regex" if gold_or_regex else "soft_unk")
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    gold_count = out_df["is_gold"].sum()
    print(f"--> Labels compiled: {len(out_df)} studies | {gold_count} Gold | {(out_df['label_source']=='regex').sum()} Regex")
    return out_df


# ==============================================================================
# 4. DICOM PREPROCESSING (ON-THE-FLY)
# ==============================================================================

def apply_windowing(pixels: np.ndarray, center=None, width=None, photometric: str = "MONOCHROME2") -> np.ndarray:
    img = pixels.astype(np.float32)
    if photometric == "MONOCHROME1":
        img = img.max() - img
    if isinstance(center, (list, pydicom.multival.MultiValue)):
        center = float(center[0])
    if isinstance(width, (list, pydicom.multival.MultiValue)):
        width = float(width[0])

    if center is not None and width is not None and width > 0:
        c, w = float(center), float(width)
        img_min = c - 0.5 - (w - 1) / 2.0
        img_max = c - 0.5 + (w - 1) / 2.0
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        vmin, vmax = np.percentile(img, 1.0), np.percentile(img, 99.0)
        if vmax > vmin:
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin) * 255.0
        else:
            img = np.zeros_like(img)
    return img.astype(np.uint8)

def crop_empty_borders(img: np.ndarray, threshold: int = 10, margin: int = 5) -> np.ndarray:
    mask = img > threshold
    if not np.any(mask):
        return img
    y_idx, x_idx = np.where(mask)
    h, w = img.shape[:2]
    ymin = max(0, y_idx.min() - margin)
    ymax = min(h, y_idx.max() + margin + 1)
    xmin = max(0, x_idx.min() - margin)
    xmax = min(w, x_idx.max() + margin + 1)
    return img[ymin:ymax, xmin:xmax]

def compute_slice_position(dcm: pydicom.Dataset) -> float:
    ori = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    pos = getattr(dcm, "ImagePositionPatient", [0, 0, 0])
    f = np.array(ori[:3], dtype=float)
    c = np.array(ori[3:], dtype=float)
    normal = np.cross(f, c)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    return float(np.dot(normal, np.array(pos, dtype=float)))

def load_and_preprocess_series(dicom_files: List[str], target_size=(256, 256), num_slices: int = 16) -> torch.Tensor:
    if not dicom_files:
        return torch.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=torch.float32)

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
            img = apply_windowing(pixels, center=wc, width=ww, photometric=photometric)
            img = crop_empty_borders(img)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            slice_entries.append((coord, img))
        except Exception:
            continue

    if not slice_entries:
        return torch.zeros((num_slices, 3, target_size[0], target_size[1]), dtype=torch.float32)

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

    return torch.tensor(np.stack(slabs, axis=0), dtype=torch.float32)


# ==============================================================================
# 5. PYTORCH DATASET
# ==============================================================================

class KneeMRITrainingDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, series_df: pd.DataFrame, series_dir: str, num_slices: int = 16):
        self.labels_df = labels_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_dir = series_dir
        self.num_slices = num_slices
        self.plane_col = None
        if not series_df.empty:
            for col in ["Anatomical_Plane", "anatomical_plane", "Plane", "SeriesDescription"]:
                if col in series_df.columns:
                    self.plane_col = col
                    break

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])

        study_series = self.series_df[self.series_df["StudyInstanceUID"] == study_uid] if not self.series_df.empty else pd.DataFrame()

        plane_tensors = {}
        for plane in ["Sagittal", "Coronal", "Axial"]:
            series_files = []
            if not study_series.empty and self.plane_col is not None:
                match = study_series[study_series[self.plane_col].astype(str).str.lower().str.contains(plane.lower())]
                if not match.empty:
                    series_uid = str(match.iloc[0]["SeriesInstanceUID"])
                    pattern = os.path.join(self.series_dir, study_uid, series_uid, "*.dcm")
                    series_files = glob.glob(pattern)
            elif os.path.isdir(self.series_dir):
                s_dir = os.path.join(self.series_dir, study_uid)
                if os.path.isdir(s_dir):
                    subdirs = [d for d in os.listdir(s_dir) if os.path.isdir(os.path.join(s_dir, d))]
                    plane_idx = ["Sagittal", "Coronal", "Axial"].index(plane)
                    if plane_idx < len(subdirs):
                        series_files = glob.glob(os.path.join(s_dir, subdirs[plane_idx], "*.dcm"))

            plane_tensors[plane] = load_and_preprocess_series(series_files, target_size=(256, 256), num_slices=self.num_slices)

        targets = np.array([float(row[t]) for t in TARGET_COLS], dtype=np.float32)
        weight_val = 1.0 if row.get("label_source") in ["gold", "regex"] else 0.5
        weights = np.full_like(targets, weight_val, dtype=np.float32)

        return {
            "study_uid": study_uid,
            "sagittal": plane_tensors["Sagittal"],
            "coronal": plane_tensors["Coronal"],
            "axial": plane_tensors["Axial"],
            "targets": torch.tensor(targets, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32)
        }


# ==============================================================================
# 6. MODEL ARCHITECTURE (MATCHES INFERENCE 100%)
# ==============================================================================

class GatedAttentionPool(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        attn_scores = self.w_proj(v * u)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

class KneeAbnormalityClassifier(nn.Module):
    def __init__(self, feat_dim: int = 512, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.sag_pool = GatedAttentionPool(feat_dim)
        self.cor_pool = GatedAttentionPool(feat_dim)
        self.ax_pool = GatedAttentionPool(feat_dim)

        combined_dim = feat_dim * 3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _encode_plane(self, x: torch.Tensor, pool_module: GatedAttentionPool) -> torch.Tensor:
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = self.encoder(x_flat).flatten(1)
        feats = feats.view(B, K, self.feat_dim)
        return pool_module(feats)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        h_sag = self._encode_plane(sag, self.sag_pool)
        h_cor = self._encode_plane(cor, self.cor_pool)
        h_ax = self._encode_plane(ax, self.ax_pool)
        combined = torch.cat([h_sag, h_cor, h_ax], dim=1)
        return self.head(combined)


# ==============================================================================
# 7. LOSS & METRIC
# ==============================================================================

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.label_smoothing > 0:
            smoothed = targets * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
        else:
            smoothed = targets
        bce = F.binary_cross_entropy_with_logits(logits, smoothed, reduction="none")
        if weights is not None:
            return (bce * weights).sum() / (weights.sum() + 1e-7)
        return bce.mean()

def compute_competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    per_class = {}
    valid_aucs = []
    for i, col in enumerate(TARGET_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        # Only evaluate AUC on definitive binary ground-truth (0.0 or 1.0), excluding soft 0.5
        mask = (yt == 0.0) | (yt == 1.0)
        yt_bin = yt[mask]
        yp_bin = yp[mask]
        classes = np.unique(yt_bin)
        if len(classes) < 2:
            auc = 0.5
        else:
            try:
                auc = float(roc_auc_score(yt_bin, yp_bin))
            except Exception:
                auc = 0.5
        per_class[col] = float(auc)
        valid_aucs.append(auc)
    return float(np.mean(valid_aucs)), per_class


# ==============================================================================
# 8. MASTER TRAINING PIPELINE
# ==============================================================================

def run_training():
    train_csv, series_csv, series_dir = get_train_paths()
    if not train_csv or not os.path.exists(train_csv):
        raise FileNotFoundError("Could not locate train.csv in candidate paths.")

    # 1. Labels
    labels_df = assemble_labels(train_csv)
    series_df = pd.read_csv(series_csv) if (series_csv and os.path.exists(series_csv)) else pd.DataFrame()

    # 2. Validation Split: Keep all 58 Gold studies in Validation for true ground-truth tracking!
    val_gold = labels_df[labels_df["is_gold"] == True]
    non_gold = labels_df[labels_df["is_gold"] == False].sample(frac=1.0, random_state=42)

    val_non_gold = non_gold.iloc[:int(len(non_gold) * 0.1)]
    train_df = non_gold.iloc[int(len(non_gold) * 0.1):]
    val_df = pd.concat([val_gold, val_non_gold], ignore_index=True)

    if CONFIG["max_train_studies"] is not None:
        train_df = train_df.iloc[:CONFIG["max_train_studies"]]

    print(f"\nDataset Splits:")
    print(f"  Training Studies  : {len(train_df)}")
    print(f"  Validation Studies: {len(val_df)} (including {len(val_gold)} Gold Standard)")

    # 3. DataLoaders
    train_ds = KneeMRITrainingDataset(train_df, series_df, series_dir, num_slices=CONFIG["num_slices"])
    val_ds = KneeMRITrainingDataset(val_df, series_df, series_dir, num_slices=CONFIG["num_slices"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 4. Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on Device: {device}")
    model = KneeAbnormalityClassifier(dropout=CONFIG["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    criterion = WeightedBCEWithLogitsLoss(label_smoothing=CONFIG["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model_fold_0.pt")

    # 5. Epoch Loop
    print("\nStarting Training Loop...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Train]")
        for batch in pbar:
            sag = batch["sagittal"].to(device)
            cor = batch["coronal"].to(device)
            ax = batch["axial"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(sag, cor, ax)
                loss = criterion(logits, targets, weights=weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Val]"):
                sag = batch["sagittal"].to(device)
                cor = batch["coronal"].to(device)
                ax = batch["axial"].to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(sag, cor, ax)
                    probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(batch["targets"].cpu().numpy())

        y_true = np.vstack(val_targets)
        y_pred = np.vstack(val_preds)
        val_auc, per_class = compute_competition_metric(y_true, y_pred)

        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} Summary ---")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val AUC    : {val_auc:.4f} (Best: {best_val_auc:.4f})")
        for t in TARGET_COLS:
            print(f"    {t:18s}: {per_class[t]:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "per_class_auc": per_class,
                "config": CONFIG
            }, best_ckpt_path)
            print(f"  >>> Saved NEW BEST model to: {best_ckpt_path} (Val AUC: {best_val_auc:.4f})")

    print(f"\nTraining Complete! Best Validation Macro AUC: {best_val_auc:.4f}")
    print(f"Saved Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    run_training()
