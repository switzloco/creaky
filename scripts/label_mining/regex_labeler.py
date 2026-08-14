"""Multilingual Regex Labeler for 12 knee abnormalities across international clinical sites.

Supports English, Spanish, French, Dutch, German, Turkish, Greek, and Cyrillic/Bulgarian medical terminology.
Extracts 12 targets:
1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture
"""

import re
import unicodedata
from typing import Dict, Optional, List
import pandas as pd
import numpy as np


TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


def normalize_text(text: str) -> str:
    """Normalize text while preserving non-Latin scripts (Greek, Cyrillic)."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode to NFKC (preserves Greek/Cyrillic characters, normalizes ligatures)
    text = unicodedata.normalize("NFKC", text).lower()
    # Replace line breaks and multiple separators
    text = re.sub(r"[\r\n\t]+", " . ", text)
    text = re.sub(r"[;:]+", " . ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(text: str) -> str:
    """Strip latin accents for robust latin-script matching."""
    text_nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text_nfkd if not unicodedata.combining(c))


# Multilingual negation markers
NEGATION_WORDS = [
    # English
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bfree of\b", r"\bnegative for\b",
    r"\babsence of\b", r"\bintact\b", r"\bunremarkable\b", r"\bnormal\b",
    r"\bpreserved\b", r"\buninjured\b", r"\bcontinuous\b", r"\bno evidence of\b",
    # Spanish
    r"\bsin\b", r"\bausencia de\b", r"\bintacto\b", r"\bintacta\b", r"\bconservado\b",
    r"\bconservada\b", r"\bnormal\b", r"\brespetado\b", r"\brespetada\b",
    r"\bno se observa\b", r"\bno se aprecian\b", r"\bno hay\b", r"\bsin signos de\b",
    # French
    r"\bsans\b", r"\babsence de\b", r"\bintact\b", r"\bintacte\b", r"\bnormal\b",
    r"\bnormale\b", r"\bconserve\b", r"\bconservee\b", r"\bpas de\b", r"\bintegrite\b",
    # Dutch
    r"\bgeen\b", r"\bzonder\b", r"\bintact\b", r"\bnormaal\b", r"\bonopvallend\b",
    # Turkish
    r"\byoktur\b", r"\bnormaldir\b", r"\byok\b", r"\bizlenmedi\b", r"\bsaptanmadi\b",
    # Greek
    r"\bχωρίς\b", r"\bδεν\b", r"\bφυσιολογικ",
    # Cyrillic
    r"\bбез\b", r"\bняма\b", r"\bсъхранен", r"\bнормално\b"
]

# Multilingual Hedges
HEDGE_WORDS = [
    r"cannot exclude", r"could not exclude", r"possible", r"equivocal",
    r"indeterminate", r"questionable", r"borderline", r"no descartable",
    r"dudoso", r"no excluible", r"pequena duda", r"non exclu", r"şüpheli",
    r"suspect", r"suspected", r"sospecha"
]


class RegexLabeler:
    """Multilingual extractor for the 12 knee abnormality targets."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. ACL
        acl_terms = r"(acl|lca|voorste kruisband|croise anterieur|croises anterieurs|anterior cruciate|ligamento cruzado anterior|anterior capraz|χιαστό|кръстни връзки|anterior cruciate ligament)"
        tear_terms = r"(tear|ruptur|rotur|rupture|torn|disrupt|injury|avulsion|rupture|scheur|lesion|lacerat|bütünlük kaybı|ρήξη|разкъсване|фрактура)"
        intact_terms = r"(intact|normal|conservad|unremarkable|sin alteracion|sans particularite|behoud|integrite|normaldir|φυσιολογικ|съхранен)"

        self.acl_pos = re.compile(rf"\b{acl_terms}\b.{{0,60}}?{tear_terms}", re.IGNORECASE)
        self.acl_pos_rev = re.compile(rf"{tear_terms}.{{0,60}}?\b{acl_terms}\b", re.IGNORECASE)
        self.acl_neg = re.compile(rf"\b{acl_terms}\b.{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.acl_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b{acl_terms}\b", re.IGNORECASE)

        # 2. MCL
        mcl_terms = r"(mcl|lcm|medial collateral|ligamento colateral medial|ligamento colateral interno|mediale band|collat[eé]ral m[eé]dial|collat[eé]ral interne|medial kollateral|πλάγιοι|коллатерал)"
        sprain_terms = r"(tear|ruptur|rotur|rupture|torn|disrupt|sprain|lesion|strain|esguince|entorse|injury|scheur|incelme|ödem|ρήξη)"

        self.mcl_pos = re.compile(rf"\b{mcl_terms}\b.{{0,60}}?{sprain_terms}", re.IGNORECASE)
        self.mcl_pos_rev = re.compile(rf"{sprain_terms}.{{0,60}}?\b{mcl_terms}\b", re.IGNORECASE)
        self.mcl_neg = re.compile(rf"\b{mcl_terms}\b.{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.mcl_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b{mcl_terms}\b", re.IGNORECASE)

        # 3. Medial Meniscus
        mm_terms = r"(medial meniscus|menisco medial|menisco interno|meniscos medial|binnenmeniscus|menisque medial|menisque interne|medial menisk|έσω διαμέρισμα|έσω κνημια|медиалния менискус|medial menisküs|mm\b)"
        meniscus_tear = r"(tear|ruptur|rotur|rupture|torn|fissur|lesion|amputat|flap|scheur|cleavage|meniskopati|ρήξη|дегенерация|разкъсване|rotura)"

        self.mm_pos = re.compile(rf"\b{mm_terms}.{{0,60}}?{meniscus_tear}", re.IGNORECASE)
        self.mm_pos_rev = re.compile(rf"{meniscus_tear}.{{0,60}}?\b{mm_terms}", re.IGNORECASE)
        self.mm_neg = re.compile(rf"\b({mm_terms}|meniscos? (medial|interno|ambos|los dos)).{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.mm_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b({mm_terms}|meniscos?)", re.IGNORECASE)

        # 4. Lateral Meniscus
        lm_terms = r"(lateral meniscus|menisco lateral|menisco externo|meniscos? lateral|meniscos? externo|buitenmeniscus|menisque lateral|menisque externe|lateral menisk|έξω διαμέρισμα|латералния менискус|lateral menisküs|lm\b)"

        self.lm_pos = re.compile(rf"\b{lm_terms}.{{0,60}}?{meniscus_tear}", re.IGNORECASE)
        self.lm_pos_rev = re.compile(rf"{meniscus_tear}.{{0,60}}?\b{lm_terms}", re.IGNORECASE)
        self.lm_neg = re.compile(rf"\b({lm_terms}|meniscos? (lateral|externo|ambos|los dos)|medial y lateral).{{0,40}}?{intact_terms}", re.IGNORECASE)
        self.lm_neg_rev = re.compile(rf"(no|sin|sans|geen|yok|без|няма|{intact_terms}).{{0,40}}?\b({lm_terms}|meniscos?|medial y lateral)", re.IGNORECASE)


        # 5. Medial OA
        moa_terms = r"(medial|femorotibial medial|compartimento medial|compartiment medial|binnenste compartiment|condilo femoral medial|plateau tibial medial|platillo tibial medial|έσω κνημιαίο|медиалното тибиално)"
        oa_signs = r"(osteoarthritis|oa|arthrosis|artrosis|gonartrosis|gonarthrose|chondromalacia|chondropathy|condropatia|chondrosis|cartilage loss|cartilage defect|ulcer|pinzamiento|thinning|wear|οστεοαρθρίτιδα|хрущял|дефект)"

        self.moa_pos = re.compile(rf"\b{moa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.moa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{moa_terms}", re.IGNORECASE)
        self.moa_neg = re.compile(rf"\b{moa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin artrosis)", re.IGNORECASE)

        # 6. Lateral OA
        loa_terms = r"(lateral|femorotibial lateral|compartimento lateral|compartiment lateral|buitenste compartiment|condilo femoral lateral|plateau tibial lateral|platillo tibial lateral|έξω κνημιαίο)"

        self.loa_pos = re.compile(rf"\b{loa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.loa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{loa_terms}", re.IGNORECASE)
        self.loa_neg = re.compile(rf"\b{loa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin artrosis)", re.IGNORECASE)

        # 7. PF OA (Patellofemoral OA / Chondromalacia / Cartilage thinning)
        pfoa_terms = r"(patellofemoral|femoropatellar|femororrotuliana|femoropatelar|patellar cartilage|trochlear cartilage|rotuliana|rotula|patella|retropatellaire|patelofemoral|патела|trochlea|troclea)"

        self.pfoa_pos = re.compile(rf"\b{pfoa_terms}.{{0,60}}?{oa_signs}", re.IGNORECASE)
        self.pfoa_pos_rev = re.compile(rf"{oa_signs}.{{0,60}}?\b{pfoa_terms}", re.IGNORECASE)
        self.pfoa_neg = re.compile(rf"\b{pfoa_terms}.{{0,40}}?(normal|intact|preserved|conservad|no chondral|sin alteracion)", re.IGNORECASE)

        # 8. Effusion (Joint fluid / effusion / derrame / liquid / hydrops / hydrarthrose)
        self.eff_pos = re.compile(
            r"\b(joint effusion|effusion|derrame|derrame articular|epanchement|hydrops|vochtuitstorting|fluid accumulation|fluid in the joint|excess fluid|liquid articular|sıvı artışı|υγρού|излив|bursitis|hidrartrosis)\b",
            re.IGNORECASE
        )
        self.eff_neg = re.compile(
            r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(joint effusion|effusion|derrame|epanchement|hydrops|fluid|sıvı|υγρού|излив)\b",
            re.IGNORECASE
        )

        # 9. Synovitis (Synovial thickening / synovitis / hoffitis / plica)
        self.syn_pos = re.compile(
            r"\b(synovitis|sinovitis|synoviale|synovial thickening|synovial hypertrophy|synovial enhancement|engrosamiento sinovial|hipertrofia sinovial|epaississement synovial|hoffitis|synovial proliferation|sinovit|συνοβιακ)\b",
            re.IGNORECASE
        )
        self.syn_neg = re.compile(
            r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(synovitis|sinovitis|synovial thickening|synovial hypertrophy|sinovit)\b",
            re.IGNORECASE
        )

        # 10. Baker's Cyst
        self.baker_pos = re.compile(
            r"\b(baker|bakers|popliteal cyst|quiste de baker|quiste popliteo|kyste poplite|kyste de baker|bakerse cyste|popliteale cyste|baker kisti|κύστη baker)\b",
            re.IGNORECASE
        )
        self.baker_neg = re.compile(
            r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(baker|bakers|popliteal cyst|quiste de baker|quiste popliteo|kyste poplite)\b",
            re.IGNORECASE
        )

        # 11. Contusion (Bone bruise / marrow edema / contusion osea / kissing contusion)
        self.cont_pos = re.compile(
            r"\b(bone contusion|bone bruise|contusion osea|contusion osseuse|botcontusie|marrow edema|edema oseo|edema medular|bone marrow edema|oedeme osseux|osseous contusion|osteocondral contusion|kemik iliği ödem|οστεομυελικό οίδημα|костномозъчен едем|kissing contusion|contusion|contusiones)\b",
            re.IGNORECASE
        )
        self.cont_neg = re.compile(
            r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(bone contusion|bone bruise|contusion osea|marrow edema|edema oseo|edema medular|bone marrow edema)\b",
            re.IGNORECASE
        )

        # 12. Fracture (Fracture / fractura / fraktur / microfracture)
        self.frac_pos = re.compile(
            r"\b(fracture|fractura|fractur|fraktür|break|insufficiency fracture|subchondral fracture|trabecular fracture|fracture de fatigue|κατάγματα|фрактура)\b",
            re.IGNORECASE
        )
        self.frac_neg = re.compile(
            r"\b(no|not|sin|sans|geen|yok|без|няма|without|ausencia de).{0,30}?\b(fracture|fractura|fractur|fraktür|break|κατάγματα|фрактура)\b",
            re.IGNORECASE
        )

    def extract_single_finding(self, text: str, pos_re, neg_re, pos_rev_re=None, neg_rev_re=None) -> Optional[float]:
        """Extract finding state from normalized text."""
        # Split on sentence/bullet bounds
        sentences = re.split(r"[.\n;•\->]+", text)
        has_pos = False
        has_neg = False

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Strip latin accents for regex matching while preserving Greek/Cyrillic
            clean_s = strip_accents(sentence)

            # Check for negative patterns
            if neg_re and neg_re.search(clean_s):
                has_neg = True
            elif neg_rev_re and neg_rev_re.search(clean_s):
                has_neg = True

            # Check for positive patterns
            matched_pos = False
            if pos_re and pos_re.search(clean_s):
                matched_pos = True
            elif pos_rev_re and pos_rev_re.search(clean_s):
                matched_pos = True

            if matched_pos:
                # Check for hedges / negation in the sentence
                neg_words_pattern = r"(" + "|".join(NEGATION_WORDS) + r")"
                hedge_pattern = r"(" + "|".join(HEDGE_WORDS) + r")"

                if re.search(hedge_pattern, clean_s, re.IGNORECASE):
                    # Ambiguous/hedged statement -> don't confirm positive
                    continue

                if re.search(neg_words_pattern, clean_s, re.IGNORECASE):
                    # Check if negation was matched
                    pass
                else:
                    has_pos = True

        if has_pos:
            return 1.0
        elif has_neg:
            return 0.0
        return None

    def extract_from_report(self, report_text: str) -> Dict[str, Optional[float]]:
        """Extract all 12 target values from a radiology report."""
        norm = normalize_text(report_text)
        if not norm:
            return {target: None for target in TARGET_COLS}

        results = {}
        results["ACL"] = self.extract_single_finding(norm, self.acl_pos, self.acl_neg, self.acl_pos_rev, self.acl_neg_rev)
        results["MCL"] = self.extract_single_finding(norm, self.mcl_pos, self.mcl_neg, self.mcl_pos_rev, self.mcl_neg_rev)
        results["Medial Meniscus"] = self.extract_single_finding(norm, self.mm_pos, self.mm_neg, self.mm_pos_rev, self.mm_neg_rev)
        results["Lateral Meniscus"] = self.extract_single_finding(norm, self.lm_pos, self.lm_neg, self.lm_pos_rev, self.lm_neg_rev)
        results["Medial OA"] = self.extract_single_finding(norm, self.moa_pos, self.moa_neg, self.moa_pos_rev, None)
        results["Lateral OA"] = self.extract_single_finding(norm, self.loa_pos, self.loa_neg, self.loa_pos_rev, None)
        results["PF OA"] = self.extract_single_finding(norm, self.pfoa_pos, self.pfoa_neg, self.pfoa_pos_rev, None)
        results["Effusion"] = self.extract_single_finding(norm, self.eff_pos, self.eff_neg, None, None)
        results["Synovitis"] = self.extract_single_finding(norm, self.syn_pos, self.syn_neg, None, None)
        results["Baker's"] = self.extract_single_finding(norm, self.baker_pos, self.baker_neg, None, None)
        results["Contusion"] = self.extract_single_finding(norm, self.cont_pos, self.cont_neg, None, None)
        results["Fracture"] = self.extract_single_finding(norm, self.frac_pos, self.frac_neg, None, None)
        return results

    def label_dataframe(self, df: pd.DataFrame, report_col: str = "Report") -> pd.DataFrame:
        """Run regex labeler across all rows in a DataFrame."""
        extracted_rows = []
        for _, row in df.iterrows():
            report_text = row.get(report_col, "")
            res = self.extract_from_report(report_text)
            res["StudyInstanceUID"] = row["StudyInstanceUID"]
            extracted_rows.append(res)

        out_df = pd.DataFrame(extracted_rows)
        cols = ["StudyInstanceUID"] + TARGET_COLS
        return out_df[cols]
