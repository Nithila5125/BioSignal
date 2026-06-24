# src/nlp_pipeline.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 5: NLP Pipeline (Final)
#
# Uses REAL text (title + summary) when available.
# Falls back to neutral structured text when not.
# Never inflates NLP scores with artificial keywords.
#
# HOW TO RUN:
#   python src/nlp_pipeline.py
# ═══════════════════════════════════════════════════════════════

import re
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("biosignal.nlp")


# ═══════════════════════════════════════════════════════════════
# KEYWORD LISTS
# ═══════════════════════════════════════════════════════════════

URGENCY_WORDS = [
    "outbreak", "epidemic", "pandemic", "emergency", "death",
    "deaths", "fatal", "fatality", "severe", "surge", "spread",
    "spreading", "alert", "warning", "crisis", "escalation",
    "uncontrolled", "confirmed cases", "suspected cases",
    "public health emergency", "critical", "killed", "rapidly",
]

CONTAINMENT_WORDS = [
    "contained", "controlled", "declining", "reduced", "recovered",
    "vaccinated", "prevention", "response plan", "surveillance",
    "preparedness", "treatment", "isolation", "contact tracing",
    "risk communication", "stable", "improving", "eliminated",
    "eradicated", "resolved", "ended",
]

RISK_KEYWORDS = [
    "risk", "threat", "danger", "vulnerable", "exposure",
    "transmission", "infectious", "contagious", "mortality",
    "morbidity", "case fatality", "attack rate",
]

RESPONSE_KEYWORDS = [
    "response", "deployed", "intervention", "vaccination",
    "campaign", "investigation", "team", "support", "aid",
    "relief", "coordination", "measures", "protocol",
]


def keyword_score(text: str, keywords: list) -> int:
    """
    Counts keyword matches using regex word boundaries.
    Returns 0 for empty text.
    """
    if not text:
        return 0
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if " " in kw:
            score += len(re.findall(re.escape(kw), text_lower))
        else:
            score += len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
    return score


# ═══════════════════════════════════════════════════════════════
# LOAD SPACY
# ═══════════════════════════════════════════════════════════════

def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        log.info("  spaCy model loaded: en_core_web_sm")
        return nlp
    except OSError:
        log.error("  spaCy model not found.")
        log.error("  Run: python -m spacy download en_core_web_sm")
        return None


# ═══════════════════════════════════════════════════════════════
# TEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_text_for_row(row: pd.Series) -> tuple[str, bool]:
    """
    Returns (text, used_real_text).

    Priority:
    1. title + summary  → real text → used_real_text = True
    2. neutral fallback → no artificial keywords added

    IMPORTANT: fallback text does NOT include words like
    'outbreak', 'alert', 'emergency' to avoid score inflation.
    """
    title   = str(row.get("title",   "") or "").strip()
    summary = str(row.get("summary", "") or "").strip()

    # Check if real text is available and meaningful
    has_real_title   = len(title)   > 5
    has_real_summary = len(summary) > 5

    if has_real_title or has_real_summary:
        parts = []
        if has_real_title:
            parts.append(title)
        if has_real_summary:
            parts.append(summary)
        return " ".join(parts)[:600], True

    # Neutral fallback — no artificial outbreak keywords
    disease  = str(row.get("disease",     "") or "")
    country  = str(row.get("country",     "") or "")
    severity = str(row.get("severity",    "") or "")
    deaths   = row.get("deaths",          0)
    cases    = row.get("cases_total",     0)

    parts = []
    if disease:  parts.append(f"{disease} in {country}.")
    if severity: parts.append(f"Severity level: {severity}.")
    if deaths:   parts.append(f"Deaths: {int(deaths)}.")
    if cases:    parts.append(f"Cases: {int(cases)}.")

    fallback = " ".join(parts) if parts else f"{disease} {country}"
    return fallback[:600], False


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_features(row: pd.Series, doc, used_real_text: bool) -> dict:
    """
    Extracts NLP features from spaCy doc + row data.
    is conservative with scores when fallback text is used.
    """
    text       = doc.text
    text_lower = text.lower()
    source     = str(row.get("source",  "") or "").lower()
    disease    = str(row.get("disease", "") or "").lower()
    country    = str(row.get("country", "") or "").lower()

    # ── Entity extraction ─────────────────────────────────────
    locations = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC")]
    orgs      = [e.text for e in doc.ents if e.label_ == "ORG"]
    persons   = [e.text for e in doc.ents if e.label_ == "PERSON"]
    dates     = [e.text for e in doc.ents if e.label_ == "DATE"]
    numbers   = [e.text for e in doc.ents
                 if e.label_ in ("CARDINAL", "QUANTITY", "PERCENT")]

    # ── WHO/CDC — only from REAL text ─────────────────────────
    # Do NOT auto-mark WHO just because source is who.int
    has_who_in_text = int(
        "who"              in text_lower or
        "world health"     in text_lower
    )
    has_cdc_in_text = int(
        "cdc"                         in text_lower or
        "centers for disease control" in text_lower
    )

    # Separate source flag — honest and transparent
    source_is_who = int(
        "who.int"        in source or
        "georgetown_don" in source or
        "who_don"        in source
    )

    # ── Scores — conservative for fallback text ───────────────
    if used_real_text:
        urgency_score     = keyword_score(text, URGENCY_WORDS)
        containment_score = keyword_score(text, CONTAINMENT_WORDS)
        risk_score        = keyword_score(text, RISK_KEYWORDS)
        response_score    = keyword_score(text, RESPONSE_KEYWORDS)
    else:
        # Fallback text has no outbreak keywords intentionally
        # Use structured data instead for scoring
        urgency_score     = 0
        containment_score = 0
        risk_score        = 0
        response_score    = 0

    # ── Text quality ──────────────────────────────────────────
    sentences          = [s.text.strip() for s in doc.sents if s.text.strip()]
    sentence_count     = max(len(sentences), 1)
    word_count         = len(text.split())
    avg_sent_len       = round(word_count / sentence_count, 1)

    # ── Disease/country flags ─────────────────────────────────
    has_disease_in_text = int(disease in text_lower) if disease else 0
    has_country_in_text = int(
        country in text_lower
    ) if country and country != "unknown" else 0

    # ── Escalation/containment flags ─────────────────────────
    # Only flag if using real text and score is meaningful
    if used_real_text:
        escalation_flag  = int(urgency_score >= 2 and urgency_score > containment_score)
        containment_flag = int(containment_score >= 2 and containment_score >= urgency_score)
    else:
        escalation_flag  = 0
        containment_flag = 0

    return {
        # Text source flags
        "used_real_text":            int(used_real_text),
        "used_fallback_text":        int(not used_real_text),
        "text_available":            int(used_real_text),
        "source_is_who":             source_is_who,
        # Entity features
        "entity_count":              len(doc.ents),
        "location_count":            len(locations),
        "org_count":                 len(orgs),
        "person_count":              len(persons),
        "date_entity_count":         len(dates),
        "numeric_entity_count":      len(numbers),
        # WHO/CDC from text only
        "has_who_mention":           has_who_in_text,
        "has_cdc_mention":           has_cdc_in_text,
        # Urgency scores
        "urgency_score":             urgency_score,
        "containment_score":         containment_score,
        "net_urgency":               urgency_score - containment_score,
        "risk_keyword_count":        risk_score,
        "response_keyword_count":    response_score,
        "escalation_language_flag":  escalation_flag,
        "containment_language_flag": containment_flag,
        # Text quality
        "word_count":                word_count,
        "sentence_count":            sentence_count,
        "avg_sentence_length":       avg_sent_len,
        "has_disease_in_text":       has_disease_in_text,
        "has_country_in_text":       has_country_in_text,
        # Locations
        "top_location_entities":     ", ".join(set(locations))[:200],
    }


# ═══════════════════════════════════════════════════════════════
# UNIFORMITY WARNING
# ═══════════════════════════════════════════════════════════════

def warn_if_uniform(df: pd.DataFrame, cols: list):
    """
    Warns if more than 90% of rows have the same value
    for an important NLP feature — signals inflated scores.
    """
    for col in cols:
        if col not in df.columns:
            continue
        top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
        if top_pct > 90:
            log.warning(
                f"  ⚠ '{col}' is {round(top_pct,1)}% uniform — "
                f"may not help ML"
            )


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_nlp_pipeline() -> pd.DataFrame:

    # Step 1 — Load
    log.info("─" * 55)
    log.info("STEP 1 — Loading clean features")
    log.info("─" * 55)

    in_path = config.PROCESSED_DIR / "features_clean.csv"
    if not in_path.exists():
        log.error(f"Not found: {in_path} — run cleaning.py first")
        return pd.DataFrame()

    df = pd.read_csv(in_path)
    log.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    has_title   = "title"   in df.columns
    has_summary = "summary" in df.columns
    log.info(f"  Has title column     : {has_title}")
    log.info(f"  Has summary column   : {has_summary}")

    # Step 2 — Load spaCy
    log.info("\n" + "─" * 55)
    log.info("STEP 2 — Loading spaCy model")
    log.info("─" * 55)

    nlp = load_spacy_model()
    if nlp is None:
        return pd.DataFrame()

    # Step 3 — Build texts
    log.info("\n" + "─" * 55)
    log.info("STEP 3 — Building texts and extracting features")
    log.info("─" * 55)

    texts          = []
    real_text_flags = []

    for _, row in df.iterrows():
        text, used_real = build_text_for_row(row)
        texts.append(text)
        real_text_flags.append(used_real)

    real_count     = sum(real_text_flags)
    fallback_count = len(real_text_flags) - real_count
    log.info(f"  Rows with real text  : {real_count}")
    log.info(f"  Rows using fallback  : {fallback_count}")
    log.info(f"  Processing with nlp.pipe()...")

    # Process with nlp.pipe() — batch processing, much faster
    docs         = list(nlp.pipe(texts, batch_size=50))
    nlp_features = []

    for i, (doc, (_, row), used_real) in enumerate(
        zip(docs, df.iterrows(), real_text_flags)
    ):
        features = extract_features(row, doc, used_real)
        nlp_features.append(features)
        if (i + 1) % 100 == 0:
            log.info(f"  Processed {i+1}/{len(df)} rows...")

    log.info(f"  Done — {len(nlp_features)} rows processed")

    # Step 4 — Merge
    nlp_df = pd.DataFrame(nlp_features)
    result = pd.concat([df.reset_index(drop=True), nlp_df], axis=1)

    # Step 5 — Validate
    log.info("\n" + "─" * 55)
    log.info("STEP 4 — Validation")
    log.info("─" * 55)

    assert len(result) == len(df), "Row count mismatch!"

    new_cols   = nlp_df.columns.tolist()
    null_count = result[new_cols].isnull().sum().sum()
    assert null_count == 0, f"Null values in NLP columns: {null_count}"

    log.info(f"  Row count            : {len(result)} ✓")
    log.info(f"  Null values          : {null_count} ✓")
    log.info(f"  Total columns        : {len(result.columns)}")
    log.info(f"  New NLP columns      : {len(new_cols)}")

    # Check for uniform columns
    check_cols = [
        "urgency_score", "containment_score",
        "has_who_mention", "escalation_language_flag",
        "containment_language_flag",
    ]
    warn_if_uniform(result, check_cols)

    # Step 6 — Save
    out_path = config.PROCESSED_DIR / "features_nlp.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    log.info(f"\n  Saved → data/processed/features_nlp.csv")

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 5: NLP Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    result = run_nlp_pipeline()

    if result.empty:
        print("Pipeline failed. Check errors above.")
        return

    print("\n" + "═" * 55)
    print("  Phase 5 Complete")
    print("═" * 55)
    print(f"  Rows processed         : {len(result)}")
    print(f"  Total features         : {len(result.columns)}")
    print(f"  Rows using real text   : {int(result['used_real_text'].sum())}")
    print(f"  Rows using fallback    : {int(result['used_fallback_text'].sum())}")
    print(f"  Rows with WHO mention  : {int(result['has_who_mention'].sum())}")
    print(f"  Source is WHO          : {int(result['source_is_who'].sum())}")
    print(f"  Avg urgency score      : {result['urgency_score'].mean():.2f}")
    print(f"  Max urgency score      : {result['urgency_score'].max()}")
    print(f"  Escalation flagged     : {int(result['escalation_language_flag'].sum())}")
    print(f"  Containment flagged    : {int(result['containment_language_flag'].sum())}")
    print(f"\n  Output:")
    print(f"    data/processed/features_nlp.csv")
    print(f"\n  Next step: Phase 6 — Feature Engineering")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()