# src/risk_scorer.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 8: Risk Scoring
#
# HOW TO RUN:
#   python src/risk_scorer.py
# ═══════════════════════════════════════════════════════════════

import re
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
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
log = logging.getLogger("biosignal.scorer")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

DISEASE_RISK_PROFILE = {
    "ebola":        0.9,  "marburg":      0.9,
    "plague":       0.8,  "lassa fever":  0.7,
    "rift valley":  0.7,  "yellow fever": 0.65,
    "cholera":      0.6,  "mpox":         0.55,
    "dengue":       0.5,  "meningitis":   0.5,
    "influenza":    0.5,  "covid":        0.85,
    "measles":      0.45, "polio":        0.4,
    "typhoid":      0.35, "hantavirus":   0.75,
    "nipah":        0.88, "monkeypox":    0.55,
}

DISEASE_FATALITY_WEIGHT = {
    "ebola":        0.9,  "marburg":      0.88,
    "plague":       0.75, "lassa fever":  0.65,
    "rift valley":  0.6,  "yellow fever": 0.55,
    "cholera":      0.45, "mpox":         0.3,
    "dengue":       0.35, "meningitis":   0.5,
    "influenza":    0.4,  "covid":        0.7,
    "measles":      0.3,  "polio":        0.25,
    "typhoid":      0.2,  "hantavirus":   0.7,
    "nipah":        0.85, "monkeypox":    0.3,
}

DISEASE_PEAK_MONTHS = {
    "cholera":      [5,6,7,8,9],
    "dengue":       [6,7,8,9,10],
    "yellow fever": [1,2,3],
    "meningitis":   [1,2,3,4],
    "influenza":    [11,12,1,2,3],
    "lassa fever":  [1,2,3,4],
    "ebola":        [1,2,3,4,5],
    "marburg":      [1,2,3],
    "plague":       [5,6,7,8],
    "mpox":         [1,2,3,4,5,6],
    "rift valley":  [10,11,12,1],
    "measles":      [1,2,3,4,5],
    "polio":        [6,7,8,9],
    "typhoid":      [5,6,7,8,9,10],
    "covid":        [10,11,12,1,2],
    "hantavirus":   [4,5,6,7,8],
    "nipah":        [1,2,3,4,5],
    "monkeypox":    [1,2,3,4,5,6],
}

DISEASE_ORDER = [
    "cholera", "covid", "dengue", "ebola", "influenza",
    "lassa fever", "marburg", "measles", "meningitis",
    "mpox", "plague", "polio", "rift valley", "typhoid",
    "yellow fever",
]

# Known location aliases — maps text mentions to clean country names
LOCATION_ALIASES = {
    "tenerife":           "Spain (Tenerife)",
    "canary islands":     "Spain (Canary Islands)",
    "drc":                "Democratic Republic of the Congo",
    "congo":              "Democratic Republic of the Congo",
    "democratic republic":"Democratic Republic of the Congo",
    "brasil":             "Brazil",
    "brasilia":           "Brazil",
    "usa":                "United States",
    "u.s.":               "United States",
}

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
    "stable", "improving", "eliminated", "resolved", "ended",
]

RISK_KEYWORDS = [
    "risk", "threat", "danger", "vulnerable", "exposure",
    "transmission", "infectious", "contagious", "mortality",
]

RESPONSE_KEYWORDS = [
    "response", "deployed", "intervention", "vaccination",
    "campaign", "investigation", "team", "support", "coordination",
]


# ═══════════════════════════════════════════════════════════════
# RISK LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════

def get_risk_level(score: float) -> str:
    if score >= 81:   return "Critical Risk"
    if score >= 61:   return "High Risk"
    if score >= 31:   return "Medium Risk"
    return "Low Risk"


def get_risk_emoji(level: str) -> str:
    return {
        "Critical Risk": "🚨 Critical Risk",
        "High Risk":     "🔴 High Risk",
        "Medium Risk":   "🟡 Medium Risk",
        "Low Risk":      "🟢 Low Risk",
    }.get(level, level)


def get_confidence_band(prob: float) -> str:
    if prob >= 0.75: return "High Confidence"
    if prob >= 0.55: return "Medium Confidence"
    return "Low Confidence"


def get_top_risk_reason(
    urgency_score: int,
    spike_ratio: float,
    is_peak: int,
    disease_risk: float,
    disease_count: int,
    net_urgency: int,
) -> str:
    if urgency_score >= 3 and disease_risk >= 0.7:
        return "High NLP urgency and disease risk"
    if spike_ratio >= config.SPIKE_RATIO_HIGH:
        return "Google Trends spike detected"
    if spike_ratio >= config.SPIKE_RATIO_MEDIUM:
        return "Elevated search volume trend"
    if is_peak and disease_risk >= 0.6:
        return "Seasonal risk and outbreak frequency"
    if disease_count >= 50 and disease_risk >= 0.6:
        return "Strong disease/location historical pattern"
    if urgency_score >= 2 or net_urgency > 0:
        return "Moderate early-warning signal"
    return "Low current early-warning signal"


def fix_country(country: str, title: str, summary: str) -> str:
    """
    Fixes unknown country using location aliases found in
    article title and summary text.
    """
    if country and country.lower() != "unknown":
        return country

    text = f"{title} {summary}".lower()
    for alias, clean_name in LOCATION_ALIASES.items():
        if alias in text:
            return clean_name

    return "Unknown"


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

def load_model():
    model_path = ROOT / "models" / "xgboost_early_warning_model.pkl"

    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        log.error("Run model.py first.")
        return None, None

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    mode = saved.get("mode", "unknown")
    if mode != "early_warning":
        log.error(f"Wrong model loaded! Mode={mode}. Need early_warning model.")
        return None, None

    model        = saved["model"]
    feature_cols = saved["feature_cols"]

    log.info(f"  Model loaded         : xgboost_early_warning_model.pkl")
    log.info(f"  Mode                 : {mode} ✓")
    log.info(f"  Trained at           : {saved.get('trained_at', 'unknown')}")
    log.info(f"  Feature count        : {len(feature_cols)}")

    return model, feature_cols


# ═══════════════════════════════════════════════════════════════
# KEYWORD SCORER
# ═══════════════════════════════════════════════════════════════

def keyword_score(text: str, keywords: list) -> int:
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
# FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_features_for_row(
    row: pd.Series,
    trends_df: pd.DataFrame,
    nlp,
    feature_cols: list,
    hist_df: pd.DataFrame,
) -> dict:
    """
    Builds exact same features as Phase 6 for one live row.
    All missing values filled with 0.
    """
    disease = str(row.get("disease", "") or "").lower().strip()
    title   = str(row.get("title",   "") or "")
    summary = str(row.get("summary", "") or "")

    # Fix unknown country using location aliases
    raw_country = str(row.get("country", "") or "").strip()
    country     = fix_country(raw_country, title, summary)

    try:
        date  = pd.to_datetime(row.get("date", datetime.today()))
        year  = date.year
        month = date.month
        week  = int(date.isocalendar()[1])
    except Exception:
        now   = datetime.today()
        year  = now.year
        month = now.month
        week  = int(now.isocalendar()[1])

    # Season encoding
    if month in [12,1,2]:   season = 0
    elif month in [3,4,5]:  season = 1
    elif month in [6,7,8]:  season = 2
    else:                   season = 3

    text = f"{title} {summary}".strip() or f"{disease} in {country}."

    # NLP
    doc       = nlp(text[:500])
    locations = [e.text for e in doc.ents if e.label_ in ("GPE","LOC")]
    orgs      = [e.text for e in doc.ents if e.label_ == "ORG"]
    persons   = [e.text for e in doc.ents if e.label_ == "PERSON"]
    dates_ent = [e.text for e in doc.ents if e.label_ == "DATE"]
    numbers   = [e.text for e in doc.ents
                 if e.label_ in ("CARDINAL","QUANTITY","PERCENT")]
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    urgency_score     = keyword_score(text, URGENCY_WORDS)
    containment_score = keyword_score(text, CONTAINMENT_WORDS)
    risk_kw           = keyword_score(text, RISK_KEYWORDS)
    response_kw       = keyword_score(text, RESPONSE_KEYWORDS)
    word_count        = len(text.split())
    sentence_count    = max(len(sentences), 1)
    avg_sent_len      = round(word_count / sentence_count, 1)
    net_urgency       = urgency_score - containment_score
    escalation_flag   = 1 if (urgency_score >= 2 and net_urgency > 0) else 0

    # Trends
    spike_ratio_max = 0.0
    spike_ratio_avg = 0.0
    search_vol_avg  = 0.0
    search_vol_max  = 0.0
    baseline_avg    = 0.0
    spike_level_enc = 0

    if not trends_df.empty and disease in trends_df["disease"].values:
        d_trends        = trends_df[trends_df["disease"] == disease]
        recent          = d_trends.sort_values("date").tail(4)
        search_vol_avg  = round(recent["search_volume"].mean(), 1)
        search_vol_max  = round(recent["search_volume"].max(),  1)
        baseline        = max(d_trends["search_volume"].mean(), 1)
        spike_ratio_max = round(search_vol_max / baseline, 2)
        spike_ratio_avg = round(search_vol_avg / baseline, 2)
        baseline_avg    = round(baseline, 1)

        if spike_ratio_max >= config.SPIKE_RATIO_HIGH:
            spike_level_enc = 3
        elif spike_ratio_max >= config.SPIKE_RATIO_MEDIUM:
            spike_level_enc = 2
        elif spike_ratio_max >= config.SPIKE_RATIO_LOW:
            spike_level_enc = 1

    # Disease profile
    disease_risk    = DISEASE_RISK_PROFILE.get(disease, 0.5)
    fatality_weight = DISEASE_FATALITY_WEIGHT.get(disease, 0.4)
    is_peak         = 1 if month in DISEASE_PEAK_MONTHS.get(disease, []) else 0
    season_risk     = is_peak * 0.3
    disease_encoded = DISEASE_ORDER.index(disease) \
                      if disease in DISEASE_ORDER else -1

    # Frequency from historical data
    disease_count = int((hist_df["disease"] == disease).sum()) \
                    if not hist_df.empty else 1
    country_count = int((hist_df["country"] == country).sum()) \
                    if not hist_df.empty else 1
    dc_count      = int(
        ((hist_df["disease"] == disease) &
         (hist_df["country"] == country)).sum()
    ) if not hist_df.empty else 0
    is_endemic    = 1 if dc_count >= 3 else 0

    has_trend_data      = 1 if search_vol_avg > 0 else 0
    spike_alert         = 1 if spike_ratio_max >= config.SPIKE_RATIO_HIGH   else 0
    spike_warning       = 1 if spike_ratio_max >= config.SPIKE_RATIO_MEDIUM else 0
    has_who_mention     = 1 if ("who" in text.lower() or
                                "world health" in text.lower()) else 0
    has_disease_in_text = 1 if disease in text.lower() else 0
    has_country_in_text = 1 if (country.lower() in text.lower()
                                and country.lower() != "unknown") else 0

    all_features = {
        "year":                      year,
        "month":                     month,
        "week":                      week,
        "search_volume_avg":         search_vol_avg,
        "search_volume_max":         search_vol_max,
        "spike_ratio_max":           spike_ratio_max,
        "spike_ratio_avg":           spike_ratio_avg,
        "baseline_avg":              baseline_avg,
        "entity_count":              len(doc.ents),
        "location_count":            len(locations),
        "org_count":                 len(orgs),
        "person_count":              len(persons),
        "date_entity_count":         len(dates_ent),
        "numeric_entity_count":      len(numbers),
        "has_who_mention":           has_who_mention,
        "urgency_score":             urgency_score,
        "containment_score":         containment_score,
        "net_urgency":               net_urgency,
        "risk_keyword_count":        risk_kw,
        "response_keyword_count":    response_kw,
        "escalation_language_flag":  escalation_flag,
        "word_count":                word_count,
        "sentence_count":            sentence_count,
        "avg_sentence_length":       avg_sent_len,
        "title_word_count":          len(title.split()),
        "summary_word_count":        len(summary.split()),
        "has_disease_in_text":       has_disease_in_text,
        "has_country_in_text":       has_country_in_text,
        "disease_risk_score":        disease_risk,
        "disease_fatality_weight":   fatality_weight,
        "disease_outbreak_count":    disease_count,
        "country_outbreak_count":    country_count,
        "disease_country_count":     dc_count,
        "is_endemic_country":        is_endemic,
        "is_peak_season":            is_peak,
        "season_risk":               season_risk,
        "has_trend_data":            has_trend_data,
        "spike_alert":               spike_alert,
        "spike_warning":             spike_warning,
        "disease_encoded":           disease_encoded,
        "season_encoded":            season,
        "spike_level_encoded":       spike_level_enc,
    }

    return {k: float(all_features.get(k, 0)) for k in feature_cols}


# ═══════════════════════════════════════════════════════════════
# EARLY WARNING DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_early_warnings(
    trends_df: pd.DataFrame,
    who_df: pd.DataFrame,
) -> pd.DataFrame:
    if trends_df.empty:
        return pd.DataFrame()

    latest   = trends_df.sort_values("date").groupby("disease").last().reset_index()
    baseline = trends_df.groupby("disease")["search_volume"].mean()
    latest["baseline"]    = latest["disease"].map(baseline).replace(0, 1)
    latest["spike_ratio"] = (latest["search_volume"] / latest["baseline"]).round(2)

    who_diseases = set()
    if not who_df.empty and "disease" in who_df.columns:
        try:
            recent_who = who_df[
                pd.to_datetime(who_df["date"], errors="coerce") >=
                pd.Timestamp.now() - pd.Timedelta(days=30)
            ]
            who_diseases = set(
                recent_who["disease"].dropna().str.lower().unique()
            )
        except Exception:
            who_diseases = set(who_df["disease"].dropna().str.lower().unique())

    rows = []
    for _, row in latest.iterrows():
        disease     = row["disease"]
        spike_ratio = row["spike_ratio"]
        search_vol  = row["search_volume"]
        who_rep     = disease in who_diseases

        if spike_ratio >= config.SPIKE_RATIO_LOW:
            if spike_ratio >= config.SPIKE_RATIO_HIGH:
                signal_level = "Alert"
            elif spike_ratio >= config.SPIKE_RATIO_MEDIUM:
                signal_level = "Warning"
            else:
                signal_level = "Watch"

            rows.append({
                "disease":       disease,
                "search_volume": search_vol,
                "spike_ratio":   spike_ratio,
                "signal_level":  signal_level,
                "who_reported":  who_rep,
                "early_warning": not who_rep,
                "date_checked":  datetime.now().strftime("%Y-%m-%d"),
            })

    return pd.DataFrame(rows).sort_values(
        "spike_ratio", ascending=False
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# MAIN SCORER
# ═══════════════════════════════════════════════════════════════

def run_risk_scorer() -> pd.DataFrame:

    log.info("─" * 55)
    log.info("STEP 1 — Loading Model")
    log.info("─" * 55)

    model, feature_cols = load_model()
    if model is None:
        return pd.DataFrame()

    if not feature_cols:
        log.error("Feature columns are empty. Re-run model.py.")
        return pd.DataFrame()

    log.info("\n" + "─" * 55)
    log.info("STEP 2 — Loading Live Data")
    log.info("─" * 55)

    who_df = pd.DataFrame()
    if config.WHO_RAW_FILE.exists():
        who_df = pd.read_csv(config.WHO_RAW_FILE)
        who_df["disease"] = who_df["disease"].str.lower().str.strip()
        log.info(f"  WHO reports          : {len(who_df)} articles")
    else:
        log.warning("  WHO reports not found — run data_collection.py")

    trends_df = pd.DataFrame()
    if config.TRENDS_RAW_FILE.exists():
        trends_df = pd.read_csv(config.TRENDS_RAW_FILE)
        trends_df["date"]    = pd.to_datetime(trends_df["date"])
        trends_df["disease"] = trends_df["disease"].str.lower().str.strip()
        log.info(f"  Trends records       : {len(trends_df)} rows")
        log.info(f"  Diseases             : {sorted(trends_df['disease'].unique().tolist())}")
    else:
        log.warning("  Trends not found — run data_collection.py")

    hist_df = pd.DataFrame()
    if config.COMBINED_FILE.exists():
        hist_df = pd.read_csv(config.COMBINED_FILE)
        hist_df["disease"] = hist_df["disease"].str.lower().str.strip()

    try:
        nlp = spacy.load("en_core_web_sm")
        log.info(f"  spaCy loaded         : en_core_web_sm")
    except OSError:
        log.error("spaCy not found. Run: python -m spacy download en_core_web_sm")
        return pd.DataFrame()

    log.info("\n" + "─" * 55)
    log.info("STEP 3 — Scoring WHO Articles")
    log.info("─" * 55)

    scored_rows = []
    skipped     = 0

    if not who_df.empty:
        for _, row in who_df.iterrows():
            disease = str(row.get("disease", "") or "").lower().strip()

            if disease == "unknown" or not disease:
                skipped += 1
                continue

            try:
                features   = build_features_for_row(
                    row, trends_df, nlp, feature_cols, hist_df
                )
                feat_array = np.array([[features[k] for k in feature_cols]])
                risk_prob  = float(model.predict_proba(feat_array)[0][1])
                risk_score = round(risk_prob * 100, 1)
                predicted  = int(model.predict(feat_array)[0])
                risk_level = get_risk_level(risk_score)
                confidence = get_confidence_band(risk_prob)

                # Spike ratio
                spike_ratio = 0.0
                has_spike   = False
                if not trends_df.empty and disease in trends_df["disease"].values:
                    d_t         = trends_df[trends_df["disease"] == disease]
                    recent_vol  = d_t.sort_values("date").tail(4)["search_volume"].mean()
                    base        = max(d_t["search_volume"].mean(), 1)
                    spike_ratio = round(recent_vol / base, 2)
                    has_spike   = spike_ratio >= config.SPIKE_RATIO_LOW

                month_now    = pd.to_datetime(
                    row.get("date", datetime.today()), errors="coerce"
                ).month
                is_peak_val  = 1 if month_now in DISEASE_PEAK_MONTHS.get(disease, []) else 0
                disease_risk = DISEASE_RISK_PROFILE.get(disease, 0.5)
                disease_count= int((hist_df["disease"] == disease).sum()) \
                               if not hist_df.empty else 1
                urgency_text = str(row.get("title","")) + " " + str(row.get("summary",""))
                urgency_val  = keyword_score(urgency_text, URGENCY_WORDS)
                net_urg      = urgency_val - keyword_score(urgency_text, CONTAINMENT_WORDS)

                top_reason = get_top_risk_reason(
                    urgency_val, spike_ratio, is_peak_val,
                    disease_risk, disease_count, net_urg
                )

                # Fix country
                raw_country = str(row.get("country", "unknown"))
                country     = fix_country(
                    raw_country,
                    str(row.get("title",   "")),
                    str(row.get("summary", "")),
                )

                scored_rows.append({
                    "date":                   str(row.get("date",    "")),
                    "disease":                disease,
                    "country":                country,
                    "severity":               str(row.get("severity","low")),
                    "source":                 str(row.get("source",  "")),
                    "title":                  str(row.get("title",   ""))[:120],
                    "summary":                str(row.get("summary", ""))[:200],
                    "escalation_probability": round(risk_prob, 4),
                    "risk_score":             risk_score,
                    "risk_level":             risk_level,
                    "predicted_escalated":    predicted,
                    "confidence_band":        confidence,
                    "top_risk_reason":        top_reason,
                    "spike_ratio":            spike_ratio,
                    "has_spike":              has_spike,
                    "scored_at":              datetime.now().strftime("%Y-%m-%d %H:%M"),
                })

            except Exception as e:
                log.warning(f"  Could not score '{disease}': {e}")
                skipped += 1

        if skipped > 0:
            log.warning(f"  Skipped {skipped} articles (unknown disease or scoring error)")
            if skipped > len(who_df) * 0.5:
                log.warning("  ⚠ More than 50% skipped — check disease detection")

    scored_df = pd.DataFrame(scored_rows).sort_values(
        ["risk_score", "escalation_probability"], ascending=False
    ).reset_index(drop=True)

    log.info(f"  Scored {len(scored_df)} WHO articles")

    log.info("\n" + "─" * 55)
    log.info("STEP 4 — Early Warning Detection")
    log.info("─" * 55)

    ew_df = detect_early_warnings(trends_df, who_df)

    if not ew_df.empty:
        true_ew = ew_df[ew_df["early_warning"]]
        log.info(f"  Signals detected     : {len(ew_df)}")
        log.info(f"  True early warnings  : {len(true_ew)}")
        log.info(f"  WHO confirmed        : {len(ew_df) - len(true_ew)}")
        log.info(f"\n  ── Signals ────────────────────────────────")
        for _, r in ew_df.iterrows():
            status = "⚡ NO WHO REPORT YET" if r["early_warning"] else "✓ WHO confirmed"
            log.info(
                f"    [{r['signal_level']}]  {r['disease']:<15} "
                f"spike={r['spike_ratio']}x  {status}"
            )
    else:
        log.info("  No spike signals above threshold")

    log.info("\n" + "─" * 55)
    log.info("STEP 5 — Saving Outputs")
    log.info("─" * 55)

    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scored_df.empty:
        scored_df.to_csv(output_dir / "risk_scores.csv", index=False)
        log.info(f"  Risk scores          → data/outputs/risk_scores.csv")

    if not scored_df.empty:
        alerts_df = scored_df[
            scored_df["risk_level"].isin(["High Risk", "Critical Risk"])
        ].sort_values(
            ["risk_score", "escalation_probability"], ascending=False
        ).reset_index(drop=True)
        alerts_df.to_csv(output_dir / "high_risk_alerts.csv", index=False)
        log.info(f"  High risk alerts     → data/outputs/high_risk_alerts.csv "
                 f"({len(alerts_df)} alerts)")

    if not ew_df.empty:
        ew_df.to_csv(output_dir / "early_warnings.csv", index=False)
        log.info(f"  Early warnings       → data/outputs/early_warnings.csv")

    true_ew_count = int(ew_df["early_warning"].sum()) if not ew_df.empty else 0
    summary = {
        "total_scored":          len(scored_df),
        "low_risk":              int((scored_df["risk_level"] == "Low Risk").sum())      if not scored_df.empty else 0,
        "medium_risk":           int((scored_df["risk_level"] == "Medium Risk").sum())   if not scored_df.empty else 0,
        "high_risk":             int((scored_df["risk_level"] == "High Risk").sum())     if not scored_df.empty else 0,
        "critical_risk":         int((scored_df["risk_level"] == "Critical Risk").sum()) if not scored_df.empty else 0,
        "early_warning_signals": len(ew_df),
        "true_early_warnings":   true_ew_count,
        "model_used":            "xgboost_early_warning_model.pkl",
        "scored_at":             datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(output_dir / "risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Risk summary         → data/outputs/risk_summary.json")

    return scored_df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 8: Risk Scoring")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    scored_df = run_risk_scorer()

    print("\n" + "═" * 55)
    print("  Phase 8 Complete — Risk Scoring")
    print("═" * 55)

    if not scored_df.empty:
        print(f"  Articles scored       : {len(scored_df)}")
        print(f"  Critical Risk         : {(scored_df['risk_level'] == 'Critical Risk').sum()}")
        print(f"  High Risk             : {(scored_df['risk_level'] == 'High Risk').sum()}")
        print(f"  Medium Risk           : {(scored_df['risk_level'] == 'Medium Risk').sum()}")
        print(f"  Low Risk              : {(scored_df['risk_level'] == 'Low Risk').sum()}")

        high_crit = scored_df[
            scored_df["risk_level"].isin(["High Risk", "Critical Risk"])
        ]
        if not high_crit.empty:
            print(f"\n  Top High-Risk Alerts:")
            print(f"  {'Disease':<15} {'Country':<30} {'Score':<8} {'Level':<15} {'Reason'}")
            print(f"  {'-'*95}")
            for _, r in high_crit.head(5).iterrows():
                print(
                    f"  {r['disease']:<15} {r['country']:<30} "
                    f"{r['risk_score']:<8} {get_risk_emoji(r['risk_level']):<15} "
                    f"{r['top_risk_reason']}"
                )

    ew_path = ROOT / "data" / "outputs" / "early_warnings.csv"
    if ew_path.exists():
        ew_df   = pd.read_csv(ew_path)
        true_ew = ew_df[ew_df["early_warning"]]
        print(f"\n  Early Warning Signals : {len(ew_df)}")
        print(f"  True early warnings   : {len(true_ew)} (spike, no WHO report)")
        if not true_ew.empty:
            print(f"\n  ⚡ Unconfirmed spikes:")
            for _, r in true_ew.iterrows():
                print(
                    f"    [{r['signal_level']}]  {r['disease']:<15} "
                    f"spike={r['spike_ratio']}x above baseline"
                )

    print(f"\n  Output:")
    print(f"    data/outputs/risk_scores.csv")
    print(f"    data/outputs/high_risk_alerts.csv")
    print(f"    data/outputs/early_warnings.csv")
    print(f"    data/outputs/risk_summary.json")
    print(f"\n  Next step: Phase 9 — SHAP Explainability")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()