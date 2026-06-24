# src/process_don.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Process Georgetown WHO DON Database
# HOW TO RUN: python src/process_don.py
# ═══════════════════════════════════════════════════════════════

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

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
log = logging.getLogger("biosignal.don")

DISEASE_MAP = {
    "cholera":               "cholera",
    "ebola":                 "ebola",
    "ebola virus":           "ebola",
    "marburg":               "marburg",
    "marburg fever":         "marburg",
    "yellow fever":          "yellow fever",
    "dengue fever":          "dengue",
    "dengue":                "dengue",
    "lassa fever":           "lassa fever",
    "plague":                "plague",
    "meningococcal disease": "meningitis",
    "meningitis":            "meningitis",
    "measles":               "measles",
    "polio":                 "polio",
    "rift valley fever":     "rift valley",
    "influenza":             "influenza",
    "avian influenza":       "influenza",
    "mpox":                  "mpox",
    "monkeypox":             "mpox",
    "covid-19":              "covid",
    "sars-cov":              "covid",
    "mers-cov":              "influenza",
    "typhoid":               "typhoid",
    "typhoid fever":         "typhoid",
    "zika virus disease":    "unknown",
    "west nile virus":       "unknown",
    "chikungunya":           "unknown",
    "anthrax":               "unknown",
}


def safe_float(val) -> float:
    """
    Safely converts any value to float.
    Handles messy strings like '>200000', '~500', 'unknown'.
    Returns 0.0 if conversion fails.
    """
    if pd.isna(val):
        return 0.0
    try:
        cleaned = (
            str(val)
            .replace(">", "")
            .replace("<", "")
            .replace("~", "")
            .replace(",", "")
            .strip()
        )
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def map_disease(raw: str) -> str:
    if pd.isna(raw) or not raw:
        return "unknown"
    return DISEASE_MAP.get(raw.strip().lower(), "unknown")


def create_severity(deaths, cases_total) -> str:
    d = safe_float(deaths)
    c = safe_float(cases_total)
    if d >= 10:
        return "high"
    if d >= 1 or c >= 50:
        return "medium"
    return "low"


def create_label(deaths, cases_total) -> int:
    d = safe_float(deaths)
    c = safe_float(cases_total)
    return 1 if (d >= 10 or c >= 100) else 0


def create_outbreak_score(deaths, cases_total, has_lab, has_ph) -> int:
    score = 0
    if safe_float(deaths) > 0:        score += 2
    if safe_float(cases_total) > 50:  score += 1
    if str(has_lab).lower() == "yes": score += 1
    if str(has_ph).lower() == "yes":  score += 1
    return score


def parse_report_date(date_str: str) -> str:
    if pd.isna(date_str) or not date_str:
        return ""
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(
                str(date_str).strip(), fmt
            ).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
    return str(date_str)


def process_don() -> pd.DataFrame:
    log.info("═" * 55)
    log.info("Processing Georgetown WHO DON Database")
    log.info("═" * 55)

    raw_path = config.RAW_DIR / "DONdatabase.csv"
    if not raw_path.exists():
        log.error(f"File not found: {raw_path}")
        return pd.DataFrame()

    df = pd.read_csv(raw_path)
    log.info(f"  Loaded {len(df)} raw rows")
    log.info(f"  Date range: {df['ReportDate'].min()} → {df['ReportDate'].max()}")

    records = []
    for _, row in df.iterrows():
        disease     = map_disease(row.get("DiseaseLevel1", ""))
        title       = str(row.get("Headline",   "")).strip()
        country     = str(row.get("Country",    "")).strip()
        deaths      = row.get("Deaths",         np.nan)
        cases_total = row.get("CasesTotal",     np.nan)

        severity  = create_severity(deaths, cases_total)
        escalated = create_label(deaths, cases_total)
        score     = create_outbreak_score(
            deaths, cases_total,
            row.get("LabConfirmation", ""),
            row.get("PHIntervention",  ""),
        )

        # Build summary from real fields
        parts = []
        c = safe_float(cases_total)
        d = safe_float(deaths)
        if c > 0: parts.append(f"Total cases: {int(c)}")
        if d > 0: parts.append(f"Deaths: {int(d)}")
        if not pd.isna(row.get("OutbreakStartYear")):
            parts.append(f"Outbreak start: {int(row['OutbreakStartYear'])}")
        summary = ". ".join(parts)

        records.append({
            "date":                     parse_report_date(row.get("ReportDate", "")),
            "title":                    title,
            "disease":                  disease,
            "country":                  country,
            "severity":                 severity,
            "outbreak_relevance_score": score,
            "escalated":                escalated,
            "cases_total":              int(c),
            "deaths":                   int(d),
            "summary":                  summary,
            "link":                     str(row.get("Link", "")),
            "source":                   "georgetown_don",
        })

    result   = pd.DataFrame(records)
    known_df = result[result["disease"] != "unknown"].copy()
    unknown  = result[result["disease"] == "unknown"].copy()

    log.info(f"\n  Known disease rows   : {len(known_df)}")
    log.info(f"  Unknown disease rows : {len(unknown)} (skipped)")

    before   = len(known_df)
    known_df = known_df.drop_duplicates(
        subset=["title", "country"]
    ).reset_index(drop=True)
    log.info(f"  Duplicates removed   : {before - len(known_df)}")

    known_df = known_df.sort_values(
        "date", ascending=False
    ).reset_index(drop=True)

    out_path = config.RAW_DIR / "don_processed.csv"
    known_df.to_csv(out_path, index=False)

    esc = known_df["escalated"].sum()
    log.info(f"\n  Total rows           : {len(known_df)}")
    log.info(f"  Escalated (label=1)  : {esc} ({round(esc/len(known_df)*100,1)}%)")
    log.info(f"  Contained (label=0)  : {len(known_df) - esc}")
    log.info(f"\n  Disease breakdown:")
    for d, c in known_df["disease"].value_counts().items():
        log.info(f"    {d:<20} {c}")
    log.info(f"\n  Saved → don_processed.csv")

    return known_df


def merge_with_existing() -> pd.DataFrame:
    log.info("\n" + "═" * 55)
    log.info("Merging with existing WHO RSS data")
    log.info("═" * 55)

    all_dfs = []

    don_path = config.RAW_DIR / "don_processed.csv"
    if don_path.exists():
        df = pd.read_csv(don_path)
        log.info(f"  DON processed  : {len(df)} rows")
        all_dfs.append(df)

    if config.WHO_RAW_FILE.exists():
        df = pd.read_csv(config.WHO_RAW_FILE)
        if "escalated"   not in df.columns: df["escalated"]   = df["severity"].apply(lambda s: 1 if s == "high" else 0)
        if "cases_total" not in df.columns: df["cases_total"] = 0
        if "deaths"      not in df.columns: df["deaths"]      = 0
        log.info(f"  WHO RSS        : {len(df)} rows")
        all_dfs.append(df)

    if not all_dfs:
        log.error("No data found.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=["title"]).reset_index(drop=True)
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

    out_path = config.RAW_DIR / "combined_historical.csv"
    combined.to_csv(out_path, index=False)

    esc = combined["escalated"].sum()
    log.info(f"  Duplicates removed : {before - len(combined)}")
    log.info(f"  Combined total     : {len(combined)} rows")
    log.info(f"  Escalated (1)      : {esc}")
    log.info(f"  Contained (0)      : {len(combined) - esc}")
    log.info(f"  Saved → combined_historical.csv")

    return combined


def main():
    print("\n" + "═" * 55)
    print("  BioSignal — DON Database Processing")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    don_df      = process_don()
    combined_df = merge_with_existing()

    print("\n" + "═" * 55)
    print("  Processing Complete")
    print("═" * 55)
    print(f"  DON records processed  : {len(don_df)}")
    print(f"  Combined total         : {len(combined_df)}")
    print(f"  Escalated (label=1)    : {int(combined_df['escalated'].sum())}")
    print(f"  Contained (label=0)    : {len(combined_df) - int(combined_df['escalated'].sum())}")
    print(f"\n  Files:")
    print(f"    data/raw/don_processed.csv")
    print(f"    data/raw/combined_historical.csv")
    print(f"\n  Next step: Phase 4 — Data Cleaning")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()