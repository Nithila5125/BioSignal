# src/cleaning.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 4: Data Cleaning
#
# What this does:
#   1. Loads combined_historical.csv (WHO outbreak records)
#   2. Loads trends_raw.csv (Google Trends search volume)
#   3. Cleans both datasets (nulls, types, formats)
#   4. Merges them on disease + date
#   5. Saves clean merged dataset to data/processed/features_clean.csv
#
# HOW TO RUN:
#   python src/cleaning.py
# ═══════════════════════════════════════════════════════════════

import sys
import logging
from pathlib import Path

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
log = logging.getLogger("biosignal.cleaning")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CLEAN WHO HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════

def clean_who_data() -> pd.DataFrame:
    """
    Loads and cleans combined_historical.csv.

    Cleaning steps:
    - Parse dates properly
    - Fill missing values
    - Standardize disease and country names
    - Remove rows with no disease or date
    - Add month and year columns for merging with Trends
    - Ensure escalated label is 0 or 1
    """
    log.info("─" * 55)
    log.info("STEP 1 — Cleaning WHO Historical Data")
    log.info("─" * 55)

    path = config.COMBINED_FILE
    if not path.exists():
        log.error(f"File not found: {path}")
        log.error("Run process_don.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"  Loaded {len(df)} rows")

    # ── Fix dates ─────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date couldn't be parsed
    before = len(df)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    log.info(f"  Dropped {before - len(df)} rows with unparseable dates")

    # Add year, month, week columns — needed for merging with Trends
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)

    # ── Fix disease column ────────────────────────────────────
    df["disease"] = df["disease"].str.lower().str.strip()

    # Drop rows with no disease
    before = len(df)
    df = df[df["disease"] != "unknown"].reset_index(drop=True)
    df = df[df["disease"].notna()].reset_index(drop=True)
    log.info(f"  Dropped {before - len(df)} rows with unknown disease")

    # ── Fix country column ────────────────────────────────────
    df["country"] = df["country"].fillna("unknown").str.strip()

    # ── Fix severity column ───────────────────────────────────
    valid_severities = ["low", "medium", "high"]
    df["severity"] = df["severity"].str.lower().str.strip()
    df["severity"] = df["severity"].where(
        df["severity"].isin(valid_severities), "low"
    )

    # ── Fix numeric columns ───────────────────────────────────
    for col in ["cases_total", "deaths", "outbreak_relevance_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    # ── Fix escalated label ───────────────────────────────────
    df["escalated"] = pd.to_numeric(
        df["escalated"], errors="coerce"
    ).fillna(0).astype(int)
    df["escalated"] = df["escalated"].clip(0, 1)

    # ── Add severity as numeric ───────────────────────────────
    severity_map = {"low": 0, "medium": 1, "high": 2}
    df["severity_score"] = df["severity"].map(severity_map).fillna(0).astype(int)

    # ── Add season column ─────────────────────────────────────
    # Useful feature — many diseases are seasonal
    def get_season(month):
        if month in [12, 1, 2]:  return "winter"
        if month in [3, 4, 5]:   return "spring"
        if month in [6, 7, 8]:   return "summer"
        return "autumn"

    df["season"] = df["month"].apply(get_season)

    # ── Final column selection ────────────────────────────────
    keep_cols = [
        "date", "year", "month", "week", "season",
        "disease", "country", "severity", "severity_score",
        "outbreak_relevance_score", "cases_total", "deaths",
        "escalated", "source",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    log.info(f"  Clean WHO rows       : {len(df)}")
    log.info(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  Diseases             : {sorted(df['disease'].unique().tolist())}")
    log.info(f"  Escalated (1)        : {df['escalated'].sum()}")
    log.info(f"  Contained (0)        : {(df['escalated'] == 0).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2 — CLEAN GOOGLE TRENDS DATA
# ═══════════════════════════════════════════════════════════════

def clean_trends_data() -> pd.DataFrame:
    """
    Loads and cleans trends_raw.csv.

    Cleaning steps:
    - Parse dates
    - Fill missing search volumes with 0
    - Add year, month, week columns for merging
    - Calculate rolling 12-week baseline per disease
    - Calculate spike ratio = current / baseline
    - Add spike level label
    """
    log.info("\n" + "─" * 55)
    log.info("STEP 2 — Cleaning Google Trends Data")
    log.info("─" * 55)

    path = config.TRENDS_RAW_FILE
    if not path.exists():
        log.error(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"  Loaded {len(df)} rows")

    # ── Parse dates ───────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # ── Add time columns ──────────────────────────────────────
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)

    # ── Fix disease names ─────────────────────────────────────
    df["disease"] = df["disease"].str.lower().str.strip()

    # ── Fix search volume ─────────────────────────────────────
    df["search_volume"] = pd.to_numeric(
        df["search_volume"], errors="coerce"
    ).fillna(0).astype(int)

    # ── Sort by disease and date ──────────────────────────────
    df = df.sort_values(["disease", "date"]).reset_index(drop=True)

    # ── Calculate spike ratio per disease ─────────────────────
    # This is the KEY novel feature:
    # spike_ratio = this week / 12-week rolling average
    # spike_ratio > 2.0 means search volume doubled = early warning

    spike_ratios  = []
    baselines     = []

    for disease in df["disease"].unique():
        mask   = df["disease"] == disease
        subset = df[mask].copy()

        # 12-week rolling average as baseline
        baseline = subset["search_volume"].rolling(
            window=12, min_periods=1
        ).mean()

        # Avoid division by zero
        baseline = baseline.replace(0, 1)

        # Spike ratio
        ratio = subset["search_volume"] / baseline

        spike_ratios.extend(ratio.tolist())
        baselines.extend(baseline.tolist())

    df["baseline_avg"]  = baselines
    df["spike_ratio"]   = spike_ratios
    df["spike_ratio"]   = df["spike_ratio"].round(2)
    df["baseline_avg"]  = df["baseline_avg"].round(1)

    # ── Add spike level label ─────────────────────────────────
    def get_spike_level(ratio):
        if ratio >= config.SPIKE_RATIO_HIGH:   return "alert"
        if ratio >= config.SPIKE_RATIO_MEDIUM: return "warning"
        if ratio >= config.SPIKE_RATIO_LOW:    return "watch"
        return "normal"

    df["spike_level"] = df["spike_ratio"].apply(get_spike_level)

    log.info(f"  Clean Trends rows    : {len(df)}")
    log.info(f"  Diseases             : {sorted(df['disease'].unique().tolist())}")
    log.info(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  Spike level counts   : {df['spike_level'].value_counts().to_dict()}")
    log.info(f"  Max spike ratio      : {df['spike_ratio'].max()}")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3 — MERGE WHO + TRENDS
# ═══════════════════════════════════════════════════════════════

def merge_who_and_trends(
    who_df: pd.DataFrame,
    trends_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges WHO outbreak records with Google Trends data.

    Merge strategy:
    - WHO data is per outbreak event (one row per event)
    - Trends data is weekly (one row per disease per week)
    - We merge on disease + year + month
    - We take the MAX spike ratio for that month
      (captures the peak search interest around the outbreak)

    Why month-level merge:
    - WHO reports don't always have exact outbreak start dates
    - Month-level is more robust than week-level
    """
    log.info("\n" + "─" * 55)
    log.info("STEP 3 — Merging WHO + Trends Data")
    log.info("─" * 55)

    if who_df.empty or trends_df.empty:
        log.warning("  One or both dataframes empty — skipping merge")
        return who_df

    # ── Aggregate Trends to monthly level ─────────────────────
    trends_monthly = trends_df.groupby(
        ["disease", "year", "month"]
    ).agg(
        search_volume_avg  = ("search_volume", "mean"),
        search_volume_max  = ("search_volume", "max"),
        spike_ratio_max    = ("spike_ratio",   "max"),
        spike_ratio_avg    = ("spike_ratio",   "mean"),
        spike_level_top    = ("spike_level",   lambda x: x.value_counts().index[0]),
        baseline_avg       = ("baseline_avg",  "mean"),
    ).reset_index()

    trends_monthly["search_volume_avg"] = trends_monthly["search_volume_avg"].round(1)
    trends_monthly["spike_ratio_max"]   = trends_monthly["spike_ratio_max"].round(2)
    trends_monthly["spike_ratio_avg"]   = trends_monthly["spike_ratio_avg"].round(2)

    log.info(f"  WHO rows             : {len(who_df)}")
    log.info(f"  Trends monthly rows  : {len(trends_monthly)}")

    # ── Merge ─────────────────────────────────────────────────
    merged = pd.merge(
        who_df,
        trends_monthly,
        on=["disease", "year", "month"],
        how="left",   # keep all WHO rows even if no Trends match
    )

    # ── Fill missing Trends values ────────────────────────────
    # Not all diseases/dates have Trends data
    # Fill with 0 for volumes, 1.0 for ratios (neutral)
    trends_cols = [
        "search_volume_avg", "search_volume_max",
        "spike_ratio_max", "spike_ratio_avg", "baseline_avg",
    ]
    for col in trends_cols:
        merged[col] = merged[col].fillna(0)

    merged["spike_level_top"] = merged["spike_level_top"].fillna("normal")

    matched = merged["search_volume_avg"].gt(0).sum()
    log.info(f"  Merged rows          : {len(merged)}")
    log.info(f"  Rows with Trends     : {matched} ({round(matched/len(merged)*100,1)}%)")
    log.info(f"  Rows without Trends  : {len(merged) - matched}")

    return merged


# ═══════════════════════════════════════════════════════════════
# STEP 4 — FINAL VALIDATION + SAVE
# ═══════════════════════════════════════════════════════════════

def validate_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final validation before saving.
    Checks column types, removes any remaining nulls in key columns,
    prints a data quality report, saves to features_clean.csv
    """
    log.info("\n" + "─" * 55)
    log.info("STEP 4 — Validation + Save")
    log.info("─" * 55)

    if df.empty:
        log.error("  Empty dataframe — nothing to save")
        return df

    # ── Ensure key columns exist ──────────────────────────────
    required = [
        "date", "disease", "country", "severity_score",
        "cases_total", "deaths", "outbreak_relevance_score",
        "escalated",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        log.warning(f"  Missing columns: {missing_cols} — filling with 0")
        for col in missing_cols:
            df[col] = 0

    # ── Final null check on key columns ──────────────────────
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            log.warning(f"  {col} has {null_count} nulls — filling")
            df[col] = df[col].fillna(0)

    # ── Ensure correct types ──────────────────────────────────
    int_cols = [
        "cases_total", "deaths", "outbreak_relevance_score",
        "escalated", "severity_score", "year", "month",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0).astype(int)

    float_cols = [
        "search_volume_avg", "search_volume_max",
        "spike_ratio_max", "spike_ratio_avg", "baseline_avg",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0.0).astype(float)

    # ── Sort by date descending ───────────────────────────────
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "features_clean.csv"
    df.to_csv(out_path, index=False)

    # ── Data quality report ───────────────────────────────────
    log.info(f"\n  ── Data Quality Report ────────────────────")
    log.info(f"  Total rows           : {len(df)}")
    log.info(f"  Total columns        : {len(df.columns)}")
    log.info(f"  Columns              : {df.columns.tolist()}")
    log.info(f"  Date range           : {df['date'].min()} → {df['date'].max()}")
    log.info(f"  Null values          : {df.isnull().sum().sum()}")
    log.info(f"  Escalated (1)        : {df['escalated'].sum()}")
    log.info(f"  Contained (0)        : {(df['escalated']==0).sum()}")
    log.info(f"  Label balance        : {round(df['escalated'].mean()*100,1)}% escalated")
    log.info(f"\n  Disease counts:")
    for d, c in df["disease"].value_counts().items():
        log.info(f"    {d:<20} {c}")
    log.info(f"\n  Saved → data/processed/features_clean.csv")

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    from datetime import datetime
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 4: Data Cleaning")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    # Step 1 — Clean WHO data
    who_df    = clean_who_data()

    # Step 2 — Clean Trends data
    trends_df = clean_trends_data()

    # Step 3 — Merge
    merged_df = merge_who_and_trends(who_df, trends_df)

    # Step 4 — Validate and save
    final_df  = validate_and_save(merged_df)

    print("\n" + "═" * 55)
    print("  Phase 4 Complete")
    print("═" * 55)
    print(f"  Clean rows             : {len(final_df)}")
    print(f"  Columns                : {len(final_df.columns)}")
    print(f"  Escalated (label=1)    : {int(final_df['escalated'].sum())}")
    print(f"  Contained (label=0)    : {int((final_df['escalated']==0).sum())}")
    print(f"  Has Trends data        : {int(final_df['search_volume_avg'].gt(0).sum())} rows")
    print(f"\n  Output:")
    print(f"    data/processed/features_clean.csv")
    print(f"\n  Next step: Phase 5 — NLP Pipeline")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()