# src/features.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 6: Feature Engineering
#
# HOW TO RUN:
#   python src/features.py
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
log = logging.getLogger("biosignal.features")


# ═══════════════════════════════════════════════════════════════
# DISEASE PROFILES
# ═══════════════════════════════════════════════════════════════

DISEASE_RISK_PROFILE = {
    "ebola":        0.9,  "marburg":      0.9,
    "plague":       0.8,  "lassa fever":  0.7,
    "rift valley":  0.7,  "yellow fever": 0.65,
    "cholera":      0.6,  "mpox":         0.55,
    "dengue":       0.5,  "meningitis":   0.5,
    "influenza":    0.5,  "covid":        0.85,
    "measles":      0.45, "polio":        0.4,
    "typhoid":      0.35,
}

DISEASE_FATALITY_WEIGHT = {
    "ebola":        0.9,  "marburg":      0.88,
    "plague":       0.75, "lassa fever":  0.65,
    "rift valley":  0.6,  "yellow fever": 0.55,
    "cholera":      0.45, "mpox":         0.3,
    "dengue":       0.35, "meningitis":   0.5,
    "influenza":    0.4,  "covid":        0.7,
    "measles":      0.3,  "polio":        0.25,
    "typhoid":      0.2,
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
}

# Columns derived from or strongly correlated with the target
# Must NOT be used as ML input features
TARGET_LEAKAGE_COLS = [
    "disease_escalation_rate",
    "combined_risk_score",
    "estimated_lead_days",
    "lead_time_encoded",
]

# Nearly constant columns — add no signal to ML
LOW_VARIANCE_COLS = [
    "used_real_text",
    "used_fallback_text",
    "text_available",
    "source_is_who",
    "has_cdc_mention",
    "containment_language_flag",
]


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def add_disease_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding disease risk features...")

    df["disease_risk_score"]      = df["disease"].map(DISEASE_RISK_PROFILE).fillna(0.5)
    df["disease_fatality_weight"] = df["disease"].map(DISEASE_FATALITY_WEIGHT).fillna(0.4)

    # disease_escalation_rate is computed from the target — leakage
    # kept in full df for analysis only, dropped from ML input
    df["disease_escalation_rate"] = df.groupby("disease")["escalated"].transform("mean").round(3)

    disease_counts               = df.groupby("disease").size()
    df["disease_outbreak_count"] = df["disease"].map(disease_counts).fillna(0).astype(int)

    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding seasonal features...")

    def is_peak(row):
        peaks = DISEASE_PEAK_MONTHS.get(row["disease"], [])
        return 1 if row["month"] in peaks else 0

    df["is_peak_season"] = df.apply(is_peak, axis=1)
    df["season_risk"]    = df["is_peak_season"] * 0.3
    return df


def add_outbreak_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding outbreak frequency features...")

    country_counts               = df.groupby("country").size()
    df["country_outbreak_count"] = df["country"].map(country_counts).fillna(0).astype(int)

    disease_country_counts      = df.groupby(["disease", "country"]).size()
    df["disease_country_count"] = df.apply(
        lambda r: disease_country_counts.get((r["disease"], r["country"]), 0), axis=1
    ).astype(int)

    df["is_endemic_country"] = (df["disease_country_count"] >= 3).astype(int)
    return df


def add_case_severity_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding case severity features...")

    cases  = df["cases_total"].fillna(0).astype(float)
    deaths = df["deaths"].fillna(0).astype(float)

    df["case_fatality_ratio"] = np.where(cases > 0, (deaths / cases).round(4), 0.0)
    df["has_deaths"]          = (deaths > 0).astype(int)
    df["has_cases"]           = (cases > 0).astype(int)
    df["log_cases"]           = np.log1p(cases).round(4)
    df["log_deaths"]          = np.log1p(deaths).round(4)
    df["cases_per_death"]     = np.where(deaths > 0, (cases / deaths).round(2), 0.0)
    return df


def add_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding spike detection features...")

    df["has_trend_data"] = (df["search_volume_avg"] > 0).astype(int)
    df["spike_alert"]    = (df["spike_ratio_max"] >= config.SPIKE_RATIO_HIGH).astype(int)
    df["spike_warning"]  = (df["spike_ratio_max"] >= config.SPIKE_RATIO_MEDIUM).astype(int)

    # combined_risk_score — exploratory only, NOT an ML input
    # Uses disease profile + severity + spike only (no deaths/cases)
    df["combined_risk_score"] = (
        df["disease_risk_score"] * 0.5 +
        df["severity_score"] / 2 * 0.3 +
        df["spike_ratio_max"].clip(0, 5) / 5 * 0.2
    ).round(4)

    return df


def add_lead_time_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Adding lead time estimation features...")

    base_lead       = config.LEAD_TIME_AVG
    fatality_weight = df["disease_fatality_weight"].fillna(0.5)
    severity_score  = df["severity_score"].fillna(0)

    # estimated_lead_days — exploratory only, NOT an ML input
    df["estimated_lead_days"] = (
        base_lead
        - (fatality_weight * 8).round(0)
        - (severity_score * 2).round(0)
    ).clip(3, 30).astype(int)

    df["lead_time_category"] = (df["estimated_lead_days"] >= 14).map({True: "early", False: "medium"})
    df["lead_time_encoded"]  = (df["estimated_lead_days"] >= 14).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Encoding categorical features...")

    disease_order        = sorted(df["disease"].unique())
    disease_map          = {d: i for i, d in enumerate(disease_order)}
    df["disease_encoded"]     = df["disease"].map(disease_map).fillna(-1).astype(int)
    df["season_encoded"]      = df["season"].map({"winter":0,"spring":1,"summer":2,"autumn":3}).fillna(0).astype(int)
    df["spike_level_encoded"] = df["spike_level_top"].map({"normal":0,"watch":1,"warning":2,"alert":3}).fillna(0).astype(int)

    return df


def check_low_variance(df: pd.DataFrame, candidate_cols: list, threshold: float = 0.95) -> list:
    """Returns columns where top value appears in >= threshold fraction of rows."""
    low_var = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        top_pct = df[col].value_counts(normalize=True).iloc[0]
        if top_pct >= threshold:
            low_var.append(col)
    return low_var


def build_feature_quality_report(
    df: pd.DataFrame,
    selected_features: list,
    dropped_leakage: list,
    dropped_low_var: list,
) -> pd.DataFrame:
    """Builds a quality report showing why each feature was kept or dropped."""
    rows     = []
    all_cols = [c for c in df.columns if c != "escalated"]

    for col in all_cols:
        try:
            variance = round(float(df[col].var()), 6) if pd.api.types.is_numeric_dtype(df[col]) else None
        except Exception:
            variance = None

        if col in selected_features:
            selected = "yes"
            reason   = ""
        elif col in dropped_leakage:
            selected = "no"
            reason   = "target leakage"
        elif col in dropped_low_var:
            selected = "no"
            reason   = "low variance"
        else:
            selected = "no"
            reason   = "non-numeric or metadata"

        rows.append({
            "feature":         col,
            "dtype":           str(df[col].dtype),
            "missing_count":   int(df[col].isnull().sum()),
            "unique_values":   int(df[col].nunique()),
            "variance":        variance,
            "selected_for_ml": selected,
            "drop_reason":     reason,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_feature_engineering():

    # Step 1 — Load
    log.info("─" * 55)
    log.info("STEP 1 — Loading NLP features")
    log.info("─" * 55)

    in_path = config.PROCESSED_DIR / "features_nlp.csv"
    if not in_path.exists():
        log.error(f"Not found: {in_path} — run nlp_pipeline.py first")
        return pd.DataFrame(), pd.DataFrame(), [], []

    df = pd.read_csv(in_path)
    log.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 2 — Engineer features
    log.info("\n" + "─" * 55)
    log.info("STEP 2 — Engineering features")
    log.info("─" * 55)

    df = add_disease_risk_features(df)
    df = add_seasonal_features(df)
    df = add_outbreak_frequency_features(df)
    df = add_case_severity_features(df)
    df = add_spike_features(df)
    df = add_lead_time_features(df)
    df = encode_categoricals(df)

    log.info(f"  Total columns after engineering: {len(df.columns)}")

    # Step 3 — Select ML features
    log.info("\n" + "─" * 55)
    log.info("STEP 3 — Selecting ML input features")
    log.info("─" * 55)

    non_feature_cols = [
        "escalated",
        "date", "link", "source",
        "title", "summary",
        "top_location_entities",
        "disease", "country",
        "severity", "season",
        "spike_level_top",
        "lead_time_category",
        "disease_escalation_rate",
    ]

    actual_low_var = check_low_variance(df, LOW_VARIANCE_COLS, threshold=0.95)
    log.info(f"  Low-variance cols detected : {actual_low_var}")

    drop_from_features = set(non_feature_cols + TARGET_LEAKAGE_COLS + actual_low_var)

    all_numeric  = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in all_numeric
        if c not in drop_from_features and c != "escalated"
    ]

    log.info(f"  Input ML features    : {len(feature_cols)}")
    log.info(f"  Leakage cols dropped : {[c for c in TARGET_LEAKAGE_COLS if c in df.columns]}")
    log.info(f"  Low-var cols dropped : {actual_low_var}")

    # Step 4 — Build ML-ready dataframe
    ml_df = df[feature_cols + ["escalated"]].copy().fillna(0)

    # Step 5 — Validate
    log.info("\n" + "─" * 55)
    log.info("STEP 4 — Validation")
    log.info("─" * 55)

    assert "escalated" in ml_df.columns, "Target column missing!"
    assert len(ml_df) == len(df),        "Row count changed!"

    null_count = ml_df[feature_cols].isnull().sum().sum()
    log.info(f"  Rows                 : {len(ml_df)} ✓")
    log.info(f"  Input features       : {len(feature_cols)} ✓")
    log.info(f"  Null values          : {null_count} ✓")
    log.info(f"  Escalated (1)        : {ml_df['escalated'].sum()}")
    log.info(f"  Contained (0)        : {(ml_df['escalated']==0).sum()}")

    # Step 6 — Save
    log.info("\n" + "─" * 55)
    log.info("STEP 5 — Saving outputs")
    log.info("─" * 55)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(config.PROCESSED_DIR / "features_engineered.csv", index=False)
    log.info(f"  Full dataset         → features_engineered.csv")

    ml_df.to_csv(config.PROCESSED_DIR / "features_ml_ready.csv", index=False)
    log.info(f"  ML dataset           → features_ml_ready.csv")

    feat_col_path = config.PROCESSED_DIR / "feature_columns.txt"
    with open(feat_col_path, "w") as f:
        f.write("\n".join(feature_cols))
    log.info(f"  Feature list         → feature_columns.txt")

    quality_report = build_feature_quality_report(
        df,
        selected_features=feature_cols,
        dropped_leakage=[c for c in TARGET_LEAKAGE_COLS if c in df.columns],
        dropped_low_var=actual_low_var,
    )
    quality_report.to_csv(config.PROCESSED_DIR / "feature_quality_report.csv", index=False)
    log.info(f"  Quality report       → feature_quality_report.csv")

    return df, ml_df, feature_cols, actual_low_var


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 6: Feature Engineering")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    result = run_feature_engineering()

    if len(result[0]) == 0:
        print("Pipeline failed. Check errors above.")
        return

    df, ml_df, feature_cols, actual_low_var = result

    # ── Safe summary stats ────────────────────────────────────
    # Always read from df (full) not ml_df for dropped columns

    def safe_mean(frame, col):
        """Safely get mean of a column from a dataframe."""
        return f"{frame[col].mean():.3f}" if col in frame.columns else "N/A"

    def safe_sum(frame, col):
        """Safely get sum of a column from a dataframe."""
        return int(frame[col].sum()) if col in frame.columns else "N/A"

    print("\n" + "═" * 55)
    print("  Phase 6 Complete")
    print("═" * 55)
    print(f"  Rows                   : {len(ml_df)}")
    print(f"  Input ML features      : {len(feature_cols)}")
    print(f"  Target column          : escalated")
    print(f"  Escalated (label=1)    : {int(ml_df['escalated'].sum())}")
    print(f"  Contained (label=0)    : {int((ml_df['escalated']==0).sum())}")
    print(f"  Dropped leakage cols   : {len([c for c in TARGET_LEAKAGE_COLS if c in df.columns])}")
    print(f"  Dropped low-var cols   : {len(actual_low_var)}")
    print(f"\n  Key feature stats:")
    # combined_risk_score is in df (full), not ml_df — read from df
    print(f"    Avg combined risk    : {safe_mean(df,  'combined_risk_score')}")
    print(f"    Peak season rows     : {safe_sum(ml_df, 'is_peak_season')}")
    print(f"    Endemic country rows : {safe_sum(ml_df, 'is_endemic_country')}")
    print(f"    Avg disease risk     : {safe_mean(ml_df, 'disease_risk_score')}")
    print(f"    Spike alerts         : {safe_sum(ml_df, 'spike_alert')}")
    print(f"\n  Output:")
    print(f"    data/processed/features_engineered.csv")
    print(f"    data/processed/features_ml_ready.csv")
    print(f"    data/processed/feature_columns.txt")
    print(f"    data/processed/feature_quality_report.csv")
    print(f"\n  Next step: Phase 7 — XGBoost Model Training")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()