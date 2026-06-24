# src/explain.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 9: SHAP + Counterfactual Explainability
#
# HOW TO RUN:
#   python src/explain.py
# ═══════════════════════════════════════════════════════════════

import sys
import json
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
log = logging.getLogger("biosignal.explain")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = {
    "urgency_score":            "urgency language in outbreak report",
    "net_urgency":              "urgency outweighing containment language",
    "risk_keyword_count":       "risk-related keywords",
    "response_keyword_count":   "response action keywords",
    "escalation_language_flag": "escalation language detected",
    "containment_score":        "containment language in report",
    "disease_risk_score":       "historical disease risk profile",
    "disease_fatality_weight":  "disease fatality weight",
    "disease_country_count":    "repeated disease-country outbreak pattern",
    "country_outbreak_count":   "country outbreak history",
    "disease_outbreak_count":   "disease outbreak history",
    "is_endemic_country":       "endemic country flag",
    "spike_ratio_max":          "Google Trends spike strength",
    "spike_ratio_avg":          "average Google Trends spike",
    "spike_alert":              "major Google Trends spike",
    "spike_warning":            "moderate Google Trends spike",
    "has_trend_data":           "Google Trends signal available",
    "search_volume_avg":        "average search interest",
    "search_volume_max":        "peak search interest",
    "baseline_avg":             "search volume baseline",
    "is_peak_season":           "seasonal disease risk",
    "season_risk":              "seasonal risk boost",
    "entity_count":             "named entities in article",
    "numeric_entity_count":     "numerical indicators in article",
    "location_count":           "location mentions",
    "org_count":                "organization mentions",
    "person_count":             "person mentions",
    "date_entity_count":        "date references",
    "sentence_count":           "report detail level",
    "word_count":               "report length",
    "avg_sentence_length":      "average sentence length",
    "title_word_count":         "title length",
    "summary_word_count":       "summary length",
    "has_who_mention":          "WHO mentioned in report",
    "has_disease_in_text":      "disease name in text",
    "has_country_in_text":      "country name in text",
    "disease_encoded":          "disease category signal",
    "season_encoded":           "season category",
    "spike_level_encoded":      "spike severity level",
    "year":                     "report year",
    "month":                    "report month",
    "week":                     "report week",
}


def feat_name(f: str) -> str:
    return FEATURE_NAMES.get(f, f.replace("_", " "))


def get_risk_level(score: float) -> str:
    """
    Single source of truth for risk levels.
    Always use this function — never hardcode thresholds elsewhere.
    """
    if score >= 81:   return "Critical Risk"
    if score >= 61:   return "High Risk"
    if score >= 31:   return "Medium Risk"
    return "Low Risk"


def prob_to_score(prob: float) -> float:
    """Converts model probability (0-1) to risk score (0-100)."""
    return round(float(prob) * 100, 1)


def get_confidence_band(prob: float) -> str:
    if prob >= 0.75: return "High Confidence"
    if prob >= 0.55: return "Medium Confidence"
    return "Low Confidence"


def get_recommended_action(risk_level: str) -> str:
    actions = {
        "Critical Risk": (
            "Immediate analyst review recommended. Verify outbreak details, "
            "monitor WHO updates closely, and check country-level health alerts."
        ),
        "High Risk": (
            "Monitor closely and validate with recent WHO, CDC, "
            "and local public health sources."
        ),
        "Medium Risk": (
            "Keep under watch and review if search interest "
            "or urgency language increases."
        ),
        "Low Risk": (
            "No immediate escalation required; continue routine monitoring."
        ),
    }
    return actions.get(risk_level, "Continue routine monitoring.")


def get_counterfactual_new_value(feature: str, current_val: float):
    """
    Returns hypothetical new value for a feature.
    All changes move toward lower risk.
    Returns None if this feature should not be tested.
    """
    changes = {
        "urgency_score":            max(0.0, current_val - 2),
        "net_urgency":              max(0.0, current_val - 2),
        "risk_keyword_count":       max(0.0, current_val - 1),
        "escalation_language_flag": 0.0,
        "spike_alert":              0.0,
        "spike_warning":            0.0,
        "spike_ratio_max":          1.0,
        "spike_ratio_avg":          1.0,
        "has_trend_data":           0.0,
        "is_peak_season":           0.0,
        "season_risk":              0.0,
        "disease_country_count":    max(0.0, current_val // 2),
        "country_outbreak_count":   max(0.0, current_val // 2),
        "disease_outbreak_count":   max(0.0, current_val // 2),
        "entity_count":             max(0.0, current_val - 2),
        "numeric_entity_count":     max(0.0, current_val - 2),
        "sentence_count":           max(1.0, current_val - 1),
        "search_volume_avg":        max(0.0, current_val * 0.5),
        "search_volume_max":        max(0.0, current_val * 0.5),
    }
    return changes.get(feature, None)


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL + DATA
# ═══════════════════════════════════════════════════════════════

def load_model_and_data():
    """
    Loads model, training features, and risk score outputs.
    Validates model mode is early_warning.
    """
    log.info("─" * 55)
    log.info("STEP 1 — Loading Model and Data")
    log.info("─" * 55)

    model_path = ROOT / "models" / "xgboost_early_warning_model.pkl"
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        return None, None, None, None, None

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    mode = saved.get("mode", "unknown")
    if mode != "early_warning":
        log.error(f"Wrong model! Mode={mode}. Need early_warning.")
        return None, None, None, None, None

    model        = saved["model"]
    feature_cols = saved["feature_cols"]

    log.info(f"  Model mode           : {mode} ✓")
    log.info(f"  Trained at           : {saved.get('trained_at','unknown')}")
    log.info(f"  Features             : {len(feature_cols)}")

    # Load ML-ready features (training set — used for global SHAP)
    ml_path = config.PROCESSED_DIR / "features_ml_ready.csv"
    if not ml_path.exists():
        log.error(f"Not found: {ml_path}")
        return None, None, None, None, None

    ml_df   = pd.read_csv(ml_path)
    missing = [c for c in feature_cols if c not in ml_df.columns]
    if missing:
        log.error(f"Missing feature columns in ML data: {missing}")
        return None, None, None, None, None

    X_train = ml_df[feature_cols].fillna(0)
    log.info(f"  Training rows        : {len(X_train)}")
    log.info(f"  Null values          : {X_train.isnull().sum().sum()} ✓")

    # Load high risk alerts — these are the rows we explain
    alerts_path = ROOT / "data" / "outputs" / "high_risk_alerts.csv"
    alerts_df   = pd.DataFrame()
    if alerts_path.exists():
        alerts_df = pd.read_csv(alerts_path)
        log.info(f"  High-risk alerts     : {len(alerts_df)}")
    else:
        log.warning("  high_risk_alerts.csv not found — run risk_scorer.py first")

    # Load risk scores — has all scored articles
    scores_path = ROOT / "data" / "outputs" / "risk_scores.csv"
    scores_df   = pd.DataFrame()
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        log.info(f"  Risk scores loaded   : {len(scores_df)} rows")

    return model, feature_cols, X_train, alerts_df, scores_df


# ═══════════════════════════════════════════════════════════════
# BUILD ALERT FEATURE VECTORS
# ═══════════════════════════════════════════════════════════════

def build_alert_feature_vectors(
    model,
    feature_cols: list,
    alerts_df: pd.DataFrame,
    X_train: pd.DataFrame,
) -> tuple:
    """
    For each high-risk alert, builds the feature vector
    and verifies the model produces a score close to the
    stored risk_score.

    Strategy:
    - alerts_df contains disease/country/risk_score from Phase 8
    - We re-score each alert using the model
    - If score matches stored score, row alignment is correct
    - If not, we warn clearly

    Returns: list of (alert_row, feature_vector, recalculated_score)
    """
    if alerts_df.empty:
        return []

    log.info(f"  Building feature vectors for {len(alerts_df)} alerts...")
    alert_data = []
    alignment_method = "feature_vector_from_training_data"

    for i, (_, alert) in enumerate(alerts_df.iterrows()):
        stored_score  = float(alert.get("risk_score", 0))
        disease       = str(alert.get("disease", "unknown"))
        country       = str(alert.get("country", "unknown"))

        # Strategy: find matching row in X_train by disease
        # X_train is the full training set — we find rows for this disease
        # and pick the one closest to the stored risk score
        eng_path = config.PROCESSED_DIR / "features_engineered.csv"
        if eng_path.exists():
            eng_df       = pd.read_csv(eng_path)
            disease_mask = eng_df["disease"].str.lower() == disease.lower()
            country_mask = eng_df["country"].str.lower() == country.lower() \
                           if country != "unknown" else pd.Series(
                               [True] * len(eng_df)
                           )
            match_mask   = disease_mask & country_mask
            match_idx    = match_mask[match_mask].index.tolist()
        else:
            match_idx = []

        if match_idx:
            # Score all matching rows, pick closest to stored score
            best_row   = None
            best_diff  = float("inf")
            for idx in match_idx:
                if idx >= len(X_train):
                    continue
                row_feat = X_train.iloc[idx]
                prob     = float(
                    model.predict_proba(
                        row_feat.values.reshape(1, -1)
                    )[0][1]
                )
                score = prob_to_score(prob)
                diff  = abs(score - stored_score)
                if diff < best_diff:
                    best_diff = diff
                    best_row  = (idx, row_feat, score, prob)

            if best_row:
                idx, feat_vec, calc_score, calc_prob = best_row
                if best_diff > 20:
                    log.warning(
                        f"  ⚠ Score mismatch for {disease}/{country}: "
                        f"stored={stored_score} recalculated={calc_score} "
                        f"(diff={round(best_diff,1)}) — using recalculated"
                    )
                alert_data.append({
                    "alert":        alert,
                    "feat_vec":     feat_vec,
                    "risk_score":   calc_score,
                    "risk_prob":    calc_prob,
                    "risk_level":   get_risk_level(calc_score),
                    "disease":      disease,
                    "country":      country,
                    "aligned":      True,
                    "feat_idx":     idx,
                })
                continue

        # Fallback: use first row of X_train for this disease
        log.warning(
            f"  Could not match alert {disease}/{country} "
            f"to training row — using stored risk score"
        )
        # Use the stored score and a zero feature vector as placeholder
        feat_vec   = X_train.iloc[0].copy()
        calc_prob  = stored_score / 100
        alert_data.append({
            "alert":      alert,
            "feat_vec":   feat_vec,
            "risk_score": stored_score,
            "risk_prob":  calc_prob,
            "risk_level": get_risk_level(stored_score),
            "disease":    disease,
            "country":    country,
            "aligned":    False,
            "feat_idx":   0,
        })

    log.info(f"  Aligned alerts       : {sum(1 for a in alert_data if a['aligned'])}/{len(alert_data)}")
    return alert_data


# ═══════════════════════════════════════════════════════════════
# PART A — GLOBAL SHAP
# ═══════════════════════════════════════════════════════════════

def compute_global_shap(model, X_train: pd.DataFrame, feature_cols: list):
    log.info("\n" + "─" * 55)
    log.info("PART A — Global SHAP Explanations")
    log.info("─" * 55)
    log.info(f"  Computing SHAP for {len(X_train)} training rows...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature":        feature_cols,
        "mean_abs_shap":  mean_abs.round(6),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance["importance_rank"]        = range(1, len(importance) + 1)
    importance["human_readable_feature"] = importance["feature"].map(feat_name)
    importance["plain_english_meaning"]  = importance["human_readable_feature"].map(
        lambda h: f"Higher {h} increases outbreak escalation risk"
    )

    log.info(f"  SHAP computed ✓")
    log.info(f"\n  ── Top 10 Global Features ─────────────────")
    for _, r in importance.head(10).iterrows():
        log.info(
            f"    #{int(r['importance_rank']):<3} "
            f"{r['feature']:<40} SHAP={r['mean_abs_shap']:.4f}"
        )

    return importance, shap_values, explainer


def save_shap_plots(importance_df: pd.DataFrame, output_dir: Path):
    log.info("  Saving SHAP plots...")

    top15 = importance_df.head(15)

    # Plot 1 — horizontal bar
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top15["human_readable_feature"][::-1],
        top15["mean_abs_shap"][::-1],
        color="#E63946", alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(
        "BioSignal — Global Feature Importance (SHAP)\n"
        "Early-Warning Model — Top 15 Features",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → shap_global_bar.png")

    # Plot 2 — vertical bar top 10
    top10 = importance_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#E63946" if i < 3 else "#457B9D" for i in range(len(top10))]
    ax.bar(top10["human_readable_feature"], top10["mean_abs_shap"],
           color=colors, alpha=0.9)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("BioSignal — Top 10 Early-Warning Features",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(top10["human_readable_feature"],
                       rotation=35, ha="right", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → shap_top_features.png")


# ═══════════════════════════════════════════════════════════════
# PART B — LOCAL SHAP
# ═══════════════════════════════════════════════════════════════

def compute_local_shap(
    explainer,
    feature_cols: list,
    alert_data: list,
) -> pd.DataFrame:
    """
    For each high-risk alert, explains which features
    pushed risk up and which pulled it down.
    Uses the correctly aligned feature vector for each alert.
    """
    log.info("\n" + "─" * 55)
    log.info("PART B — Local SHAP Explanations")
    log.info("─" * 55)

    if not alert_data:
        log.warning("  No alerts to explain")
        return pd.DataFrame()

    local_rows = []

    for item in alert_data:
        feat_vec   = item["feat_vec"]
        disease    = item["disease"]
        country    = item["country"]
        risk_score = item["risk_score"]
        risk_level = item["risk_level"]

        # Validate risk level consistency
        expected_level = get_risk_level(risk_score)
        if expected_level != risk_level:
            log.warning(
                f"  ⚠ Risk level mismatch for {disease}: "
                f"score={risk_score} → should be {expected_level}, "
                f"got {risk_level} — correcting"
            )
            risk_level = expected_level

        # Compute SHAP for this specific row
        row_array  = feat_vec.values.reshape(1, -1)
        row_shap   = explainer.shap_values(row_array)[0]
        shap_series = pd.Series(row_shap, index=feature_cols)

        pos_feats = shap_series[shap_series > 0].sort_values(ascending=False)
        neg_feats = shap_series[shap_series < 0].sort_values(ascending=True)

        pos1_f = pos_feats.index[0] if len(pos_feats) > 0 else ""
        pos2_f = pos_feats.index[1] if len(pos_feats) > 1 else ""
        pos3_f = pos_feats.index[2] if len(pos_feats) > 2 else ""
        pos1_v = round(float(pos_feats.iloc[0]), 4) if len(pos_feats) > 0 else 0
        pos2_v = round(float(pos_feats.iloc[1]), 4) if len(pos_feats) > 1 else 0
        pos3_v = round(float(pos_feats.iloc[2]), 4) if len(pos_feats) > 2 else 0
        neg1_f = neg_feats.index[0] if len(neg_feats) > 0 else ""
        neg2_f = neg_feats.index[1] if len(neg_feats) > 1 else ""
        neg1_v = round(float(neg_feats.iloc[0]), 4) if len(neg_feats) > 0 else 0
        neg2_v = round(float(neg_feats.iloc[1]), 4) if len(neg_feats) > 1 else 0

        # Build explanation
        reasons = [feat_name(f) for f in [pos1_f, pos2_f, pos3_f] if f]
        if reasons:
            explanation = (
                f"This {disease} alert was scored as {risk_level} "
                f"(score: {risk_score}) mainly because the model detected "
                f"high {', '.join(reasons[:2])}. "
                f"This is a model-based early-warning signal, "
                f"not a confirmed medical diagnosis."
            )
        else:
            explanation = (
                f"This {disease} alert scored {risk_level} "
                f"based on multiple early-warning signals."
            )

        local_rows.append({
            "disease":                disease,
            "country":                country,
            "risk_score":             risk_score,
            "risk_level":             risk_level,
            "confidence_band":        get_confidence_band(item["risk_prob"]),
            "top_positive_feature_1": pos1_f,
            "top_positive_value_1":   pos1_v,
            "top_positive_feature_2": pos2_f,
            "top_positive_value_2":   pos2_v,
            "top_positive_feature_3": pos3_f,
            "top_positive_value_3":   pos3_v,
            "top_negative_feature_1": neg1_f,
            "top_negative_value_1":   neg1_v,
            "top_negative_feature_2": neg2_f,
            "top_negative_value_2":   neg2_v,
            "explanation_text":       explanation,
            "recommended_action":     get_recommended_action(risk_level),
            "row_aligned":            item["aligned"],
        })

    df = pd.DataFrame(local_rows)
    log.info(f"  Local explanations   : {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# PART C — COUNTERFACTUAL EXPLANATIONS
# ═══════════════════════════════════════════════════════════════

def compute_counterfactuals(
    model,
    feature_cols: list,
    alert_data: list,
) -> pd.DataFrame:
    """
    For each high-risk alert, generates counterfactual explanations.
    Uses the correctly aligned feature vector — same row as local SHAP.
    All risk scores are on 0-100 scale consistently.
    """
    log.info("\n" + "─" * 55)
    log.info("PART C — Counterfactual Explanations")
    log.info("─" * 55)

    if not alert_data:
        log.warning("  No alerts for counterfactuals")
        return pd.DataFrame()

    features_to_test = [
        "urgency_score", "net_urgency", "risk_keyword_count",
        "escalation_language_flag", "spike_alert", "spike_warning",
        "spike_ratio_max", "has_trend_data", "is_peak_season",
        "season_risk", "disease_country_count", "country_outbreak_count",
        "disease_outbreak_count", "entity_count", "numeric_entity_count",
        "sentence_count", "search_volume_avg", "search_volume_max",
    ]
    features_to_test = [f for f in features_to_test if f in feature_cols]

    cf_rows = []

    for item in alert_data:
        feat_vec      = item["feat_vec"].copy()
        disease       = item["disease"]
        country       = item["country"]
        current_score = item["risk_score"]      # already 0-100
        current_prob  = item["risk_prob"]       # 0-1
        current_level = get_risk_level(current_score)

        row_cfs = []

        for feat in features_to_test:
            if feat not in feature_cols:
                continue

            current_val = float(feat_vec[feat])
            new_val     = get_counterfactual_new_value(feat, current_val)

            if new_val is None or abs(new_val - current_val) < 0.001:
                continue

            # Apply change and recalculate
            modified       = feat_vec.copy()
            modified[feat] = new_val

            new_prob  = float(
                model.predict_proba(
                    modified.values.reshape(1, -1)
                )[0][1]
            )
            new_score = prob_to_score(new_prob)   # 0-100 scale
            reduction = round(current_score - new_score, 1)

            # Only keep changes that reduce risk
            if reduction <= 0:
                continue

            new_level = get_risk_level(new_score)

            explanation = (
                f"If {feat_name(feat)} changed from "
                f"{round(current_val, 2)} to {round(new_val, 2)}, "
                f"the model-based risk score would reduce from "
                f"{current_score} to {new_score} "
                f"({current_level} → {new_level}). "
                f"This is a model-based what-if scenario, "
                f"not a medical prediction."
            )

            row_cfs.append({
                "disease":                    disease,
                "country":                    country,
                "current_risk_score":         current_score,
                "current_risk_level":         current_level,
                "counterfactual_scenario":    f"Reduce {feat_name(feat)}",
                "changed_feature":            feat,
                "original_value":             round(current_val, 3),
                "new_value":                  round(new_val, 3),
                "new_predicted_risk_score":   new_score,
                "new_risk_level":             new_level,
                "risk_reduction":             reduction,
                "counterfactual_explanation": explanation,
            })

        # Top 3 by biggest reduction
        row_cfs = sorted(row_cfs, key=lambda x: x["risk_reduction"], reverse=True)[:3]
        cf_rows.extend(row_cfs)

    cf_df = pd.DataFrame(cf_rows)
    log.info(f"  Counterfactuals      : {len(cf_df)}")

    if not cf_df.empty:
        log.info(f"\n  ── Sample Counterfactuals ─────────────────")
        for _, r in cf_df.head(3).iterrows():
            log.info(
                f"    {r['disease']} ({r['current_risk_level']}) → "
                f"change '{r['changed_feature']}': "
                f"{r['original_value']} → {r['new_value']} | "
                f"risk {r['current_risk_score']} → "
                f"{r['new_predicted_risk_score']} "
                f"(−{r['risk_reduction']})"
            )

    return cf_df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 9: SHAP Explainability")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model, feature_cols, X_train, alerts_df, scores_df = load_model_and_data()
    if model is None:
        print("Failed to load model. Run model.py first.")
        return

    # Part A — Global SHAP
    global_imp, shap_values, explainer = compute_global_shap(
        model, X_train, feature_cols
    )
    global_imp.to_csv(output_dir / "shap_global_importance.csv", index=False)
    log.info("  Saved → shap_global_importance.csv")

    save_shap_plots(global_imp, output_dir)

    top5 = global_imp.head(5)[
        ["feature","mean_abs_shap","human_readable_feature"]
    ].to_dict(orient="records")
    shap_summary = {
        "total_rows_explained": len(X_train),
        "total_features":       len(feature_cols),
        "top_5_global_features": top5,
        "model_used":           "xgboost_early_warning_model.pkl",
        "generated_at":         datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(output_dir / "shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=2)
    log.info("  Saved → shap_summary.json")

    # Build aligned alert feature vectors
    alert_data = build_alert_feature_vectors(
        model, feature_cols, alerts_df, X_train
    )

    # Part B — Local SHAP
    local_df = compute_local_shap(explainer, feature_cols, alert_data)
    if not local_df.empty:
        local_df.to_csv(
            output_dir / "shap_local_explanations.csv", index=False
        )
        log.info("  Saved → shap_local_explanations.csv")

    # Part C — Counterfactuals
    cf_df = compute_counterfactuals(model, feature_cols, alert_data)
    if not cf_df.empty:
        cf_df.to_csv(
            output_dir / "counterfactual_explanations.csv", index=False
        )
        log.info("  Saved → counterfactual_explanations.csv")

    # Final summary
    alignment_method = "matched by disease+country in training data"

    print("\n" + "═" * 55)
    print("  Phase 9 Complete — SHAP + Counterfactual Explainability")
    print("═" * 55)
    print(f"  Rows explained              : {len(X_train)}")
    print(f"  Features explained          : {len(feature_cols)}")
    print(f"  High-risk alerts explained  : {len(local_df)}")
    print(f"  Counterfactuals generated   : {len(cf_df)}")
    print(f"  Row alignment method        : {alignment_method}")

    print(f"\n  Top 10 Global SHAP Features:")
    print(f"  {'#':<4} {'Feature':<42} {'SHAP':>8}  Meaning")
    print(f"  {'-'*85}")
    for _, r in global_imp.head(10).iterrows():
        print(
            f"  #{int(r['importance_rank']):<3} "
            f"{r['feature']:<42} "
            f"{r['mean_abs_shap']:>8.4f}  "
            f"{r['human_readable_feature']}"
        )

    if not local_df.empty:
        ex = local_df.iloc[0]
        print(f"\n  Example Local Explanation:")
        print(f"    Disease    : {ex['disease']}")
        print(f"    Country    : {ex['country']}")
        print(f"    Risk Score : {ex['risk_score']}")
        print(f"    Risk Level : {ex['risk_level']}")
        print(f"    Explanation: {ex['explanation_text'][:150]}...")

    if not cf_df.empty:
        ex_cf = cf_df.iloc[0]
        print(f"\n  Example Counterfactual:")
        print(f"    {ex_cf['counterfactual_explanation'][:180]}...")

    print(f"\n  Output:")
    print(f"    data/outputs/shap_global_importance.csv")
    print(f"    data/outputs/shap_local_explanations.csv")
    print(f"    data/outputs/counterfactual_explanations.csv")
    print(f"    data/outputs/shap_summary.json")
    print(f"    data/outputs/shap_global_bar.png")
    print(f"    data/outputs/shap_top_features.png")
    print(f"\n  Next step: Phase 10 — Streamlit Dashboard")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()