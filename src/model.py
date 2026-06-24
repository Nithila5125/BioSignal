# src/model.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Phase 7: XGBoost Model Training
#
# Two modes:
#   Diagnostic model     — all features (shows leakage)
#   Early-warning model  — removes outcome features (realistic)
#
# HOW TO RUN:
#   python src/model.py
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
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Suppress convergence and other sklearn warnings
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
log = logging.getLogger("biosignal.model")


# ═══════════════════════════════════════════════════════════════
# LEAKAGE FEATURES
# These are derived directly from the label formula:
#   escalated = 1 if deaths >= 10 or cases >= 100
# Using them as features = giving the model the answer
# ═══════════════════════════════════════════════════════════════

OUTCOME_FEATURES = [
    "severity_score",           # derived from deaths/cases
    "cases_total",              # directly in label formula
    "deaths",                   # directly in label formula
    "log_cases",                # log(cases_total)
    "log_deaths",               # log(deaths)
    "case_fatality_ratio",      # deaths / cases
    "has_deaths",               # deaths > 0
    "has_cases",                # cases > 0
    "cases_per_death",          # cases / deaths
    "outbreak_relevance_score", # correlated with label
]


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("─" * 55)
    log.info("STEP 1 — Loading ML-ready data")
    log.info("─" * 55)

    ml_path   = config.PROCESSED_DIR / "features_ml_ready.csv"
    feat_path = config.PROCESSED_DIR / "feature_columns.txt"

    if not ml_path.exists():
        log.error(f"Not found: {ml_path} — run features.py first")
        return None, None, None

    if not feat_path.exists():
        log.error(f"Not found: {feat_path} — run features.py first")
        return None, None, None

    df = pd.read_csv(ml_path)
    log.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    with open(feat_path, "r") as f:
        all_feature_cols = [l.strip() for l in f if l.strip()]

    # Safety check — escalated must never be a feature
    assert "escalated" not in all_feature_cols, \
        "escalated must not be in feature list!"

    X = df[all_feature_cols].copy()
    y = df["escalated"].copy()

    log.info(f"  All features         : {len(all_feature_cols)}")
    log.info(f"  Escalated (1)        : {y.sum()}")
    log.info(f"  Contained (0)        : {(y==0).sum()}")
    log.info(f"  Label balance        : {round(y.mean()*100,1)}%")

    return X, y, all_feature_cols


# ═══════════════════════════════════════════════════════════════
# STEP 2 — EVALUATE ONE MODEL
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    model, X_train, X_test, y_train, y_test, label: str
) -> dict:
    """Trains and evaluates one model. Returns metrics dict."""
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test,  y_pred),                  4)
    prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
    rec  = round(recall_score(y_test,    y_pred, zero_division=0), 4)
    f1   = round(f1_score(y_test,        y_pred, zero_division=0), 4)
    auc  = round(roc_auc_score(y_test,   y_pred_prob),             4)
    cm   = confusion_matrix(y_test, y_pred)

    log.info(f"\n  [{label}]")
    log.info(f"    Accuracy  : {acc}")
    log.info(f"    Precision : {prec}")
    log.info(f"    Recall    : {rec}")
    log.info(f"    F1        : {f1}")
    log.info(f"    ROC-AUC   : {auc}")
    log.info(
        f"    CM → TN:{cm[0][0]} FP:{cm[0][1]} "
        f"FN:{cm[1][0]} TP:{cm[1][1]}"
    )

    if f1 > 0.98:
        log.warning(
            f"    ⚠ F1={f1} is suspiciously high — possible leakage"
        )

    return {
        "model":     label,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


def cross_validate_model(model, X, y, label: str) -> dict:
    """5-fold stratified cross-validation for one model."""
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s  = cross_val_score(model, X, y, cv=cv, scoring="f1")
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    log.info(
        f"    CV F1  : {[round(s,3) for s in f1s]} "
        f"→ mean {round(f1s.mean(),4)}"
    )
    log.info(
        f"    CV AUC : {[round(s,3) for s in aucs]} "
        f"→ mean {round(aucs.mean(),4)}"
    )

    return {
        "cv_f1_mean":    round(float(f1s.mean()),  4),
        "cv_auc_mean":   round(float(aucs.mean()), 4),
        "cv_f1_scores":  [round(float(s), 4) for s in f1s],
        "cv_auc_scores": [round(float(s), 4) for s in aucs],
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3 — RUN ONE FULL MODE
# ═══════════════════════════════════════════════════════════════

def run_mode(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    mode: str,
) -> tuple:
    """
    Runs training + evaluation for one mode.
    mode = "diagnostic" or "early_warning"

    Returns: (all_results, importances, predictions, cv_results, use_cols)
    """
    log.info("\n" + "═" * 55)
    log.info(f"MODE: {mode.upper()}")
    log.info("═" * 55)

    # For early-warning — remove outcome/leakage features
    if mode == "early_warning":
        removed  = [c for c in OUTCOME_FEATURES if c in feature_cols]
        use_cols = [c for c in feature_cols if c not in OUTCOME_FEATURES]
        log.info(f"  Removed outcome features ({len(removed)}): {removed}")
        log.info(f"  Remaining features: {len(use_cols)}")
    else:
        use_cols = feature_cols
        removed  = []
        log.info(f"  Using all features: {len(use_cols)}")

    X_mode = X[use_cols]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_mode, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    log.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale pos weight for XGBoost
    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 3)

    # Three models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            solver="saga",          # faster convergence than lbfgs
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    # Evaluate all models
    log.info(f"\n  ── Model Comparison ───────────────────────")
    all_results = []
    for model_name, model in models.items():
        result       = evaluate_model(
            model, X_train, X_test, y_train, y_test,
            label=model_name,
        )
        result["mode"] = mode
        all_results.append(result)

    # Cross-validate XGBoost
    log.info(f"\n  ── XGBoost 5-Fold CV ──────────────────────")
    xgb_model = models["XGBoost"]
    xgb_model.fit(X_train, y_train)
    cv_results = cross_validate_model(xgb_model, X_mode, y, label="XGBoost")

    # Feature importances
    importances = pd.Series(
        xgb_model.feature_importances_, index=use_cols
    ).sort_values(ascending=False)

    log.info(f"\n  ── Top 10 Feature Importances ─────────────")
    for feat, imp in importances.head(10).items():
        log.info(f"    {feat:<40} {round(imp, 4)}")

    # Test set predictions
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": xgb_model.predict(X_test),
        "y_prob": y_pred_prob.round(4),
    })

    # Save model
    model_dir  = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_name = f"xgboost_{mode}_model.pkl"
    model_path = model_dir / model_name

    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        xgb_model,
            "feature_cols": use_cols,
            "mode":         mode,
            "removed_cols": removed,
            "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        }, f)
    log.info(f"\n  Model saved → models/{model_name}")

    return all_results, importances, predictions, cv_results, use_cols


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print(f"  {config.PROJECT_NAME} — Phase 7: Model Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    # Load data
    X, y, all_feature_cols = load_data()
    if X is None:
        print("Failed to load data.")
        return

    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_comparison_rows = []
    all_metrics         = {}

    # ── Mode 1: Diagnostic ───────────────────────────────────
    log.info("\n" + "─" * 55)
    log.info("RUNNING DIAGNOSTIC MODE (all features)")
    log.info("─" * 55)

    diag_results, diag_imp, diag_preds, diag_cv, diag_cols = run_mode(
        X, y, all_feature_cols, mode="diagnostic"
    )
    all_comparison_rows.extend(diag_results)

    # Save diagnostic importances
    diag_imp_df = diag_imp.reset_index()
    diag_imp_df.columns = ["feature", "importance"]
    diag_imp_df.to_csv(
        output_dir / "diagnostic_feature_importances.csv", index=False
    )

    all_metrics["diagnostic"] = {
        "results":    diag_results,
        "cv":         diag_cv,
        "n_features": len(diag_cols),
    }

    # ── Mode 2: Early-Warning ─────────────────────────────────
    log.info("\n" + "─" * 55)
    log.info("RUNNING EARLY-WARNING MODE (outcome features removed)")
    log.info("─" * 55)

    ew_results, ew_imp, ew_preds, ew_cv, ew_cols = run_mode(
        X, y, all_feature_cols, mode="early_warning"
    )
    all_comparison_rows.extend(ew_results)

    # Save early-warning importances
    ew_imp_df = ew_imp.reset_index()
    ew_imp_df.columns = ["feature", "importance"]
    ew_imp_df.to_csv(
        output_dir / "early_warning_feature_importances.csv", index=False
    )

    # Save early-warning predictions
    ew_preds.to_csv(
        output_dir / "predictions_early_warning.csv", index=False
    )

    all_metrics["early_warning"] = {
        "results":    ew_results,
        "cv":         ew_cv,
        "n_features": len(ew_cols),
        "removed":    [c for c in OUTCOME_FEATURES if c in all_feature_cols],
    }

    # ── Save comparison CSV ───────────────────────────────────
    pd.DataFrame(all_comparison_rows).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    # ── Save metrics JSON ─────────────────────────────────────
    all_metrics["project"]  = config.PROJECT_NAME
    all_metrics["tagline"]  = config.PROJECT_TAGLINE
    all_metrics["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(output_dir / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Get summary metrics ───────────────────────────────────
    ew_xgb = next(r for r in ew_results   if r["model"] == "XGBoost")
    d_xgb  = next(r for r in diag_results if r["model"] == "XGBoost")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Phase 7 Complete")
    print("═" * 55)

    print(f"\n  Diagnostic Model (all features — shows leakage):")
    print(f"    F1      : {d_xgb['f1']}")
    print(f"    ROC-AUC : {d_xgb['roc_auc']}")
    if d_xgb["f1"] > 0.98:
        print(f"    ⚠ Metrics too high — outcome features cause leakage")

    print(f"\n  Early-Warning Model (realistic — no outcome features):")
    print(f"    F1      : {ew_xgb['f1']}")
    print(f"    ROC-AUC : {ew_xgb['roc_auc']}")
    print(f"    CV F1   : {ew_cv['cv_f1_mean']}")
    print(f"    CV AUC  : {ew_cv['cv_auc_mean']}")
    print(f"    Features used    : {len(ew_cols)}")
    print(f"    Features removed : {len([c for c in OUTCOME_FEATURES if c in all_feature_cols])}")

    print(f"\n  Top 5 Early-Warning Features:")
    for feat, imp in ew_imp.head(5).items():
        print(f"    {feat:<40} {round(imp, 4)}")

    print(f"\n  Selected final model:")
    print(f"    models/xgboost_early_warning_model.pkl")

    print(f"\n  Output files:")
    print(f"    models/xgboost_diagnostic_model.pkl")
    print(f"    models/xgboost_early_warning_model.pkl")
    print(f"    data/outputs/model_comparison.csv")
    print(f"    data/outputs/diagnostic_feature_importances.csv")
    print(f"    data/outputs/early_warning_feature_importances.csv")
    print(f"    data/outputs/predictions_early_warning.csv")
    print(f"    data/outputs/model_metrics.json")

    print(f"\n  Next step: Phase 8 — Risk Scoring")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()