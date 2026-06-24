# app.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Epidemic Early Warning System
# Phase 10: Streamlit Dashboard
#
# HOW TO RUN:
#   streamlit run app.py
# ═══════════════════════════════════════════════════════════════

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BioSignal — Epidemic Early Warning",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT   = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "outputs"


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_csv(filename, phase_hint=""):
    path = OUTPUT / filename
    if not path.exists():
        if phase_hint:
            st.warning(f"⚠ `{filename}` not found. Run: `{phase_hint}`")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(filename):
    path = OUTPUT / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def risk_badge(level):
    colors = {
        "Critical Risk": ("#FF0000", "#fff"),
        "High Risk":     ("#FF6B35", "#fff"),
        "Medium Risk":   ("#FFC300", "#000"),
        "Low Risk":      ("#2DC653", "#fff"),
    }
    bg, fg = colors.get(level, ("#888", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:3px 10px;border-radius:12px;'
        f'font-weight:bold;font-size:0.85em;">{level}</span>'
    )


def signal_badge(level):
    colors = {
        "Alert":   ("#FF0000", "#fff"),
        "Warning": ("#FF6B35", "#fff"),
        "Watch":   ("#FFC300", "#000"),
    }
    bg, fg = colors.get(level, ("#888", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:3px 10px;border-radius:12px;'
        f'font-weight:bold;font-size:0.85em;">{level}</span>'
    )


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<div style="background:linear-gradient(90deg,#1a1a2e,#16213e);'
    'padding:28px 32px;border-radius:12px;margin-bottom:20px;">'
    '<h1 style="color:#E63946;margin:0;font-size:2.2em;">🧬 BioSignal</h1>'
    '<p style="color:#a8b2c1;margin:6px 0 0 0;font-size:1.05em;">'
    'Epidemic Early Warning System — WHO Reports + Google Trends + XGBoost + SHAP'
    '</p></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Overview",
    "🚨 Risk Alerts",
    "⚡ Early Warnings",
    "🤖 Model Performance",
    "🔍 SHAP Explainability",
    "🔄 Counterfactual What-if",
    "ℹ️ About",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("System Overview")

    summary = load_json("risk_summary.json")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Articles Scored",     summary.get("total_scored",         0))
    c2.metric("🚨 Critical Risk",    summary.get("critical_risk",         0))
    c3.metric("🔴 High Risk",        summary.get("high_risk",             0))
    c4.metric("⚡ Early Warnings",   summary.get("early_warning_signals", 0))
    c5.metric("✅ True Unconfirmed", summary.get("true_early_warnings",   0))

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### What BioSignal Does")
        st.markdown(
            "**BioSignal** is an autonomous epidemic early warning system that detects "
            "disease outbreak signals **before** official WHO reports are published.\n\n"
            "**How it works:**\n"
            "1. 🌐 Fetches live WHO outbreak news via RSS feed\n"
            "2. 📈 Monitors Google Trends search volume for 8 diseases\n"
            "3. 🤖 Scores each outbreak with an XGBoost early-warning model\n"
            "4. 🔍 Explains every prediction using SHAP + counterfactual analysis\n"
            "5. ⚡ Flags Google Trends spikes with no WHO confirmation yet\n\n"
            "**The key insight:** When people start searching for symptoms like "
            "'dengue fever symptoms' or 'cholera outbreak', search volume spikes "
            "**8-18 days before** WHO publishes an official report. "
            "BioSignal detects those early signals."
        )

    with col2:
        st.markdown("### Model Performance")
        st.markdown(
            "| Metric | Score |\n"
            "|--------|-------|\n"
            "| F1 Score | **0.889** |\n"
            "| ROC-AUC | **0.951** |\n"
            "| CV F1 | **0.897** |\n"
            "| CV AUC | **0.943** |\n"
            "| Training rows | 1,093 |\n"
            "| Features used | 40 |"
        )
        st.caption("Early-warning model — outcome features removed to prevent leakage.")

    st.divider()
    st.markdown("### Data Sources")
    sc1, sc2, sc3 = st.columns(3)
    sc1.info("🌍 **WHO RSS Feed**\nReal outbreak news articles fetched live from who.int")
    sc2.info("📈 **Google Trends**\nWeekly search volume for 8 epidemic diseases worldwide")
    sc3.info("📚 **Georgetown DON**\n1,093 historical WHO outbreak records (1996-2015)")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — RISK ALERTS
# ═══════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Live Outbreak Risk Alerts")
    st.caption(
        "Risk scores are model-based early-warning signals, "
        "not confirmed medical diagnoses."
    )

    scores_df = load_csv("risk_scores.csv", "python src/risk_scorer.py")

    if not scores_df.empty:
        fc1, fc2, fc3 = st.columns(3)
        diseases  = ["All"] + sorted(scores_df["disease"].unique().tolist())
        countries = ["All"] + sorted(scores_df["country"].unique().tolist())
        levels    = ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"]

        sel_disease  = fc1.selectbox("Filter by Disease",    diseases)
        sel_country  = fc2.selectbox("Filter by Country",    countries)
        sel_level    = fc3.selectbox("Filter by Risk Level", levels)

        filtered = scores_df.copy()
        if sel_disease != "All":
            filtered = filtered[filtered["disease"] == sel_disease]
        if sel_country != "All":
            filtered = filtered[filtered["country"] == sel_country]
        if sel_level != "All":
            filtered = filtered[filtered["risk_level"] == sel_level]

        high_crit = filtered[
            filtered["risk_level"].isin(["Critical Risk", "High Risk"])
        ]

        if not high_crit.empty:
            st.markdown("#### 🚨 High Priority Alerts")
            for _, row in high_crit.iterrows():
                with st.expander(
                    f"{row['risk_level']} — {row['disease'].title()} "
                    f"| {row['country']} | Score: {row['risk_score']}",
                    expanded=True,
                ):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Risk Score", row["risk_score"])
                    cc2.markdown(
                        f"**Level:** {risk_badge(row['risk_level'])}",
                        unsafe_allow_html=True,
                    )
                    cc3.metric("Confidence", row.get("confidence_band", "—"))
                    st.markdown(f"**Reason:** {row.get('top_risk_reason','—')}")
                    if pd.notna(row.get("title", "")) and row.get("title", ""):
                        st.caption(f"📄 {row['title']}")

        st.divider()
        st.markdown("#### All Scored Articles")
        display_cols = [
            c for c in [
                "date", "disease", "country", "risk_score",
                "risk_level", "confidence_band", "top_risk_reason",
                "spike_ratio", "has_spike",
            ] if c in filtered.columns
        ]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("No risk scores found. Run: `python src/risk_scorer.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — EARLY WARNING SIGNALS
# ═══════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("⚡ Early Warning Signals")
    st.markdown(
        "Early warning signals are detected when **Google Trends search volume spikes** "
        "above the 90-day baseline for a disease — before WHO publishes an official report.\n\n"
        "| Signal Level | Spike Ratio | Meaning |\n"
        "|---|---|---|\n"
        "| 🚨 Alert | ≥ 3.0x | Major spike — high priority |\n"
        "| ⚠️ Warning | ≥ 2.0x | Significant elevation |\n"
        "| 👁️ Watch | ≥ 1.5x | Elevated — monitor |"
    )

    ew_df = load_csv("early_warnings.csv", "python src/risk_scorer.py")

    if not ew_df.empty:
        true_ew  = ew_df[ew_df["early_warning"] == True]
        who_conf = ew_df[ew_df["early_warning"] == False]

        ew_c1, ew_c2 = st.columns(2)
        ew_c1.metric("Total Signals Detected",       len(ew_df))
        ew_c2.metric("⚡ Unconfirmed (No WHO Report)", len(true_ew))

        if not true_ew.empty:
            st.markdown("#### ⚡ True Early Warnings — Not Yet Confirmed by WHO")
            st.error(
                "These diseases show elevated search interest "
                "with no recent WHO outbreak report. Monitor closely."
            )
            for _, row in true_ew.iterrows():
                st.markdown(
                    f"**{signal_badge(row['signal_level'])}** "
                    f"**{row['disease'].title()}** — "
                    f"Search spike: **{row['spike_ratio']}x** above baseline",
                    unsafe_allow_html=True,
                )

        if not who_conf.empty:
            st.markdown("#### ✅ WHO-Confirmed Signals")
            st.success("These spikes are already matched with a WHO outbreak report.")
            for _, row in who_conf.iterrows():
                st.markdown(
                    f"**{signal_badge(row['signal_level'])}** "
                    f"**{row['disease'].title()}** — "
                    f"Spike: **{row['spike_ratio']}x** | WHO confirmed ✓",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("#### All Early Warning Data")
        st.dataframe(ew_df, use_container_width=True)
    else:
        st.info("No early warning data found. Run: `python src/risk_scorer.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("🤖 Model Performance")

    st.markdown(
        "BioSignal trains two models to demonstrate the importance of avoiding **target leakage**:\n\n"
        "| Model | What it uses | F1 | ROC-AUC | Honest? |\n"
        "|---|---|---|---|---|\n"
        "| **Diagnostic** | All features including deaths/cases | ~1.0 | ~1.0 | ❌ Leakage |\n"
        "| **Early-Warning** | Only early-warning signals | 0.889 | 0.951 | ✅ Realistic |\n\n"
        "**Why the diagnostic model is wrong:** "
        "The label `escalated=1` was created using `deaths >= 10 OR cases >= 100`. "
        "Using `deaths` and `cases` as input features gives the model the answer — "
        "that is target leakage. The early-warning model removes these features "
        "and achieves honest, realistic performance."
    )

    comparison_df = load_csv("model_comparison.csv", "python src/model.py")
    if not comparison_df.empty:
        st.markdown("#### Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)

    st.divider()
    st.markdown("#### ✅ Final Selected Model: Early-Warning XGBoost")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 Score", "0.889")
    m2.metric("ROC-AUC",  "0.951")
    m3.metric("CV F1",    "0.897")
    m4.metric("CV AUC",   "0.943")
    st.caption(
        "5-fold stratified cross-validation. "
        "Outcome features excluded. "
        "Model: `models/xgboost_early_warning_model.pkl`"
    )

    st.divider()
    imp_df = load_csv("early_warning_feature_importances.csv")
    if not imp_df.empty:
        st.markdown("#### Top Feature Importances (Early-Warning Model)")
        st.dataframe(imp_df.head(15), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("🔍 SHAP Explainability")

    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** measures how much each feature "
        "contributed to the model's prediction for each outbreak.\n\n"
        "- **Positive SHAP value** → feature pushed risk score **higher**\n"
        "- **Negative SHAP value** → feature pushed risk score **lower**\n"
        "- **Mean |SHAP|** → average impact across all 1,093 outbreaks"
    )

    shap_summary = load_json("shap_summary.json")
    if shap_summary:
        sc1, sc2 = st.columns(2)
        sc1.metric("Rows Explained",    shap_summary.get("total_rows_explained", 0))
        sc2.metric("Features Analyzed", shap_summary.get("total_features",       0))

    shap_imp = load_csv("shap_global_importance.csv", "python src/explain.py")
    if not shap_imp.empty:
        st.markdown("#### Global Feature Importance (Mean |SHAP|)")
        display_shap = shap_imp[[
            "importance_rank", "feature",
            "mean_abs_shap", "human_readable_feature",
        ]].head(15)
        st.dataframe(display_shap, use_container_width=True)

    st.divider()

    bar_path = OUTPUT / "shap_global_bar.png"
    top_path = OUTPUT / "shap_top_features.png"

    if bar_path.exists() or top_path.exists():
        st.markdown("#### SHAP Feature Importance Charts")
        ic1, ic2 = st.columns(2)
        if bar_path.exists():
            ic1.image(str(bar_path), caption="Global SHAP — Top 15 Features")
        if top_path.exists():
            ic2.image(str(top_path), caption="Top 10 Early-Warning Features")

    st.divider()
    local_df = load_csv("shap_local_explanations.csv", "python src/explain.py")
    if not local_df.empty:
        st.markdown("#### Local Alert Explanations")
        st.caption("These explain why each specific high-risk alert was scored high.")
        for _, row in local_df.iterrows():
            with st.expander(
                f"{row['disease'].title()} — {row['country']} "
                f"| Score: {row['risk_score']} | {row['risk_level']}"
            ):
                lc1, lc2 = st.columns(2)
                lc1.markdown(f"**Risk Score:** {row['risk_score']}")
                lc1.markdown(
                    f"**Risk Level:** {risk_badge(row['risk_level'])}",
                    unsafe_allow_html=True,
                )
                lc2.markdown(f"**Confidence:** {row.get('confidence_band','—')}")
                st.markdown("**Explanation:**")
                st.info(row.get("explanation_text", "—"))

                feat_data = []
                for n in [1, 2, 3]:
                    f = row.get(f"top_positive_feature_{n}", "")
                    v = row.get(f"top_positive_value_{n}", 0)
                    if f:
                        feat_data.append({"Feature": f, "SHAP Value": v})
                if feat_data:
                    st.markdown("**Top Risk-Increasing Features:**")
                    st.table(pd.DataFrame(feat_data))

                st.markdown("**Recommended Action:**")
                st.warning(row.get("recommended_action", "—"))


# ═══════════════════════════════════════════════════════════════
# TAB 6 — COUNTERFACTUAL WHAT-IF
# ═══════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("🔄 Counterfactual What-if Explanations")

    st.markdown(
        "**Counterfactual explanations** answer the question: "
        "*What would need to change for this outbreak to be lower risk?*\n\n"
        "For each high-risk alert, the model tests small hypothetical changes "
        "to individual features and measures how much the risk score would decrease."
    )
    st.warning(
        "Counterfactuals are model-based what-if scenarios, "
        "not medical predictions or confirmed public health guidance."
    )

    cf_df = load_csv("counterfactual_explanations.csv", "python src/explain.py")

    if not cf_df.empty:
        for disease in cf_df["disease"].unique():
            disease_cf = cf_df[cf_df["disease"] == disease]
            country    = disease_cf.iloc[0]["country"]
            curr_score = disease_cf.iloc[0]["current_risk_score"]
            curr_level = disease_cf.iloc[0]["current_risk_level"]

            st.markdown(
                f"### {disease.title()} — {country} | "
                f"Current Score: {curr_score} "
                f"({risk_badge(curr_level)})",
                unsafe_allow_html=True,
            )

            for _, row in disease_cf.iterrows():
                wc1, wc2, wc3, wc4 = st.columns([3, 2, 2, 2])
                wc1.markdown(f"**{row['counterfactual_scenario']}**")
                wc2.markdown(
                    f"`{row['changed_feature']}`: "
                    f"{row['original_value']} → {row['new_value']}"
                )
                wc3.metric(
                    "New Risk Score",
                    row["new_predicted_risk_score"],
                    delta=f"-{row['risk_reduction']}",
                    delta_color="inverse",
                )
                wc4.markdown(
                    f"{risk_badge(row['new_risk_level'])}",
                    unsafe_allow_html=True,
                )
                st.caption(row.get("counterfactual_explanation", ""))
                st.divider()
    else:
        st.info("No counterfactual data found. Run: `python src/explain.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ═══════════════════════════════════════════════════════════════

with tabs[6]:
    st.subheader("About BioSignal")

    st.markdown("### Problem Statement")
    st.markdown(
        "Disease outbreaks cause the most harm in their first few weeks — "
        "before official health agencies publish reports. "
        "By the time WHO issues a formal alert, the window for early intervention "
        "has often already closed. "
        "BioSignal addresses this gap by combining publicly available signals "
        "to detect outbreak precursors **8-18 days earlier** than traditional reporting."
    )

    st.divider()
    st.markdown("### Why This Project Is Unique")
    st.markdown(
        "| What others do | What BioSignal does |\n"
        "|---|---|\n"
        "| Show confirmed case counts | Detect search spikes before confirmation |\n"
        "| React to WHO reports | Flag signals WHO has not reported yet |\n"
        "| Black-box ML | SHAP + counterfactual explanations |\n"
        "| Static dashboards | Live WHO + Trends pipeline |"
    )

    st.divider()
    st.markdown("### Data Sources")
    st.markdown(
        "| Source | Description |\n"
        "|---|---|\n"
        "| WHO RSS Feed | Live outbreak news from who.int |\n"
        "| Google Trends | Weekly search volume for 8 diseases |\n"
        "| Georgetown DON | 1,093 historical WHO outbreak records (1996-2015) |"
    )

    st.divider()
    st.markdown("### Pipeline")
    st.code(
        "1. Data Collection     -> WHO RSS + Google Trends  (src/data_collection.py)\n"
        "2. Historical Data     -> Georgetown DON database  (src/process_don.py)\n"
        "3. Data Cleaning       -> Merge + standardise      (src/cleaning.py)\n"
        "4. NLP Pipeline        -> spaCy entity extraction  (src/nlp_pipeline.py)\n"
        "5. Feature Engineering -> Spike detection          (src/features.py)\n"
        "6. Model Training      -> XGBoost early-warning    (src/model.py)\n"
        "7. Risk Scoring        -> Live outbreak scoring     (src/risk_scorer.py)\n"
        "8. Explainability      -> SHAP + counterfactuals   (src/explain.py)\n"
        "9. Dashboard           -> This Streamlit app       (app.py)",
        language="text",
    )

    st.divider()
    st.markdown("### Tech Stack")
    st.markdown(
        "`Python` · `XGBoost` · `SHAP` · `spaCy` · `Streamlit` · "
        "`Google Trends` · `WHO RSS` · `Pandas` · `Scikit-learn`"
    )

    st.divider()
    st.markdown("### Resume Line")
    st.info(
        "Built BioSignal — autonomous epidemic early-warning system detecting "
        "outbreak precursor signals from Google Trends anomaly detection + "
        "NLP-processed WHO reports. XGBoost risk scorer trained on 1,093 real "
        "outbreak records, counterfactual SHAP explainability, deployed live "
        "on Hugging Face Spaces. F1: 0.889, ROC-AUC: 0.951."
    )

    st.divider()
    st.markdown("### Disclaimer")
    st.warning(
        "BioSignal is a research and portfolio project. "
        "All risk scores are model-based early-warning signals. "
        "They are not confirmed medical diagnoses or official public health guidance. "
        "Always verify with WHO, CDC, and local health authorities."
    )

    st.caption(
        "BioSignal v1.0 — Built for public health awareness | "
        "Data: WHO, Google Trends, Georgetown University"
    )