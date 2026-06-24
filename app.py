# app.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Epidemic Early Warning System
# Phase 10.5: Live Intelligence Dashboard
# ═══════════════════════════════════════════════════════════════

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BioSignal — Epidemic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
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


def get_recommended_action(risk_level):
    actions = {
        "Critical Risk": (
            "Immediate analyst review recommended. Verify WHO updates, "
            "monitor local health ministry alerts, and track Google Trends "
            "for the next 48 hours."
        ),
        "High Risk": (
            "Monitor closely and validate with recent WHO, CDC, "
            "and local public health sources."
        ),
        "Medium Risk": (
            "Keep under watch and review if search interest "
            "or urgency language increases."
        ),
        "Low Risk": "Routine monitoring only.",
    }
    return actions.get(risk_level, "Continue routine monitoring.")


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 BioSignal")
    st.markdown("**Epidemic Intelligence System**")
    st.divider()

    demo_mode = st.checkbox("🎮 Demo Mode: Simulate outbreak spike", value=False)
    if demo_mode:
        st.warning(
            "**SIMULATION ONLY**\n\n"
            "Demo mode shows a hypothetical outbreak scenario. "
            "This is NOT real data."
        )

    st.divider()
    st.markdown("**Model**")
    st.markdown("`xgboost_early_warning_model.pkl`")
    st.markdown("F1: **0.889** | AUC: **0.951**")
    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- WHO RSS Feed")
    st.markdown("- Google Trends (18 diseases)")
    st.markdown("- Georgetown DON (1,093 records)")
    st.divider()
    st.caption(
        "BioSignal v1.0 — Research & Portfolio Project\n"
        "Not medical advice."
    )


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<div style="background:linear-gradient(90deg,#1a1a2e,#16213e);'
    'padding:24px 32px;border-radius:12px;margin-bottom:16px;">'
    '<h1 style="color:#E63946;margin:0;font-size:2em;">🧬 BioSignal</h1>'
    '<p style="color:#a8b2c1;margin:4px 0 0 0;font-size:1em;">'
    'Epidemic Early Warning Intelligence System — '
    'WHO + Google Trends + XGBoost + SHAP'
    '</p></div>',
    unsafe_allow_html=True,
)

# ── Live status bar ───────────────────────────────────────────
summary      = load_json("risk_summary.json")
scores_df    = load_csv("risk_scores.csv")
ew_df        = load_csv("early_warnings.csv")

total_scored  = summary.get("total_scored", len(scores_df))
critical      = summary.get("critical_risk", 0)
high          = summary.get("high_risk", 0)
ew_signals    = summary.get("early_warning_signals", len(ew_df))
true_ew       = summary.get("true_early_warnings", 0)
scored_at     = summary.get("scored_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

if true_ew > 0:
    system_status = "🚨 ALERT — Unconfirmed trend spike detected"
    status_color  = "#FF0000"
elif ew_signals > 0:
    system_status = "👁️ WATCH — Trend spike detected, WHO confirmed"
    status_color  = "#FFC300"
else:
    system_status = "✅ Monitoring — No unconfirmed critical spike"
    status_color  = "#2DC653"

st.markdown(
    f'<div style="background:#0d1117;border:1px solid #30363d;'
    f'border-radius:8px;padding:12px 20px;margin-bottom:16px;'
    f'display:flex;align-items:center;gap:16px;">'
    f'<span style="color:{status_color};font-weight:bold;">'
    f'{system_status}</span>'
    f'<span style="color:#666;font-size:0.85em;margin-left:auto;">'
    f'Last updated: {scored_at}</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Overview",
    "🚨 Risk Alerts",
    "⚡ Early Warning Monitor",
    "🧪 Leakage Demo",
    "🔍 SHAP Explainability",
    "🔄 Counterfactual What-if",
    "📅 Timeline",
    "ℹ️ About",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════

with tabs[0]:

    # Demo mode alert
    if demo_mode:
        st.error(
            "🎮 **DEMO MODE ACTIVE** — Simulated outbreak scenario shown below. "
            "This is NOT real data."
        )
        with st.expander("🎮 Simulated Alert — Dengue / India (DEMO ONLY)", expanded=True):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Simulated Risk Score", "87.4")
            dc2.markdown("**Level:** " + risk_badge("Critical Risk"), unsafe_allow_html=True)
            dc3.metric("Google Trends Spike", "4.2x above baseline")
            st.markdown("**WHO Report:** ❌ Not yet published")
            st.markdown("**Signal:** ⚡ True Early Warning — Unconfirmed")
            st.info(
                "**Demo explanation:** This simulates how BioSignal behaves when "
                "Google Trends shows a 4.2x spike for dengue in India but WHO has "
                "not published an outbreak report yet. This would be flagged as a "
                "genuine early warning for analyst review. "
                "*This is a model-based demo scenario, not a real outbreak.*"
            )
        st.divider()

    # Metric cards
    st.subheader("System Status")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Articles Scored",    total_scored)
    c2.metric("🚨 Critical",        critical)
    c3.metric("🔴 High Risk",       high)
    c4.metric("⚡ EW Signals",      ew_signals)
    c5.metric("✅ Unconfirmed",     true_ew)
    c6.metric("Diseases Monitored", 18)

    st.divider()

    # Why BioSignal is different
    st.markdown("### Why BioSignal Is Different From Normal Dashboards")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #E63946;'
            'border-radius:8px;padding:16px;">'
            '<h4 style="color:#E63946;margin:0 0 8px 0;">❌ What normal dashboards do</h4>'
            '<ul style="color:#a8b2c1;">'
            '<li>Show confirmed case counts after outbreak</li>'
            '<li>React to WHO reports already published</li>'
            '<li>Black-box ML with no explanation</li>'
            '<li>Static historical data</li>'
            '<li>No early detection signal</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #2DC653;'
            'border-radius:8px;padding:16px;">'
            '<h4 style="color:#2DC653;margin:0 0 8px 0;">✅ What BioSignal does</h4>'
            '<ul style="color:#a8b2c1;">'
            '<li>Detects search spikes before WHO confirms</li>'
            '<li>Flags signals WHO has not reported yet</li>'
            '<li>SHAP + counterfactual explanations</li>'
            '<li>Live WHO + Google Trends pipeline</li>'
            '<li>8-18 day lead time detection goal</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # How it works
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### How BioSignal Works")
        st.markdown(
            "**Step 1 — Data Collection**\n"
            "BioSignal fetches live WHO outbreak articles and monitors "
            "Google Trends search volume for 18 epidemic diseases weekly.\n\n"
            "**Step 2 — Early Signal Detection**\n"
            "When search volume spikes above 1.5x baseline for a disease, "
            "BioSignal flags it as a potential early warning — even before "
            "WHO publishes anything.\n\n"
            "**Step 3 — ML Risk Scoring**\n"
            "An XGBoost early-warning model scores each outbreak on 0-100. "
            "Critically, it uses **no outcome features** — no deaths, no case counts, "
            "no severity. Only early signals: NLP urgency, disease risk profiles, "
            "seasonal patterns, and Trends data.\n\n"
            "**Step 4 — Explainability**\n"
            "Every prediction is explained using SHAP values and counterfactual "
            "analysis — showing exactly which signals drove the score and what "
            "would reduce it."
        )

    with col2:
        st.markdown("### Model")
        st.markdown(
            "| Metric | Score |\n"
            "|--------|-------|\n"
            "| F1 Score | **0.889** |\n"
            "| ROC-AUC | **0.951** |\n"
            "| CV F1 | **0.897** |\n"
            "| CV AUC | **0.943** |\n"
            "| Training rows | 1,093 |\n"
            "| Features used | 40 |\n"
            "| Outcome features | **0** |"
        )
        st.caption("No deaths, cases, or severity used as inputs.")

    st.divider()
    st.markdown("### Data Sources")
    sc1, sc2, sc3 = st.columns(3)
    sc1.info("🌍 **WHO RSS Feed**\nLive outbreak articles from who.int — fetched daily")
    sc2.info("📈 **Google Trends**\nWeekly search volume for 18 epidemic diseases worldwide")
    sc3.info("📚 **Georgetown DON**\n1,093 real WHO outbreak records used for training (1996–2015)")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — RISK ALERTS
# ═══════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Live Outbreak Risk Alerts")
    st.caption(
        "Risk scores are model-based early-warning signals. "
        "Requires analyst verification. Not a confirmed diagnosis."
    )

    if not scores_df.empty:
        fc1, fc2, fc3 = st.columns(3)
        diseases  = ["All"] + sorted(scores_df["disease"].unique().tolist())
        countries = ["All"] + sorted(scores_df["country"].unique().tolist())
        levels    = ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"]

        sel_disease = fc1.selectbox("Filter by Disease",    diseases,  key="ra_d")
        sel_country = fc2.selectbox("Filter by Country",    countries, key="ra_c")
        sel_level   = fc3.selectbox("Filter by Risk Level", levels,    key="ra_l")

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
                    st.info(f"**Analyst Action:** {get_recommended_action(row['risk_level'])}")
                    if pd.notna(row.get("title","")) and row.get("title",""):
                        st.caption(f"📄 {row['title']}")

        st.divider()
        st.markdown("#### All Scored Articles")
        display_cols = [
            c for c in [
                "date","disease","country","risk_score","risk_level",
                "confidence_band","top_risk_reason","spike_ratio","has_spike",
            ] if c in filtered.columns
        ]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("No risk scores found. Run: `python src/risk_scorer.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — EARLY WARNING MONITOR
# ═══════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("⚡ Early Warning Monitor")

    # Logic explanation card
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #FFC300;'
        'border-radius:8px;padding:16px;margin-bottom:16px;">'
        '<h4 style="color:#FFC300;margin:0 0 8px 0;">'
        '⚡ What is a True Early Warning?</h4>'
        '<p style="color:#a8b2c1;margin:0;">'
        'A <strong>True Early Warning</strong> occurs when:<br>'
        '&nbsp;&nbsp;✅ Google Trends spike detected (≥1.5x above baseline)<br>'
        '&nbsp;&nbsp;✅ No recent WHO outbreak report for this disease<br>'
        '&nbsp;&nbsp;✅ ML risk score is High or Critical<br><br>'
        'This means the public is searching for a disease <strong>before</strong> '
        'WHO has officially confirmed it. These signals historically appear '
        '<strong>8-18 days before</strong> official reports.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "| Signal Level | Spike Ratio | Meaning |\n"
        "|---|---|---|\n"
        "| 🚨 Alert | ≥ 3.0x | Major spike — high priority |\n"
        "| ⚠️ Warning | ≥ 2.0x | Significant elevation |\n"
        "| 👁️ Watch | ≥ 1.5x | Elevated — monitor |"
    )

    if not ew_df.empty:
        true_ew_df  = ew_df[ew_df["early_warning"] == True]
        who_conf_df = ew_df[ew_df["early_warning"] == False]

        ew_c1, ew_c2, ew_c3 = st.columns(3)
        ew_c1.metric("Total Signals",          len(ew_df))
        ew_c2.metric("⚡ True Early Warnings", len(true_ew_df),
                     help="Spike detected but no WHO report yet")
        ew_c3.metric("✅ WHO Confirmed",        len(who_conf_df))

        if not true_ew_df.empty:
            st.markdown("#### ⚡ Unconfirmed Signals — Requires Analyst Review")
            st.error(
                "These diseases show elevated search interest with "
                "no recent WHO outbreak report. Monitor closely."
            )
            for _, row in true_ew_df.iterrows():
                st.markdown(
                    f"**{signal_badge(row['signal_level'])}** "
                    f"**{row['disease'].title()}** — "
                    f"Search spike: **{row['spike_ratio']}x** above baseline | "
                    f"No WHO report ⚠️",
                    unsafe_allow_html=True,
                )
        else:
            st.success(
                "No unconfirmed spikes detected today. "
                "All signals are WHO-confirmed or below threshold."
            )

        if not who_conf_df.empty:
            st.markdown("#### ✅ WHO-Confirmed Signals")
            st.success("These spikes are already matched with a WHO outbreak report.")
            for _, row in who_conf_df.iterrows():
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
        st.info("No early warning data. Run: `python src/risk_scorer.py`")

    # Demo mode simulation
    if demo_mode:
        st.divider()
        st.error("🎮 **DEMO SIMULATION — NOT REAL DATA**")
        st.markdown("#### Simulated Early Warning: Dengue / India")
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        sim_c1.metric("Google Trends Spike", "4.2x")
        sim_c2.metric("WHO Report",          "❌ None")
        sim_c3.metric("Signal Level",        "🚨 Alert")
        st.markdown(
            "**Simulated risk score:** 87.4 (Critical Risk)\n\n"
            "**What BioSignal would do:** Flag this as a True Early Warning, "
            "generate a SHAP explanation, and recommend immediate analyst review.\n\n"
            "*This is a demo scenario showing how BioSignal behaves during an "
            "active unconfirmed outbreak. Not a real prediction.*"
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — LEAKAGE DEMO
# ═══════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("🧪 Leakage Demonstration")

    st.markdown(
        "This tab demonstrates one of the most important concepts in ML for "
        "epidemic prediction: **target leakage**."
    )

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #E63946;'
        'border-radius:8px;padding:16px;margin-bottom:16px;">'
        '<h4 style="color:#E63946;margin:0 0 8px 0;">What is target leakage?</h4>'
        '<p style="color:#a8b2c1;margin:0;">'
        'The BioSignal label is: <code>escalated = 1 if deaths ≥ 10 OR cases ≥ 100</code><br><br>'
        'If we train a model using <strong>deaths</strong> and <strong>cases</strong> '
        'as input features, the model is essentially given the answer. '
        'It achieves perfect accuracy — but it cannot make predictions '
        '<em>before</em> deaths and cases are known. '
        'This is called <strong>target leakage</strong> and makes the model '
        'useless for real early warning.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Model Comparison")
    comparison_data = {
        "Model":        ["Diagnostic Model",    "Early-Warning Model"],
        "Features Used":["Includes deaths, cases, severity (leakage)",
                         "Removes deaths, cases, severity"],
        "F1":           [1.000,                  0.889],
        "ROC-AUC":      [1.000,                  0.951],
        "Usable?":      ["❌ Leakage — unrealistic",
                         "✅ Honest — deployable"],
        "Deployed?":    ["❌ No",                "✅ Yes"],
    }

    # Try to load from file if available
    comp_df = load_csv("model_comparison.csv")
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True)
        st.divider()

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    st.divider()
    st.success(
        "**BioSignal uses only the Early-Warning Model.** "
        "The diagnostic model is shown purely to demonstrate why leakage "
        "produces unrealistic results. A model with F1=1.0 on epidemic data "
        "should always be questioned."
    )

    st.markdown("#### Outcome Features Removed from Early-Warning Model")
    removed = [
        "severity_score", "cases_total", "deaths", "log_cases",
        "log_deaths", "case_fatality_ratio", "has_deaths",
        "has_cases", "cases_per_death", "outbreak_relevance_score",
    ]
    col1, col2 = st.columns(2)
    for i, feat in enumerate(removed):
        if i % 2 == 0:
            col1.markdown(f"- `{feat}`")
        else:
            col2.markdown(f"- `{feat}`")

    st.caption(
        "These 10 features were removed because they are derived directly "
        "from the label formula. Using them would mean the model already "
        "knows the answer at prediction time."
    )

    st.divider()
    imp_df = load_csv("early_warning_feature_importances.csv")
    if not imp_df.empty:
        st.markdown("#### Early-Warning Model — Top Feature Importances")
        st.dataframe(imp_df.head(15), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("🔍 SHAP Explainability")

    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** measures how much each feature "
        "contributed to each prediction. Unlike feature importance, SHAP shows "
        "the direction of impact for each individual prediction.\n\n"
        "- **Positive SHAP** → feature pushed risk score **higher**\n"
        "- **Negative SHAP** → feature pushed risk score **lower**\n"
        "- **Mean |SHAP|** → average impact magnitude across all 1,093 outbreaks"
    )

    shap_summary = load_json("shap_summary.json")
    if shap_summary:
        sc1, sc2 = st.columns(2)
        sc1.metric("Rows Explained",    shap_summary.get("total_rows_explained", 0))
        sc2.metric("Features Analyzed", shap_summary.get("total_features", 0))

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
                st.markdown("**Plain English Explanation:**")
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

                st.markdown("**Analyst Action:**")
                st.warning(get_recommended_action(row["risk_level"]))


# ═══════════════════════════════════════════════════════════════
# TAB 6 — COUNTERFACTUAL WHAT-IF
# ═══════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("🔄 Counterfactual What-if Explanations")

    st.markdown(
        "**Counterfactual analysis** answers the question most dashboards never ask:\n\n"
        "> *What would need to change for this outbreak to be classified as lower risk?*\n\n"
        "For each high-risk alert, BioSignal tests hypothetical changes to individual "
        "features one at a time, measures the new predicted risk score, and keeps only "
        "changes that actually reduce risk. This makes the model transparent and useful "
        "for analyst decision-making."
    )
    st.warning(
        "Counterfactuals are model-based what-if scenarios. "
        "They are not medical predictions or confirmed public health guidance."
    )

    cf_df = load_csv("counterfactual_explanations.csv", "python src/explain.py")

    if not cf_df.empty:
        for disease in cf_df["disease"].unique():
            disease_cf = cf_df[cf_df["disease"] == disease]
            country    = disease_cf.iloc[0]["country"]
            curr_score = disease_cf.iloc[0]["current_risk_score"]
            curr_level = disease_cf.iloc[0]["current_risk_level"]

            st.markdown(
                f"### {disease.title()} — {country}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Current:** {risk_badge(curr_level)} Score: {curr_score}",
                unsafe_allow_html=True,
            )
            st.markdown("**What would reduce the risk?**")

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
                    risk_badge(row["new_risk_level"]),
                    unsafe_allow_html=True,
                )
                st.caption(row.get("counterfactual_explanation", ""))
                st.divider()
    else:
        st.info("No counterfactual data. Run: `python src/explain.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 7 — TIMELINE
# ═══════════════════════════════════════════════════════════════

with tabs[6]:
    st.subheader("📅 Outbreak Surveillance Timeline")
    st.markdown(
        "This view shows all scored WHO articles as a surveillance timeline — "
        "making BioSignal look and feel like a monitoring system rather than "
        "a one-time analysis."
    )

    if not scores_df.empty:
        timeline_cols = [
            c for c in [
                "date","disease","country","risk_score","risk_level",
                "has_spike","spike_ratio","top_risk_reason","confidence_band",
            ] if c in scores_df.columns
        ]
        timeline = scores_df[timeline_cols].sort_values(
            "risk_score", ascending=False
        ).reset_index(drop=True)

        # Color code by risk level
        def highlight_risk(row):
            colors = {
                "Critical Risk": "background-color:#FF000022",
                "High Risk":     "background-color:#FF6B3522",
                "Medium Risk":   "background-color:#FFC30022",
                "Low Risk":      "background-color:#2DC65322",
            }
            return [colors.get(row.get("risk_level",""), "")] * len(row)

        st.dataframe(
            timeline.style.apply(highlight_risk, axis=1),
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### Risk Distribution")
        risk_counts = scores_df["risk_level"].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Critical Risk", risk_counts.get("Critical Risk", 0))
        col2.metric("High Risk",     risk_counts.get("High Risk", 0))
        col3.metric("Medium Risk",   risk_counts.get("Medium Risk", 0))
        col4.metric("Low Risk",      risk_counts.get("Low Risk", 0))
    else:
        st.info("No timeline data. Run: `python src/risk_scorer.py`")


# ═══════════════════════════════════════════════════════════════
# TAB 8 — ABOUT
# ═══════════════════════════════════════════════════════════════

with tabs[7]:
    st.subheader("About BioSignal")

    st.markdown("### What BioSignal Is")
    st.markdown(
        "BioSignal is an explainable epidemic early-warning system. "
        "It monitors WHO outbreak reports and Google Trends disease search patterns, "
        "then uses an early-warning XGBoost model to assign a risk score "
        "before relying on direct outcome signals like deaths or case counts.\n\n"
        "The project intentionally compares a diagnostic leakage model with an "
        "honest early-warning model. The final model removes deaths, cases, and "
        "severity from inputs and achieves realistic performance.\n\n"
        "Each alert includes SHAP explanations and counterfactual what-if analysis, "
        "making it useful for analyst review rather than just prediction."
    )

    st.divider()
    st.markdown("### Why This Project Is Unique")
    st.markdown(
        "| What others do | What BioSignal does |\n"
        "|---|---|\n"
        "| Show confirmed case counts | Detect search spikes before confirmation |\n"
        "| React to WHO reports | Flag signals WHO has not reported yet |\n"
        "| Black-box ML | SHAP + counterfactual explanations |\n"
        "| Static dashboards | Live WHO + Trends pipeline |\n"
        "| One model | Two models — leakage demo + honest model |\n"
        "| Accuracy only | Explainability for analyst action |"
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
        "Built BioSignal, an explainable epidemic early-warning ML system using "
        "WHO reports, Google Trends, NLP, XGBoost, SHAP, and counterfactual "
        "explanations, deployed as a live Streamlit dashboard on Hugging Face Spaces. "
        "F1: 0.889, ROC-AUC: 0.951."
    )

    st.divider()
    st.markdown("### Data Sources")
    st.markdown(
        "| Source | Description |\n"
        "|---|---|\n"
        "| WHO RSS Feed | Live outbreak news from who.int |\n"
        "| Google Trends | Weekly search volume for 18 diseases |\n"
        "| Georgetown DON | 1,093 historical WHO outbreak records (1996–2015) |"
    )

    st.divider()
    st.warning(
        "**Disclaimer:** BioSignal is a research and portfolio project. "
        "All risk scores are model-based early-warning signals. "
        "They are not confirmed medical diagnoses or official public health guidance. "
        "Always verify with WHO, CDC, and local health authorities."
    )
    st.caption(
        "BioSignal v1.0 — Built for public health awareness | "
        "Data: WHO, Google Trends, Georgetown University"
    )