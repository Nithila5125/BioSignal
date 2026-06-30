"""
BioSignal — Epidemic Early Warning Intelligence Platform
Original ML logic untouched. Dashboard display QA pass v2:
- Risk Alert cards show new tier-aware top_risk_reason
- spike_ratio_used surfaced with clear "Recent 4-record spike ratio" label
- No st.expander() anywhere in Risk Alerts or SHAP Local Explanations
  (fixes .arrow_right / double_arrow_right icon rendering bug)
- Freshness panel HTML built via string concatenation (fixes raw tags showing)
- Early Warning cards carry safe analyst-verification wording
- About section: 5 sections only, no resume line
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="BioSignal — Epidemic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT   = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "outputs"

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def load_csv(filename, phase_hint=""):
    path = OUTPUT / filename
    if not path.exists():
        if phase_hint:
            st.markdown(
                f'<div class="bs-missing">'
                f'<div class="bs-missing-title">Data file missing: {filename}</div>'
                f'<div class="bs-missing-hint">Run: {phase_hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
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
        "Critical Risk": ("#EF4444", "#fff"),
        "High Risk":     ("#F97316", "#fff"),
        "Medium Risk":   ("#FACC15", "#000"),
        "Low Risk":      ("#22C55E", "#fff"),
    }
    bg, fg = colors.get(level, ("#94A3B8", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:3px 12px;border-radius:999px;'
        f'font-weight:700;font-size:0.78em;'
        f'text-transform:uppercase;letter-spacing:0.4px;">'
        f'{level}</span>'
    )


def signal_badge(level):
    colors = {
        "Alert":   ("#EF4444", "#fff"),
        "Warning": ("#F97316", "#fff"),
        "Watch":   ("#FACC15", "#000"),
    }
    bg, fg = colors.get(level, ("#94A3B8", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:3px 12px;border-radius:999px;'
        f'font-weight:700;font-size:0.78em;'
        f'text-transform:uppercase;letter-spacing:0.4px;">'
        f'{level}</span>'
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


def alert_card_html(row):
    """
    Render a high-priority alert as a pure HTML card — no st.expander().
    Surfaces the new tier-aware top_risk_reason and the unified
    spike_ratio_used (labeled "Recent 4-record spike ratio").
    """
    level      = str(row.get("risk_level", "Low Risk"))
    disease    = str(row.get("disease", "Unknown")).title()
    country    = str(row.get("country", "Unknown"))
    score      = row.get("risk_score", "—")
    confidence = str(row.get("confidence_band", "Model-based"))
    reason     = str(row.get("top_risk_reason", "Moderate early-warning signal from available features"))
    title      = str(row.get("title", ""))
    action     = get_recommended_action(level)

    spike_used = row.get("spike_ratio_used", row.get("spike_ratio", None))
    spike_display = ""
    if spike_used is not None and str(spike_used) not in ("nan", ""):
        try:
            spike_val = float(spike_used)
            spike_display = f"{spike_val:.2f}×"
        except (ValueError, TypeError):
            spike_display = str(spike_used)

    border_colors = {
        "Critical Risk": "#EF4444",
        "High Risk":     "#F97316",
        "Medium Risk":   "#FACC15",
        "Low Risk":      "#22C55E",
    }
    glow_colors = {
        "Critical Risk": "rgba(239,68,68,0.25)",
        "High Risk":     "rgba(249,115,22,0.15)",
        "Medium Risk":   "rgba(250,204,21,0.10)",
        "Low Risk":      "rgba(34,197,94,0.10)",
    }
    badge_bg = {
        "Critical Risk": "#EF4444",
        "High Risk":     "#F97316",
        "Medium Risk":   "#FACC15",
        "Low Risk":      "#22C55E",
    }
    badge_fg = {
        "Critical Risk": "#fff",
        "High Risk":     "#fff",
        "Medium Risk":   "#000",
        "Low Risk":      "#fff",
    }

    bc  = border_colors.get(level, "#94A3B8")
    gc  = glow_colors.get(level,   "rgba(148,163,184,0.10)")
    bbg = badge_bg.get(level,      "#94A3B8")
    bfg = badge_fg.get(level,      "#fff")

    title_html = (
        f'<div class="bs-alert-source">Source: {title}</div>'
        if title and title != "nan" else ""
    )

    spike_html = (
        f'<div class="bs-alert-row">'
        f'<span class="bs-alert-label">Spike Ratio</span>'
        f'<span class="bs-alert-value">{spike_display} '
        f'<span class="bs-spike-tag">Recent 4-record spike ratio</span></span>'
        f'</div>'
        if spike_display else ""
    )

    return f"""
    <div class="bs-alert-card" style="border-left-color:{bc};box-shadow:0 4px 24px {gc};">
        <div class="bs-alert-header">
            <div class="bs-alert-title">
                <span class="bs-alert-level-badge"
                      style="background:{bbg};color:{bfg};">{level}</span>
                <span class="bs-alert-name">{disease}</span>
                <span class="bs-alert-sep">·</span>
                <span class="bs-alert-country">{country}</span>
                <span class="bs-alert-sep">·</span>
                <span class="bs-alert-score">Score: {score}</span>
            </div>
        </div>
        <div class="bs-alert-body">
            <div class="bs-alert-row">
                <span class="bs-alert-label">Confidence</span>
                <span class="bs-alert-value">{confidence}</span>
            </div>
            {spike_html}
            <div class="bs-alert-row">
                <span class="bs-alert-label">Reason</span>
                <span class="bs-alert-value">{reason}</span>
            </div>
            <div class="bs-alert-action">
                Analyst action: {action}
            </div>
            {title_html}
        </div>
    </div>
    """


def file_freshness(filename):
    path = OUTPUT / filename
    if not path.exists():
        return {
            "file": filename,
            "status": "Missing",
            "last_modified": "—",
            "age": "—",
            "age_hours": None,
        }

    modified  = datetime.fromtimestamp(path.stat().st_mtime)
    age_delta = datetime.now() - modified
    age_hours = age_delta.total_seconds() / 3600

    status = "Fresh" if age_hours <= 24 else "Stale"

    if age_hours < 1:
        age_text = f"{int(age_delta.total_seconds() // 60)} min ago"
    elif age_hours < 24:
        age_text = f"{age_hours:.1f} hours ago"
    else:
        age_text = f"{age_hours / 24:.1f} days ago"

    return {
        "file": filename,
        "status": status,
        "last_modified": modified.strftime("%Y-%m-%d %H:%M"),
        "age": age_text,
        "age_hours": age_hours,
    }


def freshness_badge(status):
    colors = {
        "Fresh":   ("#22C55E", "#fff"),
        "Stale":   ("#FACC15", "#000"),
        "Missing": ("#EF4444", "#fff"),
    }
    bg, fg = colors.get(status, ("#94A3B8", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:3px 12px;border-radius:999px;'
        f'font-weight:700;font-size:0.74em;'
        f'text-transform:uppercase;letter-spacing:0.4px;">'
        f'{status}</span>'
    )


# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg:         #020617;
        --navy:       #071A2C;
        --blue:       #0B6EFD;
        --cyan:       #00D4FF;
        --teal:       #14B8A6;
        --ink:        #F8FAFC;
        --muted:      #94A3B8;
        --card:       rgba(15, 23, 42, 0.84);
        --card-hover: rgba(30, 41, 59, 0.92);
        --glass:      rgba(255,255,255,0.10);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, rgba(11,110,253,0.16), transparent 60%),
            radial-gradient(900px 500px at -10% 20%,  rgba(20,184,166,0.10), transparent 55%),
            linear-gradient(180deg, #020617 0%, #071A2C 60%, #020617 100%);
        color: var(--ink);
    }
    .stApp::after {
        content: "";
        position: fixed; inset: -50%;
        background-image:
            linear-gradient(rgba(0,212,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.035) 1px, transparent 1px);
        background-size: 48px 48px;
        animation: bsGridDrift 30s linear infinite;
        pointer-events: none; z-index: 0;
    }
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background: linear-gradient(
            180deg, transparent 0%,
            rgba(0,212,255,0.04) 50%, transparent 100%
        );
        background-size: 100% 260px;
        animation: bsScan 10s linear infinite;
        pointer-events: none; z-index: 0;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1360px;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4, h5, p, span, label, div, li {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif !important;
        color: var(--ink);
    }

    /* ── HERO ── */
    .bs-hero {
        border-radius: 18px; padding: 28px 36px;
        background: var(--card); border: 1px solid var(--glass);
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 55px rgba(0,0,0,0.55);
        animation: bsHeroIn 0.9s cubic-bezier(.2,.8,.2,1) both;
        position: relative; overflow: hidden; margin-bottom: 16px;
    }
    .bs-hero::before {
        content: ""; position: absolute; top: -60%; left: -10%;
        width: 120%; height: 220%;
        background: conic-gradient(
            from 0deg, transparent, rgba(0,212,255,0.08),
            transparent 30%, rgba(20,184,166,0.08), transparent 60%
        );
        animation: bsSpin 20s linear infinite; pointer-events: none;
    }
    .bs-hero > * { position: relative; z-index: 1; }
    .bs-hero h1 {
        font-size: 2.8rem !important; font-weight: 800 !important;
        margin: 0 !important; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #fff, var(--cyan), var(--teal), #fff);
        background-size: 300% 100%;
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: bsGrad 7s linear infinite;
    }
    .bs-hero .bs-sub { font-size: 1.1rem; font-weight: 600; color: var(--cyan); margin-top: 4px; }
    .bs-hero .bs-desc { font-size: 0.9rem; color: var(--muted); margin-top: 8px; letter-spacing: 0.3px; }

    /* ── STATUS BAR ── */
    .bs-status-bar {
        background: var(--card); border: 1px solid var(--glass);
        border-radius: 10px; padding: 11px 20px; margin-bottom: 16px;
        display: flex; align-items: center; gap: 16px;
        backdrop-filter: blur(12px); animation: bsFade 0.6s ease both;
    }

    /* ── LIVE BADGES ── */
    .bs-badges { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .bs-sbadge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 7px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: rgba(255,255,255,0.05); border: 1px solid var(--glass); color: var(--ink);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .bs-sbadge:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.45); }
    .bs-sbadge.live  { border-color: rgba(34,197,94,0.5);  color: #86EFAC; }
    .bs-sbadge.warnb { border-color: rgba(250,204,21,0.5); color: #FDE68A; }
    .bs-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #22C55E; animation: bsPulse 1.6s infinite;
    }

    /* ── SECTION HEADINGS ── */
    .bs-h {
        font-size: 1.2rem; font-weight: 700; color: var(--ink) !important;
        margin: 8px 0 14px 0; padding-left: 12px;
        border-left: 3px solid var(--cyan); animation: bsFade 0.5s ease both;
    }

    /* ── GLASS PANEL ── */
    .bs-panel {
        background: var(--card); border: 1px solid var(--glass);
        border-radius: 14px; padding: 18px 22px; backdrop-filter: blur(12px);
        margin-bottom: 14px; animation: bsRise 0.6s cubic-bezier(.2,.8,.2,1) both;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .bs-panel:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.3); }
    .bs-panel.accent { border-left: 4px solid var(--cyan); }
    .bs-panel.green  { border-left: 4px solid #22C55E; }
    .bs-panel.red    { border-left: 4px solid #EF4444; }
    .bs-panel.amber  { border-left: 4px solid #FACC15; }
    .bs-panel h4 { margin: 0 0 8px 0; font-size: 1rem; font-weight: 700; }
    .bs-panel p, .bs-panel li { font-size: 0.88rem; color: #CBD5E1; line-height: 1.65; margin: 0; }

    /* ── CUSTOM ALERT CARDS (no st.expander — fixes .arrow_right bug) ── */
    .bs-alert-card {
        background: var(--card);
        border: 1px solid var(--glass);
        border-left: 5px solid #94A3B8;
        border-radius: 14px;
        padding: 0;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        animation: bsRise 0.5s cubic-bezier(.2,.8,.2,1) both;
    }
    .bs-alert-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,212,255,0.2);
        border-color: rgba(0,212,255,0.3);
    }
    .bs-alert-header {
        padding: 14px 20px 12px 20px;
        border-bottom: 1px solid var(--glass);
        background: rgba(255,255,255,0.03);
    }
    .bs-alert-title {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .bs-alert-level-badge {
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.72em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: nowrap;
    }
    .bs-alert-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
    }
    .bs-alert-sep {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .bs-alert-country {
        font-size: 0.88rem;
        color: #CBD5E1;
        font-weight: 500;
    }
    .bs-alert-score {
        font-size: 0.88rem;
        color: var(--cyan);
        font-weight: 700;
        margin-left: auto;
    }
    .bs-alert-body {
        padding: 14px 20px 16px 20px;
    }
    .bs-alert-row {
        display: flex;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 0.87rem;
        line-height: 1.5;
    }
    .bs-alert-label {
        color: var(--muted);
        font-weight: 600;
        min-width: 90px;
        flex-shrink: 0;
    }
    .bs-alert-value {
        color: #E2E8F0;
    }
    .bs-spike-tag {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 500;
        margin-left: 4px;
    }
    .bs-alert-action {
        margin-top: 10px;
        padding: 10px 14px;
        background: rgba(0,212,255,0.07);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 8px;
        font-size: 0.84rem;
        color: #7DD3FC;
        font-weight: 500;
    }
    .bs-alert-source {
        margin-top: 8px;
        font-size: 0.78rem;
        color: var(--muted);
        font-style: italic;
    }

    /* ── FRESHNESS PANEL ── */
    .bs-fresh-panel {
        background: var(--card);
        border: 1px solid var(--glass);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
        animation: bsRise 0.6s cubic-bezier(.2,.8,.2,1) both;
    }
    .bs-fresh-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 14px;
    }
    .bs-fresh-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
    }
    .bs-fresh-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .bs-fresh-table th {
        text-align: left;
        color: var(--muted);
        font-weight: 600;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        padding: 8px 12px;
        border-bottom: 1px solid var(--glass);
    }
    .bs-fresh-table td {
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #CBD5E1;
    }
    .bs-fresh-table tr:last-child td {
        border-bottom: none;
    }
    .bs-fresh-table tr:hover td {
        background: rgba(255,255,255,0.02);
    }
    .bs-fresh-filename {
        font-family: monospace;
        color: var(--ink);
        font-size: 0.82rem;
    }
    .bs-refresh-box {
        background: rgba(0,212,255,0.06);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 14px;
    }
    .bs-refresh-box .bs-refresh-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #7DD3FC;
        margin-bottom: 6px;
    }
    .bs-refresh-box code {
        display: block;
        background: rgba(0,0,0,0.3);
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #BAE6FD;
        line-height: 1.8;
    }
    .bs-refresh-box p {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 8px;
        line-height: 1.5;
    }
    .bs-stale-warning {
        margin-top: 10px;
        padding: 10px 14px;
        background: rgba(250,204,21,0.08);
        border: 1px solid rgba(250,204,21,0.3);
        border-radius: 8px;
        font-size: 0.82rem;
        color: #FDE68A;
        font-weight: 600;
    }

    /* ── MISSING FILE ── */
    .bs-missing {
        background: rgba(249,115,22,0.08);
        border: 1px solid rgba(249,115,22,0.35);
        border-left: 4px solid #F97316;
        border-radius: 12px; padding: 14px 18px; margin: 8px 0;
        animation: bsFade 0.5s ease both;
    }
    .bs-missing-title { font-weight: 700; color: #FDBA74; font-size: 0.92rem; }
    .bs-missing-hint  { font-size: 0.82rem; color: var(--muted); margin-top: 5px; font-family: monospace; }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #071A2C) !important;
        border-right: 1px solid var(--glass);
    }
    .bs-side-h {
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
        color: var(--cyan); margin: 16px 0 6px 0;
    }
    .bs-side-row {
        font-size: 0.84rem; color: #CBD5E1; padding: 3px 0;
        display: flex; justify-content: space-between;
    }
    .bs-side-row b { color: var(--ink); }

    /* ── HIDE SIDEBAR COLLAPSE BUTTON (safe — data-testid only) ── */
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"]   { display: none !important; }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--card); padding: 6px;
        border-radius: 12px; border: 1px solid var(--glass);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 14px; font-weight: 600;
        font-size: 0.84rem; color: var(--muted);
        transition: color 0.15s ease, background 0.15s ease, transform 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ink); background: rgba(255,255,255,0.04); transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--blue), var(--teal)) !important;
        color: #fff !important; animation: bsTabGlow 2.4s ease-in-out infinite;
    }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px; overflow: hidden; border: 1px solid var(--glass);
    }

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--glass);
        border-radius: 12px; padding: 14px 16px; backdrop-filter: blur(10px);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px); border-color: rgba(0,212,255,0.35);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important; font-size: 0.78rem !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.4px;
    }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; }

    /* ── ALERTS ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important; border-left-width: 4px !important;
        backdrop-filter: blur(8px);
    }

    /* ── KEYFRAMES ── */
    @keyframes bsFade    { from{opacity:0;transform:translateY(10px);}   to{opacity:1;transform:translateY(0);} }
    @keyframes bsRise    { from{opacity:0;transform:translateY(18px) scale(0.98);} to{opacity:1;transform:translateY(0) scale(1);} }
    @keyframes bsHeroIn  { from{opacity:0;transform:translateY(24px) scale(0.99);} to{opacity:1;transform:translateY(0) scale(1);} }
    @keyframes bsPulse   { 0%{box-shadow:0 0 0 0 rgba(34,197,94,0.7);} 70%{box-shadow:0 0 0 9px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }
    @keyframes bsGlow    { 0%,100%{box-shadow:0 2px 10px rgba(239,68,68,0.18);} 50%{box-shadow:0 0 26px rgba(239,68,68,0.55);} }
    @keyframes bsGrad    { 0%{background-position:0% 50%;} 100%{background-position:300% 50%;} }
    @keyframes bsScan    { 0%{background-position:0 -260px;} 100%{background-position:0 100vh;} }
    @keyframes bsGridDrift { 0%{transform:translate(0,0);} 100%{transform:translate(48px,48px);} }
    @keyframes bsSpin    { from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
    @keyframes bsTabGlow { 0%,100%{box-shadow:0 0 10px rgba(11,110,253,0.4);} 50%{box-shadow:0 0 22px rgba(20,184,166,0.6);} }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="bs-side-h">System Controls</div>', unsafe_allow_html=True)
    demo_mode = st.checkbox("Demo Mode: Simulate outbreak spike", value=False)
    if demo_mode:
        st.warning(
            "**SIMULATION ONLY** — Demo mode shows a hypothetical outbreak scenario. "
            "This is NOT real data."
        )

    st.markdown('<div class="bs-side-h">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bs-side-row"><span>Model</span><b>Early-Warning XGBoost</b></div>
        <div class="bs-side-row"><span>F1 Score</span><b>0.889</b></div>
        <div class="bs-side-row"><span>ROC-AUC</span><b>0.951</b></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bs-side-h">Data Sources</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bs-side-row"><span>WHO RSS Feed</span><b>Live</b></div>
        <div class="bs-side-row"><span>Google Trends</span><b>18 diseases</b></div>
        <div class="bs-side-row"><span>Georgetown DON</span><b>1,093 records</b></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bs-side-h">Research Disclaimer</div>', unsafe_allow_html=True)
    st.caption("BioSignal v1.0 — Research & Portfolio Project. Not medical advice.")


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
summary   = load_json("risk_summary.json")
scores_df = load_csv("risk_scores.csv")
ew_df     = load_csv("early_warnings.csv")

total_scored = summary.get("total_scored",         len(scores_df))
critical     = summary.get("critical_risk",         0)
high         = summary.get("high_risk",             0)
ew_signals   = summary.get("early_warning_signals", len(ew_df))
true_ew      = summary.get("true_early_warnings",   0)
scored_at    = summary.get("scored_at",             datetime.now().strftime("%Y-%m-%d %H:%M"))


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="bs-hero">
        <h1>BioSignal</h1>
        <div class="bs-sub">Epidemic Early Warning Intelligence Platform</div>
        <div class="bs-desc">
            WHO Reports &nbsp;•&nbsp; Google Trends &nbsp;•&nbsp;
            Explainable ML &nbsp;•&nbsp; Counterfactual Analysis
        </div>
        <div class="bs-badges">
            <span class="bs-sbadge live"><span class="bs-dot"></span> Live Monitoring</span>
            <span class="bs-sbadge">Model: Early-Warning XGBoost</span>
            <span class="bs-sbadge">System Status: Operational</span>
            <span class="bs-sbadge warnb">Research Use Only</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if true_ew > 0:
    system_status = "ALERT — Unconfirmed trend spike detected"
    status_color  = "#EF4444"
elif ew_signals > 0:
    system_status = "WATCH — Trend spike detected, WHO confirmed"
    status_color  = "#FACC15"
else:
    system_status = "Monitoring — No unconfirmed critical spike"
    status_color  = "#22C55E"

st.markdown(
    f'<div class="bs-status-bar">'
    f'<span style="color:{status_color};font-weight:700;">{system_status}</span>'
    f'<span style="color:#64748B;font-size:0.84em;margin-left:auto;">'
    f'Last updated: {scored_at}</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# SYSTEM FRESHNESS CHECK PANEL
# (built via string concatenation — fixes raw HTML tags showing as text)
# ----------------------------------------------------------------------------
freshness_files = [
    "risk_scores.csv",
    "early_warnings.csv",
    "risk_summary.json",
    "high_risk_alerts.csv",
]
IMPORTANT_FILES = {"risk_scores.csv", "early_warnings.csv", "risk_summary.json"}

freshness_results  = [file_freshness(f) for f in freshness_files]
important_results  = [r for r in freshness_results if r["file"] in IMPORTANT_FILES]
any_missing         = any(r["status"] == "Missing" for r in important_results)
any_stale           = any(r["status"] == "Stale"   for r in important_results)

if any_missing:
    overall_status, overall_color = "Pipeline Incomplete", "#EF4444"
elif any_stale:
    overall_status, overall_color = "Stale Outputs", "#FACC15"
else:
    overall_status, overall_color = "Live / Fresh", "#22C55E"

table_rows = ""
for r in freshness_results:
    table_rows += (
        f"<tr>"
        f"<td class='bs-fresh-filename'>{r['file']}</td>"
        f"<td>{r['last_modified']}</td>"
        f"<td>{r['age']}</td>"
        f"<td>{freshness_badge(r['status'])}</td>"
        f"</tr>"
    )

stale_warning_html = (
    '<div class="bs-stale-warning">Outputs may be stale. Rerun the pipeline.</div>'
    if any_stale or any_missing else ""
)

refresh_code_html = (
    "python src/data_collection.py<br>"
    "python src/risk_scorer.py<br>"
    "python src/explain.py<br>"
    "streamlit run app.py"
)

fresh_panel_html = (
    '<div class="bs-fresh-panel">'
    '<div class="bs-fresh-header">'
    '<div class="bs-fresh-title">System Freshness Check</div>'
    f'<span style="background:{overall_color}1A;color:{overall_color};'
    f'border:1px solid {overall_color}55;padding:5px 14px;'
    'border-radius:999px;font-weight:700;font-size:0.78rem;'
    'text-transform:uppercase;letter-spacing:0.4px;">'
    f'{overall_status}</span>'
    '</div>'
    '<table class="bs-fresh-table"><thead><tr>'
    '<th>File</th><th>Last Modified</th><th>Age</th><th>Status</th>'
    '</tr></thead><tbody>'
    f'{table_rows}'
    '</tbody></table>'
    f'{stale_warning_html}'
    '<div class="bs-refresh-box">'
    '<div class="bs-refresh-title">To refresh live data, run:</div>'
    f'<code>{refresh_code_html}</code>'
    '<p>Restarting Streamlit only reloads the dashboard. It does not fetch '
    'new WHO or Google Trends data unless the pipeline above is rerun.</p>'
    '</div>'
    '</div>'
)

st.markdown(fresh_panel_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def metric_card(label, value, helper="", crit=False, glow=False):
    cls = ("crit " if crit else "") + ("crit-glow" if glow else "")
    return f"""
    <div class="bs-card {cls}">
        <div class="bs-label">{label}</div>
        <div class="bs-num">{value}</div>
        <div class="bs-help">{helper}</div>
    </div>
    """


def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Risk Alerts",
    "Early Warning Monitor",
    "Leakage Demo",
    "SHAP Explainability",
    "Counterfactual What-if",
    "Timeline",
    "About",
])


# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tabs[0]:

    if demo_mode:
        st.error(
            "**DEMO MODE ACTIVE** — Simulated outbreak scenario shown below. "
            "This is NOT real data."
        )
        with st.expander("Simulated Alert — Dengue / India (DEMO ONLY)", expanded=True):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Simulated Risk Score", "87.4")
            dc2.markdown("**Level:** " + risk_badge("Critical Risk"), unsafe_allow_html=True)
            dc3.metric("Google Trends Spike", "4.2x above baseline")
            st.markdown("**WHO Report:** Not yet published")
            st.markdown("**Signal:** True Early Warning — Unconfirmed")
            st.info(
                "**Demo explanation:** This simulates how BioSignal behaves when "
                "Google Trends shows a 4.2x spike for dengue in India but WHO has "
                "not published an outbreak report yet. This would be flagged as a "
                "genuine early warning for analyst review. "
                "*This is a model-based demo scenario, not a real outbreak.*"
            )
        st.divider()

    st.markdown('<div class="bs-h">System Status</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Articles Scored",    total_scored)
    c2.metric("Critical",           critical)
    c3.metric("High Risk",          high)
    c4.metric("EW Signals",         ew_signals)
    c5.metric("Unconfirmed",        true_ew)
    c6.metric("Diseases Monitored", 18)

    st.divider()
    st.markdown('<div class="bs-h">Why BioSignal Is Different</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="bs-panel red">'
            '<h4 style="color:#EF4444;">What normal dashboards do</h4>'
            '<ul>'
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
            '<div class="bs-panel green">'
            '<h4 style="color:#22C55E;">What BioSignal does</h4>'
            '<ul>'
            '<li>Detects search spikes before WHO confirms</li>'
            '<li>Flags signals WHO has not reported yet</li>'
            '<li>SHAP + counterfactual explanations</li>'
            '<li>Live WHO + Google Trends pipeline</li>'
            '<li>8–18 day lead time detection goal</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="bs-h">How BioSignal Works</div>', unsafe_allow_html=True)
        st.markdown(
            "**Step 1 — Data Collection**\n"
            "BioSignal fetches live WHO outbreak articles and monitors "
            "Google Trends search volume for 18 epidemic diseases weekly.\n\n"
            "**Step 2 — Early Signal Detection**\n"
            "When search volume spikes above 1.5× baseline for a disease, "
            "BioSignal flags it as a potential early warning — even before "
            "WHO publishes anything.\n\n"
            "**Step 3 — ML Risk Scoring**\n"
            "An XGBoost early-warning model scores each outbreak on 0–100. "
            "Critically, it uses **no outcome features** — no deaths, no case counts, "
            "no severity. Only early signals: NLP urgency, disease risk profiles, "
            "seasonal patterns, and Trends data.\n\n"
            "**Step 4 — Explainability**\n"
            "Every prediction is explained using SHAP values and counterfactual "
            "analysis — showing exactly which signals drove the score and what "
            "would reduce it."
        )
    with col2:
        st.markdown('<div class="bs-h">Model Metrics</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="bs-h">Data Sources</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    sc1.info("**WHO RSS Feed**\nLive outbreak articles from who.int — fetched daily")
    sc2.info("**Google Trends**\nWeekly search volume for 18 epidemic diseases worldwide")
    sc3.info("**Georgetown DON**\n1,093 real WHO outbreak records used for training (1996–2015)")


# ============================================================================
# TAB 2 — RISK ALERTS
# ============================================================================
with tabs[1]:
    st.markdown('<div class="bs-h">Live Outbreak Risk Alerts</div>', unsafe_allow_html=True)
    st.caption(
        "Risk scores are model-based early-warning signals. "
        "Requires analyst verification. Not a confirmed diagnosis."
    )

    if not scores_df.empty:
        fc1, fc2, fc3 = st.columns(3)
        diseases  = ["All"] + sorted(scores_df["disease"].unique().tolist())
        countries = ["All"] + sorted(scores_df["country"].unique().tolist())
        levels    = ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
        sel_disease = fc1.selectbox("Filter by Disease",    diseases, key="ra_d")
        sel_country = fc2.selectbox("Filter by Country",    countries, key="ra_c")
        sel_level   = fc3.selectbox("Filter by Risk Level", levels,   key="ra_l")

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
            st.markdown(
                '<div class="bs-h" style="margin-top:8px;">High Priority Alerts</div>',
                unsafe_allow_html=True,
            )
            for _, row in high_crit.iterrows():
                st.markdown(alert_card_html(row), unsafe_allow_html=True)
            st.divider()

        st.markdown('<div class="bs-h">All Scored Articles</div>', unsafe_allow_html=True)
        display_cols = [
            c for c in [
                "date", "disease", "country", "risk_score", "risk_level",
                "confidence_band", "top_risk_reason", "spike_ratio_used", "has_spike",
            ] if c in filtered.columns
        ]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="bs-missing">'
            '<div class="bs-missing-title">No risk scores found.</div>'
            '<div class="bs-missing-hint">Run: python src/risk_scorer.py</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB 3 — EARLY WARNING MONITOR
# ============================================================================
with tabs[2]:
    st.markdown('<div class="bs-h">Early Warning Monitor</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="bs-panel amber">'
        '<h4 style="color:#FACC15;">What is a True Early Warning?</h4>'
        '<p>A <strong>True Early Warning</strong> occurs when:<br>'
        '&nbsp;&nbsp;• Google Trends spike detected (recent 4-record spike ratio ≥1.5× baseline)<br>'
        '&nbsp;&nbsp;• No recent WHO outbreak report for this disease<br>'
        '&nbsp;&nbsp;• ML risk score is High or Critical<br><br>'
        'This means the public is searching for a disease <strong>before</strong> '
        'WHO has officially confirmed it. These signals historically appear '
        '<strong>8–18 days before</strong> official reports. '
        'This is a model-based early warning signal requiring analyst verification.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "| Signal Level | Recent 4-Record Spike Ratio | Meaning |\n"
        "|---|---|---|\n"
        "| Alert   | ≥ 3.0× | Major spike — high priority |\n"
        "| Warning | ≥ 2.0× | Significant elevation |\n"
        "| Watch   | ≥ 1.5× | Elevated — monitor |"
    )

    if not ew_df.empty:
        true_ew_df  = ew_df[ew_df["early_warning"] == True]
        who_conf_df = ew_df[ew_df["early_warning"] == False]

        ew_c1, ew_c2, ew_c3 = st.columns(3)
        ew_c1.metric("Total Signals",       len(ew_df))
        ew_c2.metric("True Early Warnings", len(true_ew_df),
                     help="Spike detected but no WHO report yet")
        ew_c3.metric("WHO Confirmed",       len(who_conf_df))

        if not true_ew_df.empty:
            st.markdown(
                '<div class="bs-h" style="margin-top:8px;">'
                'Unconfirmed Signals — Analyst Review Required</div>',
                unsafe_allow_html=True,
            )
            st.error(
                "These diseases show elevated search interest with "
                "no recent WHO outbreak report. Model-based early warning "
                "signal requiring analyst verification."
            )
            for _, r in true_ew_df.iterrows():
                spike_col = "spike_ratio_used" if "spike_ratio_used" in r else "spike_ratio"
                st.markdown(
                    f"{signal_badge(r['signal_level'])} "
                    f"**{r['disease'].title()}** — "
                    f"Recent 4-record spike ratio: **{r[spike_col]}×** above baseline | "
                    f"No WHO report",
                    unsafe_allow_html=True,
                )
        else:
            st.success(
                "No unconfirmed spikes detected today. "
                "All signals are WHO-confirmed or below threshold."
            )

        if not who_conf_df.empty:
            st.markdown(
                '<div class="bs-h" style="margin-top:8px;">WHO-Confirmed Signals</div>',
                unsafe_allow_html=True,
            )
            st.success("These spikes are already matched with a WHO outbreak report.")
            for _, r in who_conf_df.iterrows():
                spike_col = "spike_ratio_used" if "spike_ratio_used" in r else "spike_ratio"
                st.markdown(
                    f"{signal_badge(r['signal_level'])} "
                    f"**{r['disease'].title()}** — "
                    f"Recent 4-record spike ratio: **{r[spike_col]}×** | WHO confirmed",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown('<div class="bs-h">All Early Warning Data</div>', unsafe_allow_html=True)
        ew_display_cols = [
            c for c in [
                "disease", "signal_level", "spike_ratio_used", "latest_spike_ratio",
                "recent_avg_spike_ratio", "baseline_avg", "who_reported",
                "early_warning", "date_checked",
            ] if c in ew_df.columns
        ]
        st.dataframe(ew_df[ew_display_cols], use_container_width=True)
    else:
        st.markdown(
            '<div class="bs-missing">'
            '<div class="bs-missing-title">No early warning data.</div>'
            '<div class="bs-missing-hint">Run: python src/risk_scorer.py</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if demo_mode:
        st.divider()
        st.error("**DEMO SIMULATION — NOT REAL DATA**")
        st.markdown(
            '<div class="bs-h">Simulated Early Warning: Dengue / India</div>',
            unsafe_allow_html=True,
        )
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        sim_c1.metric("Google Trends Spike", "4.2×")
        sim_c2.metric("WHO Report",          "None")
        sim_c3.metric("Signal Level",        "Alert")
        st.markdown(
            "**Simulated risk score:** 87.4 (Critical Risk)\n\n"
            "**What BioSignal would do:** Flag this as a True Early Warning, "
            "generate a SHAP explanation, and recommend immediate analyst review.\n\n"
            "*This is a demo scenario showing how BioSignal behaves during an "
            "active unconfirmed outbreak. Not a real prediction.*"
        )


# ============================================================================
# TAB 4 — LEAKAGE DEMO
# ============================================================================
with tabs[3]:
    st.markdown('<div class="bs-h">Leakage Demonstration</div>', unsafe_allow_html=True)
    st.markdown(
        "This tab demonstrates one of the most important concepts in ML for "
        "epidemic prediction: **target leakage**."
    )

    st.markdown(
        '<div class="bs-panel red">'
        '<h4 style="color:#EF4444;">What is target leakage?</h4>'
        '<p>The BioSignal label is: '
        '<code>escalated = 1 if deaths ≥ 10 OR cases ≥ 100</code><br><br>'
        'If we train a model using <strong>deaths</strong> and <strong>cases</strong> '
        'as input features, the model is essentially given the answer. '
        'It achieves perfect accuracy — but it cannot make predictions '
        '<em>before</em> deaths and cases are known. '
        'This is called <strong>target leakage</strong> and makes the model '
        'useless for real early warning.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bs-h">Model Comparison</div>', unsafe_allow_html=True)
    comparison_data = {
        "Model":         ["Diagnostic Model",                           "Early-Warning Model"],
        "Features Used": ["Includes deaths, cases, severity (leakage)", "Removes deaths, cases, severity"],
        "F1":            [1.000,                                         0.889],
        "ROC-AUC":       [1.000,                                         0.951],
        "Usable?":       ["Leakage — unrealistic",                       "Honest — deployable"],
        "Deployed?":     ["No",                                          "Yes"],
    }
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

    st.markdown(
        '<div class="bs-h">Outcome Features Removed from Early-Warning Model</div>',
        unsafe_allow_html=True,
    )
    removed = [
        "severity_score", "cases_total", "deaths", "log_cases", "log_deaths",
        "case_fatality_ratio", "has_deaths", "has_cases", "cases_per_death",
        "outbreak_relevance_score",
    ]
    col1, col2 = st.columns(2)
    for i, feat in enumerate(removed):
        (col1 if i % 2 == 0 else col2).markdown(f"- `{feat}`")

    st.caption(
        "These 10 features were removed because they are derived directly "
        "from the label formula. Using them would mean the model already "
        "knows the answer at prediction time."
    )

    st.divider()
    imp_df = load_csv("early_warning_feature_importances.csv")
    if not imp_df.empty:
        st.markdown(
            '<div class="bs-h">Early-Warning Model — Top Feature Importances</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(imp_df.head(15), use_container_width=True)


# ============================================================================
# TAB 5 — SHAP EXPLAINABILITY
# ============================================================================
with tabs[4]:
    st.markdown('<div class="bs-h">SHAP Explainability</div>', unsafe_allow_html=True)
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
        sc2.metric("Features Analyzed", shap_summary.get("total_features",       0))

    shap_imp = load_csv("shap_global_importance.csv", "python src/explain.py")
    if not shap_imp.empty:
        st.markdown(
            '<div class="bs-h">Global Feature Importance (Mean |SHAP|)</div>',
            unsafe_allow_html=True,
        )
        display_shap = shap_imp[[
            "importance_rank", "feature", "mean_abs_shap", "human_readable_feature",
        ]].head(15)
        st.dataframe(display_shap, use_container_width=True)

    st.divider()
    bar_path = OUTPUT / "shap_global_bar.png"
    top_path = OUTPUT / "shap_top_features.png"
    if bar_path.exists() or top_path.exists():
        st.markdown(
            '<div class="bs-h">SHAP Feature Importance Charts</div>',
            unsafe_allow_html=True,
        )
        ic1, ic2 = st.columns(2)
        if bar_path.exists():
            ic1.image(str(bar_path), caption="Global SHAP — Top 15 Features",
                      use_container_width=True)
        if top_path.exists():
            ic2.image(str(top_path), caption="Top 10 Early-Warning Features",
                      use_container_width=True)

    st.divider()
    local_df = load_csv("shap_local_explanations.csv", "python src/explain.py")
    if not local_df.empty:
        st.markdown('<div class="bs-h">Local Alert Explanations</div>', unsafe_allow_html=True)
        st.caption("These explain why each specific high-risk alert was scored high.")
        # Pure HTML cards — no st.expander() — avoids the same .arrow_right
        # icon-rendering bug that affected Risk Alerts.
        for _, row in local_df.iterrows():
            disease     = str(row['disease']).title()
            country     = str(row['country'])
            score       = row['risk_score']
            level       = str(row['risk_level'])
            confidence  = str(row.get('confidence_band', '—'))
            explanation = str(row.get('explanation_text', '—'))
            action      = get_recommended_action(level)

            feat_rows_html = ""
            for n in [1, 2, 3]:
                f = row.get(f"top_positive_feature_{n}", "")
                v = row.get(f"top_positive_value_{n}",   0)
                if f:
                    feat_rows_html += (
                        f'<div class="bs-alert-row">'
                        f'<span class="bs-alert-label">{f}</span>'
                        f'<span class="bs-alert-value">SHAP: {v}</span>'
                        f'</div>'
                    )

            border_colors = {
                "Critical Risk": "#EF4444", "High Risk": "#F97316",
                "Medium Risk": "#FACC15",   "Low Risk": "#22C55E",
            }
            bc = border_colors.get(level, "#94A3B8")

            st.markdown(
                f'<div class="bs-alert-card" style="border-left-color:{bc};">'
                f'<div class="bs-alert-header"><div class="bs-alert-title">'
                f'<span class="bs-alert-name">{disease}</span>'
                f'<span class="bs-alert-sep">·</span>'
                f'<span class="bs-alert-country">{country}</span>'
                f'<span class="bs-alert-sep">·</span>'
                f'<span class="bs-alert-score">Score: {score}</span>'
                f'{risk_badge(level)}'
                f'</div></div>'
                f'<div class="bs-alert-body">'
                f'<div class="bs-alert-row"><span class="bs-alert-label">Confidence</span>'
                f'<span class="bs-alert-value">{confidence}</span></div>'
                f'<div class="bs-alert-row"><span class="bs-alert-label">Explanation</span>'
                f'<span class="bs-alert-value">{explanation}</span></div>'
                f'{feat_rows_html}'
                f'<div class="bs-alert-action">Analyst action: {action}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# TAB 6 — COUNTERFACTUAL WHAT-IF
# ============================================================================
with tabs[5]:
    st.markdown(
        '<div class="bs-h">Counterfactual What-if Explanations</div>',
        unsafe_allow_html=True,
    )
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

            st.markdown(f"### {disease.title()} — {country}")
            st.markdown(
                f"**Current:** {risk_badge(curr_level)} Score: {curr_score}",
                unsafe_allow_html=True,
            )
            st.markdown("**What would reduce the risk?**")

            for _, row in disease_cf.iterrows():
                wc1, wc2, wc3, wc4 = st.columns([3, 2, 2, 2])
                wc1.markdown(f"**{row['counterfactual_scenario']}**")
                wc2.markdown(
                    f"{row['changed_feature']}: "
                    f"{row['original_value']} → {row['new_value']}"
                )
                wc3.metric(
                    "New Risk Score",
                    row["new_predicted_risk_score"],
                    delta=f"-{row['risk_reduction']}",
                    delta_color="inverse",
                )
                wc4.markdown(risk_badge(row["new_risk_level"]), unsafe_allow_html=True)
                st.caption(row.get("counterfactual_explanation", ""))
            st.divider()
    else:
        st.markdown(
            '<div class="bs-missing">'
            '<div class="bs-missing-title">No counterfactual data.</div>'
            '<div class="bs-missing-hint">Run: python src/explain.py</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB 7 — TIMELINE
# ============================================================================
with tabs[6]:
    st.markdown('<div class="bs-h">Outbreak Surveillance Timeline</div>', unsafe_allow_html=True)
    st.markdown(
        "This view shows all scored WHO articles as a surveillance timeline — "
        "making BioSignal look and feel like a monitoring system rather than "
        "a one-time analysis."
    )

    if not scores_df.empty:
        timeline_cols = [
            c for c in [
                "date", "disease", "country", "risk_score", "risk_level",
                "has_spike", "spike_ratio_used", "top_risk_reason", "confidence_band",
            ] if c in scores_df.columns
        ]
        timeline = (
            scores_df[timeline_cols]
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True)
        )

        def highlight_risk(row):
            palette = {
                "Critical Risk": "background-color:#EF444422",
                "High Risk":     "background-color:#F9731622",
                "Medium Risk":   "background-color:#FACC1522",
                "Low Risk":      "background-color:#22C55E22",
            }
            return [palette.get(row.get("risk_level", ""), "")] * len(row)

        st.dataframe(
            timeline.style.apply(highlight_risk, axis=1),
            use_container_width=True,
        )

        st.divider()
        st.markdown('<div class="bs-h">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = scores_df["risk_level"].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Critical Risk", risk_counts.get("Critical Risk", 0))
        col2.metric("High Risk",     risk_counts.get("High Risk",     0))
        col3.metric("Medium Risk",   risk_counts.get("Medium Risk",   0))
        col4.metric("Low Risk",      risk_counts.get("Low Risk",      0))
    else:
        st.markdown(
            '<div class="bs-missing">'
            '<div class="bs-missing-title">No timeline data.</div>'
            '<div class="bs-missing-hint">Run: python src/risk_scorer.py</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB 8 — ABOUT
# ============================================================================
with tabs[7]:
    st.markdown('<div class="bs-h">About BioSignal</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="bs-panel accent">'
        '<h4>What BioSignal Does</h4>'
        '<p>BioSignal is a research-oriented epidemic early-warning intelligence '
        'platform that combines WHO outbreak reports, Google Trends disease search '
        'patterns, NLP, and explainable machine learning to generate model-based '
        'public health risk signals — without relying on outcome data such as '
        'deaths or case counts.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="bs-panel"><h4>Why It Is Different</h4>'
        '<p>It surfaces pre-escalation signals rather than reporting confirmed '
        'outbreaks after the fact, and pairs every model-based risk score with SHAP '
        'explanations and counterfactual what-if analysis so analysts can understand '
        'and verify the reasoning behind each signal.</p></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="bs-h">Pipeline</div>', unsafe_allow_html=True)
    st.code(
        "1. Data Collection    -> WHO RSS + Google Trends  (src/data_collection.py)\n"
        "2. Historical Data    -> Georgetown DON database  (src/process_don.py)\n"
        "3. Data Cleaning      -> Merge + standardise      (src/cleaning.py)\n"
        "4. NLP Pipeline       -> spaCy entity extraction  (src/nlp_pipeline.py)\n"
        "5. Feature Engineering-> Spike detection          (src/features.py)\n"
        "6. Model Training     -> XGBoost early-warning    (src/model.py)\n"
        "7. Risk Scoring       -> Live outbreak scoring    (src/risk_scorer.py)\n"
        "8. Explainability     -> SHAP + counterfactuals   (src/explain.py)\n"
        "9. Dashboard          -> This Streamlit app       (app.py)",
        language="text",
    )

    st.divider()
    st.markdown('<div class="bs-h">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        "Python · XGBoost · SHAP · spaCy · Streamlit · "
        "Google Trends · WHO RSS · Pandas · Scikit-learn"
    )

    st.divider()
    st.markdown(
        '<div class="bs-panel green"><h4>Research Disclaimer</h4>'
        '<p>Research use only. Outputs are model-based risk scores and early warning '
        'signals that require analyst verification. BioSignal does not provide medical '
        'advice and does not produce confirmed diagnoses.</p></div>',
        unsafe_allow_html=True,
    )