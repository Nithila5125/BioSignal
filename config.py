# config.py
# ─────────────────────────────────────────────
# BioSignal — Global Configuration
# All project-wide settings live here.
# ─────────────────────────────────────────────

import os
from pathlib import Path

# ── Project Identity ──────────────────────────
PROJECT_NAME    = "BioSignal"
PROJECT_TAGLINE = "Detects epidemic outbreak signals before WHO official reports"
PROJECT_VERSION = "1.0.0"

# ── Folder paths ──────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ── File paths ────────────────────────────────
WHO_RAW_FILE       = RAW_DIR / "who_reports.csv"
TRENDS_RAW_FILE    = RAW_DIR / "trends_raw.csv"
TRENDS_FAILED_FILE = RAW_DIR / "trends_failed.csv"
DON_RAW_FILE       = RAW_DIR / "DONdatabase.csv"
DON_PROCESSED_FILE = RAW_DIR / "don_processed.csv"
COMBINED_FILE      = RAW_DIR / "combined_historical.csv"
FEATURES_FILE      = PROCESSED_DIR / "features.csv"
MODEL_FILE         = BASE_DIR / "models" / "xgboost_model.pkl"
LEAD_TIME_MODEL    = BASE_DIR / "models" / "lead_time_model.pkl"
SHAP_OUTPUT        = BASE_DIR / "data" / "outputs" / "shap_values.csv"

# ── Diseases to track ─────────────────────────
DISEASES = [
    "cholera",
    "ebola",
    "dengue",
    "mpox",
    "marburg",
    "yellow fever",
    "plague",
    "lassa fever",
]

# ── Countries to detect ───────────────────────
# Longer names first — avoids substring collisions
# e.g. "Democratic Republic of the Congo" before "Congo"
COUNTRIES = [
    "Democratic Republic of the Congo",
    "Central African Republic",
    "Papua New Guinea",
    "South Sudan",
    "Sierra Leone",
    "Saudi Arabia",
    "Afghanistan",
    "Angola",
    "Bangladesh",
    "Brazil",
    "Cameroon",
    "Chad",
    "China",
    "Colombia",
    "Egypt",
    "Ethiopia",
    "Ghana",
    "Guinea",
    "Haiti",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Kenya",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi",
    "Mali",
    "Mexico",
    "Mozambique",
    "Myanmar",
    "Nepal",
    "Niger",
    "Nigeria",
    "Pakistan",
    "Philippines",
    "Somalia",
    "Sudan",
    "Syria",
    "Tanzania",
    "Uganda",
    "Ukraine",
    "Venezuela",
    "Yemen",
    "Zambia",
    "Zimbabwe",
]

# ── Google Trends settings ────────────────────
TRENDS_TIMEFRAME = "today 12-m"   # Last 12 months
TRENDS_GEO       = ""             # Worldwide

# ── Spike detection thresholds (Phase 6) ─────
SPIKE_RATIO_LOW    = 1.5   # 50% above baseline  = watch
SPIKE_RATIO_MEDIUM = 2.0   # 100% above baseline = warning
SPIKE_RATIO_HIGH   = 3.0   # 200% above baseline = alert

# ── Risk score thresholds (Phase 8) ──────────
RISK_LOW    = 30
RISK_MEDIUM = 60
RISK_HIGH   = 80

# ── Lead time settings (Phase 7) ─────────────
# Expected days between search spike and WHO report
LEAD_TIME_MIN = 3    # minimum days
LEAD_TIME_MAX = 30   # maximum days
LEAD_TIME_AVG = 14   # historical average (updated after training)

# ── Claude API briefing (Phase 10) ───────────
BRIEFING_MODEL      = "claude-sonnet-4-6"
BRIEFING_MAX_TOKENS = 300

# ── Dashboard settings (Phase 10) ────────────
DASHBOARD_TITLE       = "BioSignal — Epidemic Early Warning"
DASHBOARD_REFRESH_HRS = 24   # how often live data refreshes