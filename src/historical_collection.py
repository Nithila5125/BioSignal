# src/historical_collection.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Historical Data Collection
#
# Strategy (3 sources, all free, no scraping issues):
#
# Source 1: Georgetown University WHO DON database (GitHub)
#           2,789 real WHO outbreak records 1996-2019
#           Direct CSV download — no scraping needed
#
# Source 2: WHO DON direct article URLs (DON1 to DON600)
#           WHO uses predictable URLs like /item/2024-DON545
#           We hit them directly — no pagination needed
#
# Source 3: Existing WHO RSS + Trends you already have
#
# HOW TO RUN:
#   python src/historical_collection.py
# ═══════════════════════════════════════════════════════════════

import re
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

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
log = logging.getLogger("biosignal.historical")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

# Georgetown University WHO DON retrospective database
# Published in PLOS Global Public Health — peer reviewed
# Contains all 2,789 WHO DON reports from 1996-2019
# Direct raw CSV download from GitHub
GEORGETOWN_DON_CSV = (
    "https://raw.githubusercontent.com/cghss/dons/master/Data/dons.csv"
)

# WHO DON articles follow this URL pattern since ~2020
# Format: https://www.who.int/emergencies/disease-outbreak-news/item/YYYY-DONXXX
# We try DON numbers from 400 to 600 to cover 2020-2026
WHO_DON_ITEM_BASE = "https://www.who.int/emergencies/disease-outbreak-news/item"

DISEASE_KEYWORD_MAP = {
    "yellow fever": [r"\byellow fever\b"],
    "lassa fever":  [r"\blassa fever\b", r"\blassa\b"],
    "rift valley":  [r"\brift valley fever\b", r"\brvf\b"],
    "dengue":       [r"\bdengue fever\b", r"\bdengue\b"],
    "mpox":         [r"\bmpox\b", r"\bmonkeypox\b"],
    "marburg":      [r"\bmarburg\b"],
    "cholera":      [r"\bcholera\b"],
    "ebola":        [r"\bebola\b", r"\bebv\b"],
    "plague":       [r"\bplague\b", r"\byersinia pestis\b"],
    "influenza":    [r"\binfluenza\b", r"\bavian flu\b",
                     r"\bbird flu\b", r"\bh5n1\b", r"\bh1n1\b"],
    "covid":        [r"\bcovid\b", r"\bsars-cov\b", r"\bcoronavirus\b"],
    "measles":      [r"\bmeasles\b"],
    "meningitis":   [r"\bmeningitis\b", r"\bmeningococcal\b"],
    "polio":        [r"\bpolio\b", r"\bpoliovirus\b"],
    "typhoid":      [r"\btyphoid\b"],
}

HIGH_KEYWORDS = [
    "death", "deaths", "fatal", "fatality", "killed",
    "epidemic", "emergency", "critical", "severe", "surge",
]
MEDIUM_KEYWORDS = [
    "cases", "hospitaliz", "spreading", "warning", "alert",
    "monitor", "concern", "risk", "affected",
]
ESCALATION_KEYWORDS = [
    "international concern", "pheic", "global alert",
    "situation report", "emergency committee", "rapid response",
    "who team deployed", "cross-border", "multiple countries",
    "exponential", "uncontrolled", "humanitarian crisis",
    "epidemic declared", "mass casualty", "overwhelmed",
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def clean_html(raw: str) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def detect_disease(title: str, summary: str) -> str:
    for text in [title, summary]:
        text_lower = text.lower()
        for disease, patterns in DISEASE_KEYWORD_MAP.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return disease
    return "unknown"


def detect_country(title: str, summary: str) -> str:
    for text in [title, summary]:
        text_lower = text.lower()
        for country in config.COUNTRIES:
            pattern = r"\b" + re.escape(country.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return country
    return "unknown"


def detect_severity(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return "high"
    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "medium"
    return "low"


def create_label(text: str, severity: str) -> int:
    """
    ML label: 1 = escalated, 0 = contained
    Based on escalation keywords + severity.
    """
    text_lower = text.lower()
    has_escalation = any(kw in text_lower for kw in ESCALATION_KEYWORDS)
    return 1 if (has_escalation or severity == "high") else 0


def outbreak_score(text: str) -> int:
    OUTBREAK_KW = [
        "outbreak", "epidemic", "cases", "disease", "virus",
        "infection", "deaths", "fatalities", "emergency", "alert",
    ]
    text_lower = text.lower()
    return sum(1 for kw in OUTBREAK_KW if kw in text_lower)


def safe_get(url: str, wait: float = 2.0):
    try:
        time.sleep(wait)
        resp = requests.get(url, headers=HEADERS, timeout=20)
        return resp if resp.status_code == 200 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# SOURCE 1 — GEORGETOWN WHO DON DATABASE (1996-2019)
# ═══════════════════════════════════════════════════════════════

def collect_georgetown_don() -> pd.DataFrame:
    """
    Downloads the Georgetown University WHO DON retrospective database.
    This is a peer-reviewed dataset of ALL 2,789 WHO outbreak reports
    from 1996-2019. Published in PLOS Global Public Health (2023).

    Direct CSV download from GitHub — no scraping, no rate limits.
    Takes about 10 seconds.
    """
    log.info("─" * 55)
    log.info("SOURCE 1 — Georgetown WHO DON Database (1996-2019)")
    log.info("─" * 55)
    log.info("Downloading peer-reviewed WHO outbreak dataset...")
    log.info(f"URL: {GEORGETOWN_DON_CSV}")

    try:
        resp = requests.get(GEORGETOWN_DON_CSV, timeout=30)
        if resp.status_code != 200:
            log.error(f"  Download failed — HTTP {resp.status_code}")
            return pd.DataFrame()

        from io import StringIO
        raw_df = pd.read_csv(StringIO(resp.text))
        log.info(f"  Downloaded {len(raw_df)} rows")
        log.info(f"  Columns: {raw_df.columns.tolist()}")

        # Map Georgetown columns to our standard format
        # Their columns may vary — we handle common variants
        records = []
        for _, row in raw_df.iterrows():
            # Try different possible column names
            title = str(
                row.get("title") or
                row.get("Title") or
                row.get("headline") or ""
            )
            summary = str(
                row.get("summary") or
                row.get("Summary") or
                row.get("description") or
                row.get("text") or ""
            )
            date_raw = str(
                row.get("date") or
                row.get("Date") or
                row.get("pub_date") or
                row.get("published") or ""
            )
            country_raw = str(
                row.get("country") or
                row.get("Country") or
                row.get("location") or ""
            )
            disease_raw = str(
                row.get("disease") or
                row.get("Disease") or
                row.get("pathogen") or ""
            )

            # Clean and standardize
            clean_title   = clean_html(title)
            clean_summary = clean_html(summary)[:400]
            combined      = f"{clean_title} {clean_summary} {disease_raw} {country_raw}"

            # Use their disease if available, else detect
            disease = detect_disease(clean_title, clean_summary)
            if disease == "unknown" and disease_raw and disease_raw != "nan":
                # Try to match their disease label to ours
                disease_lower = disease_raw.lower()
                for our_disease in DISEASE_KEYWORD_MAP:
                    if our_disease in disease_lower:
                        disease = our_disease
                        break

            # Use their country if available, else detect
            country = detect_country(clean_title, clean_summary)
            if country == "unknown" and country_raw and country_raw != "nan":
                # Check if their country matches our list
                for our_country in config.COUNTRIES:
                    if our_country.lower() in country_raw.lower():
                        country = our_country
                        break

            severity  = detect_severity(combined)
            escalated = create_label(combined, severity)

            # Parse date
            parsed_date = datetime.today().strftime("%Y-%m-%d")
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
                        "%B %d, %Y", "%d %B %Y"]:
                try:
                    parsed_date = datetime.strptime(
                        date_raw.strip(), fmt
                    ).strftime("%Y-%m-%d")
                    break
                except (ValueError, AttributeError):
                    continue

            records.append({
                "date":                     parsed_date,
                "title":                    clean_title,
                "disease":                  disease,
                "country":                  country,
                "severity":                 severity,
                "outbreak_relevance_score": outbreak_score(combined),
                "escalated":                escalated,
                "summary":                  clean_summary,
                "link":                     str(row.get("url") or row.get("link") or ""),
                "source":                   "georgetown_who_don",
            })

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

        out_path = config.RAW_DIR / "georgetown_don.csv"
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        matched   = (df["disease"] != "unknown").sum()
        escalated = df["escalated"].sum()
        log.info(f"  Saved {len(df)} records → georgetown_don.csv")
        log.info(f"  Disease matched: {matched}/{len(df)}")
        log.info(f"  Escalated (1): {escalated} | Contained (0): {len(df)-escalated}")
        log.info(f"  Top diseases: {df['disease'].value_counts().head(5).to_dict()}")

        return df

    except Exception as e:
        log.error(f"  Georgetown download failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# SOURCE 2 — WHO DON DIRECT URLS (2020-2026)
# ═══════════════════════════════════════════════════════════════

def collect_who_don_direct(
    start_don: int = 400,
    end_don: int = 600,
    years: list = None,
) -> pd.DataFrame:
    """
    WHO DON articles since ~2020 use predictable URLs:
    https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON545

    We try year + DON number combinations directly.
    Much more reliable than pagination scraping.
    Covers 2020-2026 (post-Georgetown database).
    """
    log.info("\n" + "─" * 55)
    log.info("SOURCE 2 — WHO DON Direct URLs (2020-2026)")
    log.info("─" * 55)

    if years is None:
        years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    records    = []
    total_tried = 0
    total_found = 0

    for year in years:
        log.info(f"  Trying year {year}...")
        year_found = 0

        for don_num in range(start_don, end_don + 1):
            url = f"{WHO_DON_ITEM_BASE}/{year}-DON{don_num}"
            total_tried += 1

            resp = safe_get(url, wait=random.uniform(1.5, 2.5))
            if not resp:
                continue

            # Parse article page
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            title_tag = (
                soup.find("h1") or
                soup.find("title")
            )
            title = clean_html(title_tag.get_text()) if title_tag else ""

            if not title or len(title) < 10:
                continue

            # Extract main content
            content_div = (
                soup.find("div", class_=re.compile(
                    r"sf-detail-body|detail-body|content|article", re.I
                )) or
                soup.find("article") or
                soup.find("main")
            )
            full_text = clean_html(
                content_div.get_text()
            ) if content_div else ""
            summary   = full_text[:400]
            combined  = f"{title} {summary}"

            # Extract date from URL or page
            parsed_date = f"{year}-01-01"
            date_tag = soup.find("time") or soup.find(
                class_=re.compile(r"date|timestamp|publish", re.I)
            )
            if date_tag:
                date_text = clean_html(date_tag.get_text())
                for fmt in ["%d %B %Y", "%B %d, %Y", "%Y-%m-%d"]:
                    try:
                        parsed_date = datetime.strptime(
                            date_text.strip(), fmt
                        ).strftime("%Y-%m-%d")
                        break
                    except (ValueError, AttributeError):
                        continue

            disease   = detect_disease(title, summary)
            country   = detect_country(title, summary)
            severity  = detect_severity(combined)
            escalated = create_label(combined, severity)

            records.append({
                "date":                     parsed_date,
                "title":                    title,
                "disease":                  disease,
                "country":                  country,
                "severity":                 severity,
                "outbreak_relevance_score": outbreak_score(combined),
                "escalated":                escalated,
                "summary":                  summary,
                "link":                     url,
                "source":                   f"who_don_{year}",
            })

            year_found  += 1
            total_found += 1

            if year_found % 10 == 0:
                log.info(f"    Found {year_found} articles for {year}...")

        log.info(f"  Year {year}: {year_found} articles found")

    if not records:
        log.warning("  No WHO DON direct articles found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    out_path = config.RAW_DIR / "who_don_recent.csv"
    df.to_csv(out_path, index=False)

    matched   = (df["disease"] != "unknown").sum()
    escalated = df["escalated"].sum()
    log.info(f"\n  Tried {total_tried} URLs | Found {total_found} articles")
    log.info(f"  Saved {len(df)} records → who_don_recent.csv")
    log.info(f"  Disease matched: {matched}/{len(df)}")
    log.info(f"  Escalated (1): {escalated} | Contained (0): {len(df)-escalated}")

    return df


# ═══════════════════════════════════════════════════════════════
# SOURCE 3 — COMBINE EVERYTHING
# ═══════════════════════════════════════════════════════════════

def combine_all_sources() -> pd.DataFrame:
    """
    Merges all sources into one clean training dataset.
    Deduplicates by title.
    Separates unknown disease rows for review.
    Saves final combined_historical.csv
    """
    log.info("\n" + "─" * 55)
    log.info("SOURCE 3 — Combining All Sources")
    log.info("─" * 55)

    all_dfs = []

    files_to_load = {
        "Georgetown DON":  config.RAW_DIR / "georgetown_don.csv",
        "WHO DON Recent":  config.RAW_DIR / "who_don_recent.csv",
        "WHO RSS":         config.WHO_RAW_FILE,
    }

    for name, path in files_to_load.items():
        if path.exists():
            df = pd.read_csv(path)
            if "escalated" not in df.columns:
                df["escalated"] = df.apply(
                    lambda r: create_label(
                        str(r.get("summary", "")),
                        str(r.get("severity", "low"))
                    ), axis=1
                )
            log.info(f"  {name}: {len(df)} rows")
            all_dfs.append(df)
        else:
            log.info(f"  {name}: not found — skipping")

    if not all_dfs:
        log.error("  No source files found.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate
    before   = len(combined)
    combined = combined.drop_duplicates(
        subset=["title"]
    ).reset_index(drop=True)
    log.info(f"  Removed {before - len(combined)} duplicates")

    # Split known vs unknown disease
    unknown_df = combined[combined["disease"] == "unknown"].copy()
    known_df   = combined[combined["disease"] != "unknown"].copy()

    if not unknown_df.empty:
        unknown_path = config.RAW_DIR / "combined_unknown_review.csv"
        unknown_df.to_csv(unknown_path, index=False)
        log.info(f"  Unknown rows: {len(unknown_df)} → combined_unknown_review.csv")

    known_df = known_df.sort_values(
        "date", ascending=False
    ).reset_index(drop=True)

    out_path = config.RAW_DIR / "combined_historical.csv"
    known_df.to_csv(out_path, index=False)

    escalated = known_df["escalated"].sum()
    log.info(f"\n  Final dataset: {len(known_df)} rows")
    log.info(f"  Escalated (1): {escalated} ({round(escalated/max(len(known_df),1)*100,1)}%)")
    log.info(f"  Contained (0): {len(known_df) - escalated}")
    log.info(f"  Top diseases: {known_df['disease'].value_counts().head(8).to_dict()}")

    return known_df


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    rows      = len(df) if not df.empty else 0
    escalated = int(df["escalated"].sum()) if not df.empty else 0
    ready     = rows >= 200

    print("\n" + "═" * 55)
    print("  BioSignal Historical Collection — Done")
    print("═" * 55)
    print(f"  Total records           : {rows}")
    print(f"  Escalated outbreaks (1) : {escalated}")
    print(f"  Contained outbreaks (0) : {rows - escalated}")
    print(f"  Ready for ML training   : {'✓ YES' if ready else '✗ Need more data'}")
    print(f"\n  Files saved:")
    print(f"    data/raw/georgetown_don.csv        ← 1996-2019 WHO records")
    print(f"    data/raw/who_don_recent.csv        ← 2020-2026 WHO records")
    print(f"    data/raw/combined_historical.csv   ← Final training dataset")
    print(f"\n  Next step: Phase 4 — Data Cleaning")
    print("═" * 55 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print("  BioSignal — Historical Data Collection")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    # Source 1: Georgetown database — instant download
    georgetown_df = collect_georgetown_don()

    # Source 2: WHO DON direct URLs — 2020 to 2026
    # Tries DON numbers 400-600 across recent years
    # Takes 5-10 minutes
    recent_df = collect_who_don_direct(
        start_don=400,
        end_don=600,
        years=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
    )

    # Source 3: Combine everything
    combined_df = combine_all_sources()

    print_summary(combined_df)


if __name__ == "__main__":
    main()