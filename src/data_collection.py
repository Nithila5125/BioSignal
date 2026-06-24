# src/data_collection.py
# ═══════════════════════════════════════════════════════════════
# BioSignal — Epidemic Early Warning System
# Phase 3: Data Collection — Final Version
#
# HOW TO RUN:
#   Normal run:           python src/data_collection.py
#   Retry failed only:    python src/data_collection.py --retry
#   Force re-fetch all:   python src/data_collection.py --force
# ═══════════════════════════════════════════════════════════════

import re
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pytrends.request import TrendReq

# ── Add project root to path so config.py is importable ──────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

# ── Single clean logger ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("biosignal")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

WHO_RSS_URLS = [
    "https://www.who.int/rss-feeds/news-english.xml",
]

OUTBREAK_KEYWORDS = [
    "outbreak", "epidemic", "cases", "disease", "virus", "infection",
    "deaths", "fatalities", "emergency", "alert", "surveillance",
    "pathogen", "cholera", "ebola", "dengue", "mpox", "monkeypox",
    "marburg", "plague", "lassa", "influenza", "measles", "meningitis",
    "polio", "typhoid", "yellow fever", "rift valley", "covid",
    "health emergency", "who response",
]

# No duplicate keys — Python silently drops duplicates
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


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def clean_html(raw: str) -> str:
    """
    Strips ALL HTML tags and decodes HTML entities.
    Must be called BEFORE any disease/country detection.
    Example: "<p>Ebola &amp; cholera</p>" → "Ebola & cholera"
    """
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def parse_date(date_str: str) -> str:
    """
    Converts RSS date strings to YYYY-MM-DD.
    Falls back to today's date if parsing fails.
    """
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
    return datetime.today().strftime("%Y-%m-%d")


def is_outbreak_relevant(text: str) -> bool:
    """
    Returns True if article contains at least one outbreak keyword.
    Filters out general WHO news (blood supply, tobacco, etc.)
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in OUTBREAK_KEYWORDS)


def outbreak_relevance_score(text: str) -> int:
    """
    Counts outbreak keywords present in text.
    Higher score = more outbreak relevant.
    Used as a feature in the ML pipeline later.
    """
    text_lower = text.lower()
    return sum(1 for kw in OUTBREAK_KEYWORDS if kw in text_lower)


def detect_disease(title: str, summary: str) -> str:
    """
    Detects disease from title and summary.
    Checks both exact keywords and related terms.
    """
    text = f"{title} {summary}".lower()

    # Disease keyword map — includes aliases and related terms
    disease_map = {
        "ebola":         ["ebola", "ebola virus", "evd"],
        "marburg":       ["marburg", "marburg virus", "mvd"],
        "cholera":       ["cholera", "vibrio cholerae"],
        "dengue":        ["dengue", "dengue fever", "denv"],
        "mpox":          ["mpox", "monkeypox", "monkey pox"],
        "yellow fever":  ["yellow fever"],
        "plague":        ["plague", "yersinia pestis", "bubonic"],
        "lassa fever":   ["lassa", "lassa fever"],
        "covid":         ["covid", "covid-19", "sars-cov-2", "coronavirus"],
        "hantavirus":    ["hantavirus", "hanta virus", "hps",
                         "hantavirus pulmonary", "tenerife hantavirus",
                         "hantavirus response"],
        "influenza":     ["influenza", "flu outbreak", "h5n1", "h1n1",
                         "avian influenza", "bird flu"],
        "measles":       ["measles", "rubeola"],
        "meningitis":    ["meningitis", "meningococcal"],
        "polio":         ["polio", "poliomyelitis", "poliovirus"],
        "typhoid":       ["typhoid", "typhoid fever", "salmonella typhi"],
        "rift valley":   ["rift valley", "rift valley fever", "rvf"],
        "nipah":         ["nipah", "nipah virus"],
        "monkeypox":     ["monkeypox", "mpox"],
    }

    for disease, keywords in disease_map.items():
        for kw in keywords:
            if kw in text:
                return disease

    return "unknown"


def detect_country(title: str, summary: str) -> str:
    """
    Detects country using word boundaries.
    Checks title first to avoid false matches from
    diplomatic context in summaries (Geneva, Brasília etc.)
    Returns 'unknown' if no confident match.
    """
    for text in [title, summary]:
        text_lower = text.lower()
        for country in config.COUNTRIES:
            pattern = r"\b" + re.escape(country.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return country
    return "unknown"


def detect_severity(text: str) -> str:
    """
    Returns 'high', 'medium', or 'low' based on keywords.
    Text must be HTML-cleaned before calling.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return "high"
    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "medium"
    return "low"


# ═══════════════════════════════════════════════════════════════
# PART 1 — WHO DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect_who_data() -> pd.DataFrame:
    """
    Fetches WHO RSS feed.
    Filters outbreak-relevant articles only.
    Cleans HTML FIRST then detects disease/country/severity.
    Saves unknown articles separately for manual review.
    Saves to data/raw/who_reports.csv
    """
    log.info("─" * 55)
    log.info("PART 1 — WHO Outbreak Data")
    log.info("─" * 55)

    all_records = []

    for rss_url in WHO_RSS_URLS:
        log.info(f"Fetching: {rss_url}")
        try:
            resp = requests.get(rss_url, headers=HEADERS, timeout=20)

            if resp.status_code != 200:
                log.warning(f"  Skipped — HTTP {resp.status_code}")
                continue

            soup  = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")
            log.info(f"  Found {len(items)} articles — filtering...")

            kept    = 0
            skipped = 0

            for item in items:
                title_tag   = item.find("title")
                desc_tag    = item.find("description")
                pubdate_tag = item.find("pubDate")
                link_tag    = item.find("link")

                raw_title = title_tag.get_text(strip=True)   if title_tag   else ""
                raw_desc  = desc_tag.get_text(strip=True)    if desc_tag    else ""
                raw_date  = pubdate_tag.get_text(strip=True) if pubdate_tag else ""
                raw_link  = link_tag.get_text(strip=True)    if link_tag    else ""

                # ── Clean HTML FIRST — never detect from raw HTML ──
                clean_title   = clean_html(raw_title)
                clean_summary = clean_html(raw_desc)[:400]
                combined      = f"{clean_title} {clean_summary}"

                # ── Skip non-outbreak articles ─────────────────────
                if not is_outbreak_relevant(combined):
                    skipped += 1
                    continue

                record = {
                    "date":                     parse_date(raw_date),
                    "title":                    clean_title,
                    "disease":                  detect_disease(clean_title, clean_summary),
                    "country":                  detect_country(clean_title, clean_summary),
                    "severity":                 detect_severity(combined),
                    "outbreak_relevance_score": outbreak_relevance_score(combined),
                    "summary":                  clean_summary,
                    "link":                     raw_link,
                    "source":                   "who.int",
                }
                all_records.append(record)
                kept += 1

            log.info(f"  Kept {kept} outbreak-relevant | Skipped {skipped} general news")
            time.sleep(2)

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout — skipping {rss_url}")
        except requests.exceptions.ConnectionError:
            log.warning(f"  Connection error — skipping {rss_url}")
        except Exception as e:
            log.warning(f"  Unexpected error: {e}")

    if not all_records:
        log.error("No WHO data collected. Check internet connection.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Remove duplicates by link
    before = len(df)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    dupes = before - len(df)
    if dupes > 0:
        log.info(f"  Removed {dupes} duplicate articles")

    # Sort newest first
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # Save main CSV
    config.WHO_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.WHO_RAW_FILE, index=False)

    # Save unknown articles separately for manual review
    # Useful for improving keyword detection later
    unknown_df = df[df["disease"] == "unknown"].copy()
    if not unknown_df.empty:
        unknown_path = config.RAW_DIR / "who_unknown_review.csv"
        unknown_df.to_csv(unknown_path, index=False)
        log.info(f"  Unknown articles saved → who_unknown_review.csv")

    # Clean summary log — no dataframe rows printed
    matched = (df["disease"] != "unknown").sum()
    log.info(f"  Saved {len(df)} rows → who_reports.csv")
    log.info(f"  Disease detection: {matched}/{len(df)} matched")
    log.info(f"  Diseases found: {df['disease'].value_counts().to_dict()}")
    log.info(f"  Severity split: {df['severity'].value_counts().to_dict()}")

    return df


# ═══════════════════════════════════════════════════════════════
# PART 2 — GOOGLE TRENDS COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect_trends_data(force: bool = False) -> pd.DataFrame:
    """
    Fetches weekly Google Trends search volume per disease.
    Skips diseases already collected (safe to rerun).
    Saves each success immediately so progress is never lost.
    Saves failed diseases to trends_failed.csv.
    Uses exponential backoff on 429 rate limit errors.
    """
    log.info("─" * 55)
    log.info("PART 2 — Google Trends Data")
    log.info("─" * 55)

    trends_path = config.TRENDS_RAW_FILE
    failed_path = config.TRENDS_FAILED_FILE

    # Load existing data to skip already-collected diseases
    if trends_path.exists() and not force:
        existing_df  = pd.read_csv(trends_path)
        already_have = existing_df["disease"].unique().tolist()
        log.info(f"  Already collected: {already_have}")
    else:
        existing_df  = pd.DataFrame()
        already_have = []

    to_fetch = [d for d in config.DISEASES if d not in already_have]

    if not to_fetch:
        log.info("  All diseases already collected.")
        log.info("  Use --force flag to re-fetch everything.")
        return existing_df

    log.info(f"  To fetch: {to_fetch}")
    log.info("  Takes 10-15 minutes. Do NOT close terminal.")

    new_data        = []
    failed_diseases = []
    total           = len(to_fetch)

    for i, disease in enumerate(to_fetch, start=1):
        log.info(f"  [{i}/{total}] {disease}")
        success = False

        for attempt in range(1, 4):
            try:
                # Fresh session each attempt — resets cookies
                pt = TrendReq(
                    hl="en-US",
                    tz=360,
                    timeout=(15, 40),
                    retries=0,
                )
                pt.build_payload(
                    [disease],
                    cat=0,
                    timeframe=config.TRENDS_TIMEFRAME,
                    geo=config.TRENDS_GEO,
                )
                data = pt.interest_over_time()

                if data.empty:
                    log.warning(f"    No data returned (attempt {attempt}/3)")
                    time.sleep(15)
                    continue

                data = data.reset_index()
                data["disease"] = disease
                data = data.rename(columns={disease: "search_volume"})
                if "isPartial" in data.columns:
                    data = data.drop(columns=["isPartial"])
                data["date"] = (
                    pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
                )

                new_data.append(data)
                log.info(f"    ✓ {len(data)} weeks collected")
                success = True

                # Save partial result immediately after each success
                parts = (
                    [existing_df] + new_data
                    if not existing_df.empty
                    else new_data
                )
                partial = pd.concat(parts, ignore_index=True)
                trends_path.parent.mkdir(parents=True, exist_ok=True)
                partial.to_csv(trends_path, index=False)
                break

            except Exception as e:
                err = str(e)
                if "429" in err:
                    wait = 60 * attempt  # 60s → 120s → 180s
                    log.warning(
                        f"    Rate limit. Waiting {wait}s "
                        f"(attempt {attempt}/3)..."
                    )
                    time.sleep(wait)
                elif "timed out" in err.lower():
                    log.warning(
                        f"    Timeout. Retrying in 20s "
                        f"(attempt {attempt}/3)..."
                    )
                    time.sleep(20)
                else:
                    log.warning(
                        f"    Error (attempt {attempt}/3): {err[:60]}"
                    )
                    time.sleep(15)

        if not success:
            log.warning(f"    ✗ Failed: {disease}")
            failed_diseases.append(disease)

        if i < total:
            wait = random.randint(25, 35)
            log.info(f"    Waiting {wait}s...")
            time.sleep(wait)

    # Save failed list
    if failed_diseases:
        pd.DataFrame({"disease": failed_diseases}).to_csv(
            failed_path, index=False
        )
        log.info(f"  Failed list saved → trends_failed.csv")
    else:
        # Clear old failed file if everything succeeded
        if failed_path.exists():
            failed_path.unlink()

    # Build final combined dataframe
    all_parts = (
        [existing_df] + new_data if not existing_df.empty else new_data
    )
    if not all_parts:
        log.error("  No trends data collected at all.")
        return pd.DataFrame()

    final_df = pd.concat(all_parts, ignore_index=True)
    final_df.to_csv(trends_path, index=False)

    return final_df


# ═══════════════════════════════════════════════════════════════
# PART 3 — RETRY FAILED TRENDS
# ═══════════════════════════════════════════════════════════════

def retry_failed_trends() -> pd.DataFrame:
    """
    Reads trends_failed.csv and retries only those diseases.
    Safe to run multiple times.
    Already-collected diseases are never re-fetched.
    """
    failed_path = config.TRENDS_FAILED_FILE

    if not failed_path.exists():
        log.info("No trends_failed.csv found. Nothing to retry.")
        return pd.DataFrame()

    failed = pd.read_csv(failed_path)["disease"].tolist()
    if not failed:
        log.info("trends_failed.csv is empty. Nothing to retry.")
        return pd.DataFrame()

    log.info(f"Retrying failed diseases: {failed}")

    original_diseases = config.DISEASES
    config.DISEASES   = failed
    result            = collect_trends_data(force=False)
    config.DISEASES   = original_diseases

    return result


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(who_df: pd.DataFrame, trends_df: pd.DataFrame):
    """
    Prints clean final summary.
    Uses relative paths — no long Windows paths.
    """
    failed_path  = config.TRENDS_FAILED_FILE
    failed_list  = []
    if failed_path.exists():
        failed_list = pd.read_csv(failed_path)["disease"].tolist()

    who_rows      = len(who_df)    if not who_df.empty    else 0
    trends_rows   = len(trends_df) if not trends_df.empty else 0
    who_matched   = int((who_df["disease"] != "unknown").sum()) if not who_df.empty else 0
    trends_diseas = trends_df["disease"].nunique() if not trends_df.empty else 0
    total_diseas  = len(config.DISEASES)

    unknown_path  = config.RAW_DIR / "who_unknown_review.csv"

    print("\n" + "═" * 55)
    print("  BioSignal Data Collection Completed")
    print("═" * 55)
    print(f"  WHO rows saved        : {who_rows}")
    print(f"  WHO disease matched   : {who_matched}/{who_rows}")
    print(f"  Trends rows saved     : {trends_rows}")
    print(f"  Trends diseases saved : {trends_diseas}/{total_diseas}")
    if failed_list:
        print(f"  Failed trends         : {', '.join(failed_list)}")
    print(f"\n  Files:")
    print(f"    data/raw/who_reports.csv")
    print(f"    data/raw/trends_raw.csv")
    if failed_list:
        print(f"    data/raw/trends_failed.csv")
    if unknown_path.exists():
        print(f"    data/raw/who_unknown_review.csv  ← review these manually")
    print(f"\n  Next step: Phase 4 — Data Cleaning")
    print("═" * 55 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BioSignal — Phase 3 Data Collection"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry only failed Google Trends diseases",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch ALL Google Trends data",
    )
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  BioSignal — Phase 3: Data Collection")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 55 + "\n")

    if args.retry:
        log.info("Mode: RETRY failed trends only")
        trends_df = retry_failed_trends()
        who_df    = (
            pd.read_csv(config.WHO_RAW_FILE)
            if config.WHO_RAW_FILE.exists()
            else pd.DataFrame()
        )
    else:
        if args.force:
            log.info("Mode: FORCE refresh all trends")
        else:
            log.info("Mode: Normal run")

        who_df    = collect_who_data()
        trends_df = collect_trends_data(force=args.force)

    print_summary(who_df, trends_df)


if __name__ == "__main__":
    main()