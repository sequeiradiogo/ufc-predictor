"""
scrapers/ufc_fighter_nationality.py -- Scrape each fighter's hometown/country
from their ufc.com athlete profile page.

Neither db/ufc_ufcstats.db nor raw_data/ufc-master.csv carries fighter
nationality -- only fights.country (the *event's* location) exists. This
script fills that gap for the "home country vs foreign soil" analysis by
scraping the "Hometown" bio field on each fighter's ufc.com/athlete/<slug>
page and splitting it into city/country.

ufc.com geo-redirects to a localized mirror for non-US network egress (same
behavior noted in ufc_rankings_web.py), so the bio field label may render as
"Hometown" or "Ciudad natal" (or another locale) -- matched by keyword
instead of an exact label string. A fighter with no ufc.com athlete page (or
a slug that doesn't match, e.g. unusual accln/suffix formatting) soft-404s to
a "Search results" page with no .c-bio block; those rows are written with
country=None rather than dropped, so they're visible and re-checkable.

Checkpointed like scrape_history.py: already-scraped fighter_ids are skipped
on resume, and results are flushed to disk every CHECKPOINT_EVERY fighters.

Usage:
    python scrapers/ufc_fighter_nationality.py
    python scrapers/ufc_fighter_nationality.py --limit 50 --dry-run
"""
import argparse
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DB_UFCSTATS_PATH
from utils.logger import get_logger

log = get_logger(__name__)

OUT_CSV = ROOT_DIR / "raw_data" / "fighter_nationality.csv"
CHECKPOINT_EVERY = 25
REQUEST_DELAY_SEC = 0.4

_HOMETOWN_LABEL_RE = re.compile(r"natal|hometown|ciudad", re.IGNORECASE)

# ufc.com geo-redirects to a localized mirror per-request (observed flipping
# between English and Spanish across consecutive requests in the same
# browser session -- not a stable per-fighter or per-session locale, so it
# can't be forced off with Accept-Language/locale context options). Country
# names are normalized to canonical English here rather than relying on a
# consistent scrape locale.
COUNTRY_ALIASES = {
    "estados unidos": "United States",
    "brasil": "Brazil",
    "inglaterra": "England",
    "reino unido": "United Kingdom",
    "mexico": "Mexico",
    "méxico": "Mexico",
    "irlanda": "Ireland",
    "australia": "Australia",
    "canada": "Canada",
    "canadá": "Canada",
    "rusia": "Russia",
    "francia": "France",
    "alemania": "Germany",
    "gales": "Wales",
    "escocia": "Scotland",
    "nueva zelanda": "New Zealand",
    "sudafrica": "South Africa",
    "sudáfrica": "South Africa",
    "paises bajos": "Netherlands",
    "países bajos": "Netherlands",
    "polonia": "Poland",
    "suecia": "Sweden",
    "japon": "Japan",
    "japón": "Japan",
    "corea del sur": "South Korea",
    "china": "China",
}


def normalize_country(country: str) -> str | None:
    if not country:
        return None
    return COUNTRY_ALIASES.get(country.strip().lower(), country.strip())

_EXTRACT_JS = """
() => {
    const fields = Array.from(document.querySelectorAll('.c-bio__field'));
    for (const f of fields) {
        const label = f.querySelector('.c-bio__label');
        const text = f.querySelector('.c-bio__text');
        if (label && text) {
            const rows = { label: label.textContent.trim(), text: text.textContent.trim() };
            if (!window.__bioRows) window.__bioRows = [];
            window.__bioRows.push(rows);
        }
    }
    return window.__bioRows || [];
}
"""


def slugify(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-zA-Z0-9\s-]", "", n).strip().lower()
    return re.sub(r"\s+", "-", n)


def parse_hometown(hometown: str) -> str | None:
    """Country is conventionally the text after the last comma, e.g.
    'Rochester, United States' -> 'United States'. Some fighters have a
    country-only hometown with no comma (e.g. 'United States') -- treat the
    whole string as the country in that case rather than discarding it."""
    if not hometown:
        return None
    raw = hometown.rsplit(",", 1)[-1].strip() if "," in hometown else hometown.strip()
    return normalize_country(raw)


def fetch_fighter_bio(page, name: str) -> dict:
    slug = slugify(name)
    url = f"https://www.ufc.com/athlete/{slug}"
    try:
        page.goto(url, timeout=30000, wait_until="networkidle")
    except Exception as e:
        log.warning("Failed to load %s (%s): %s", name, url, e)
        return {"slug": slug, "hometown": None, "country": None, "found": False}

    rows = page.evaluate(_EXTRACT_JS)
    hometown = None
    for row in rows:
        if _HOMETOWN_LABEL_RE.search(row["label"]):
            hometown = row["text"]
            break

    return {
        "slug": slug,
        "hometown": hometown,
        "country": parse_hometown(hometown) if hometown else None,
        "found": hometown is not None,
    }


def load_fighters() -> pd.DataFrame:
    conn = sqlite3.connect(DB_UFCSTATS_PATH)
    df = pd.read_sql("SELECT fighter_id, name FROM fighters ORDER BY name", conn)
    conn.close()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape fighter hometown/country from ufc.com athlete pages.")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N fighters (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to disk")
    args = parser.parse_args()

    fighters = load_fighters()
    if args.limit:
        fighters = fighters.head(args.limit)

    already_done = set()
    existing_rows = []
    if OUT_CSV.exists() and not args.dry_run:
        existing = pd.read_csv(OUT_CSV)
        already_done = set(existing["fighter_id"])
        existing_rows = existing.to_dict("records")
        log.info("Resuming: %d fighters already scraped.", len(already_done))

    todo = fighters[~fighters["fighter_id"].isin(already_done)]
    log.info("Scraping %d fighters (of %d total)...", len(todo), len(fighters))

    results = list(existing_rows)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for i, row in enumerate(todo.itertuples(index=False), start=1):
            bio = fetch_fighter_bio(page, row.name)
            results.append({
                "fighter_id": row.fighter_id,
                "name": row.name,
                "slug": bio["slug"],
                "hometown": bio["hometown"],
                "country": bio["country"],
                "found": bio["found"],
            })
            if i % 10 == 0 or i == len(todo):
                log.info("  [%d/%d] %s -> %s", i, len(todo), row.name, bio["country"])

            if not args.dry_run and i % CHECKPOINT_EVERY == 0:
                pd.DataFrame.from_records(results).to_csv(OUT_CSV, index=False)

            time.sleep(REQUEST_DELAY_SEC)

        browser.close()

    out_df = pd.DataFrame.from_records(results)
    if args.dry_run:
        text = out_df.to_string(index=False)
        sys.stdout.buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.write("\n")
        return

    out_df.to_csv(OUT_CSV, index=False)
    found = out_df["found"].sum()
    print(f"Scraped {len(out_df)} fighters -- {found} with a resolved country, {len(out_df) - found} unresolved.")
    print(f"Written to {OUT_CSV}")


if __name__ == "__main__":
    main()
