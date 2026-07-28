"""
scrapers/ufc_rankings_web.py -- Live scrape of the current UFC.com rankings
snapshot, appended to raw_data/rankings_history.csv.

ufc.com/rankings geo-redirects to a localized mirror (e.g. ufcespanol.com) for
non-US network egress; the fighter names and ranks are unaffected by locale,
but the division header text is translated, so DIVISION_TRANSLATIONS maps the
handful of localized headers back to the canonical English weightclass names
already used in rankings_history.csv. Fighter names and ranks are read
directly from the DOM (Playwright), not parsed from raw HTML text, so
locale-driven wording changes elsewhere on the page don't affect them.

Usage:
    python scrapers/ufc_rankings_web.py
    python scrapers/ufc_rankings_web.py --dry-run
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.logger import get_logger

log = get_logger(__name__)

RANKINGS_URL = "https://www.ufc.com/rankings"
RANKINGS_CSV = ROOT_DIR / "raw_data" / "rankings_history.csv"

# Localized division header -> canonical weightclass name used in the CSV.
# ufc.com's rankings view has been observed rendering in Spanish (via the
# ufcespanol.com geo-redirect); "Women's Flyweight" render in English on that
# same page for unclear (possibly a missing translation string) reasons.
DIVISION_TRANSLATIONS = {
    "Men's Pound-for-Pound Top Rank":   "Men's Pound-for-Pound",
    "Women's Pound-for-Pound Top Rank": "Women's Pound-for-Pound",
    "Peso mosca":          "Flyweight",
    "Peso gallo":          "Bantamweight",
    "Peso pluma":          "Featherweight",
    "Ligero":              "Lightweight",
    "Peso welter":         "Welterweight",
    "Peso medio":          "Middleweight",
    "Peso semipesado":     "Light Heavyweight",
    "De peso pesado":      "Heavyweight",
    "Peso de la mujer":    "Women's Strawweight",
    "Women's Flyweight":   "Women's Flyweight",
    "Gallo de las mujeres": "Women's Bantamweight",
    "Peso pluma de la mujer": "Women's Featherweight",
    # Already-English fallbacks, in case the page ever serves English directly.
    "Flyweight": "Flyweight", "Bantamweight": "Bantamweight",
    "Featherweight": "Featherweight", "Lightweight": "Lightweight",
    "Welterweight": "Welterweight", "Middleweight": "Middleweight",
    "Light Heavyweight": "Light Heavyweight", "Heavyweight": "Heavyweight",
    "Women's Strawweight": "Women's Strawweight",
    "Women's Bantamweight": "Women's Bantamweight",
    "Women's Featherweight": "Women's Featherweight",
    "Men's Pound-for-Pound": "Men's Pound-for-Pound",
    "Women's Pound-for-Pound": "Women's Pound-for-Pound",
}

_EXTRACT_JS = """
() => {
    const block = document.querySelector('.block-views-blockathlete-rankings-block-1');
    if (!block) return null;
    const groupings = Array.from(block.querySelectorAll(':scope > div > .view-content > .view-grouping'));
    return groupings.map(g => {
        const header = g.querySelector('.view-grouping-header');
        const champLink = g.querySelector(
            '.rankings--athlete--champion h5 a, .rankings--athlete--champion .views-field-title a, .rankings--athlete--champion a');
        const rows = Array.from(g.querySelectorAll('.view-grouping-content table tbody tr'));
        return {
            division: header ? header.textContent.trim() : null,
            champion: champLink ? champLink.textContent.trim() : null,
            rows: rows.map(r => Array.from(r.querySelectorAll('td')).map(c => c.textContent.trim())),
        };
    });
}
"""


def scrape_current_rankings() -> pd.DataFrame:
    """Return a DataFrame with columns [date, weightclass, fighter, rank] for
    today's live UFC.com rankings snapshot. Champion = rank 0 (divisions only;
    pound-for-pound has no champion, so its ranks start at 1)."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        log.info("Loading %s ...", RANKINGS_URL)
        page.goto(RANKINGS_URL, timeout=30000, wait_until="networkidle")
        groupings = page.evaluate(_EXTRACT_JS)
        browser.close()

    if not groupings:
        raise RuntimeError("Could not find the rankings block on the page -- site structure may have changed.")

    today = date.today().isoformat()
    records = []
    for g in groupings:
        raw_division = g["division"]
        if raw_division not in DIVISION_TRANSLATIONS:
            log.warning("Unrecognized division header %r -- skipping this group.", raw_division)
            continue
        division = DIVISION_TRANSLATIONS[raw_division]
        is_p4p = "Pound-for-Pound" in division

        if not is_p4p and g["champion"]:
            records.append({"date": today, "weightclass": division, "fighter": g["champion"], "rank": 0})

        for row in g["rows"]:
            if len(row) < 2:
                continue
            rank_str, fighter = row[0], row[1]
            if not rank_str.isdigit():
                continue
            records.append({"date": today, "weightclass": division, "fighter": fighter, "rank": int(rank_str)})

    df = pd.DataFrame.from_records(records)
    return df.drop_duplicates(subset=["weightclass", "fighter", "rank"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the current UFC.com rankings snapshot.")
    parser.add_argument("--dry-run", action="store_true", help="Print the scraped snapshot without writing to disk")
    args = parser.parse_args()

    new_snapshot = scrape_current_rankings()
    today = new_snapshot["date"].iloc[0]
    print(f"Scraped {len(new_snapshot)} ranked fighters across "
          f"{new_snapshot['weightclass'].nunique()} groups, dated {today}.")

    if args.dry_run:
        # Some fighter names include non-cp1252 characters (e.g. accented
        # letters); the Windows console can't display them, so degrade
        # gracefully instead of crashing the dry-run print.
        text = new_snapshot.sort_values(["weightclass", "rank"]).to_string(index=False)
        sys.stdout.buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.write("\n")
        return

    existing = pd.read_csv(RANKINGS_CSV, parse_dates=["date"])
    existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")

    if today in existing["date"].values:
        log.info("A snapshot for %s already exists -- replacing it.", today)
        existing = existing[existing["date"] != today]

    combined = pd.concat([existing, new_snapshot], ignore_index=True)
    combined = combined.sort_values(["date", "weightclass", "rank"])
    combined.to_csv(RANKINGS_CSV, index=False)

    print(f"Appended snapshot dated {today} to {RANKINGS_CSV}")
    print(f"CSV now spans {combined['date'].min()} to {combined['date'].max()} ({len(combined)} rows).")


if __name__ == "__main__":
    main()
