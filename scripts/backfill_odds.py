"""
backfill_odds.py -- Backfill moneyline odds from bestfightodds.com into the UFCStats DB
and ufc-master.csv for historical events that are missing odds.

Usage:
    python scripts/backfill_odds.py [--dry-run]
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import difflib
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH
from utils.logger import get_logger

log = get_logger(__name__)

BASE = "https://www.bestfightodds.com"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ufc-predictor-scraper/1.0)"}
_DELAY = 1.5

# Verified BFO event URLs for each target date
BFO_EVENT_URLS: dict[str, str] = {
    "2026-04-04": f"{BASE}/events/ufc-vegas-115-4107",
    "2026-04-11": f"{BASE}/events/ufc-327-4074",
    "2026-04-18": f"{BASE}/events/ufc-winnipeg-4130",
    "2026-04-25": f"{BASE}/events/ufc-vegas-116-4147",
    "2026-05-02": f"{BASE}/events/ufc-perth-4079",
    "2026-05-09": f"{BASE}/events/ufc-328-4165",
    "2026-05-16": f"{BASE}/events/ufc-vegas-117-4186",
    "2026-05-30": f"{BASE}/events/ufc-macau-4188",
    "2026-06-06": f"{BASE}/events/ufc-vegas-118-4200",
    "2026-06-14": f"{BASE}/events/ufc-freedom-fights-250-4082",
}

# Extra BFO pages that cover prelims/undercard for split events
BFO_EXTRA_URLS: dict[str, list[str]] = {
    "2026-04-18": [f"{BASE}/events/ufc-winnipeg-4145"],
    "2026-05-16": [f"{BASE}/events/ufc-vegas-117-4178"],
}

_SKIP_WORDS = (
    "round", "decision", "draw", "sub", "tko", "inside",
    "distance", "wins", "start", "goes", "doesn",
)
_SKIP_BOOKS = {"polymarket", "kalshi", "props"}


def _name_key(s: str) -> str:
    return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()


def _best_match(target: str, candidates: list[str], cutoff: float = 0.75) -> str | None:
    key = _name_key(target)
    keys = [_name_key(c) for c in candidates]
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if not matches:
        return None
    return candidates[keys.index(matches[0])]


def _get(url: str) -> BeautifulSoup | None:
    try:
        time.sleep(_DELAY)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.warning("BFO fetch failed for %s: %s", url, exc)
        return None


def _parse_bfo_page(url: str) -> list[dict]:
    """Parse a BFO event page. Returns [{r_name, b_name, odds_r, odds_b}, ...]."""
    soup = _get(url)
    if soup is None:
        return []
    tables = soup.find_all("table")
    if len(tables) < 2:
        log.warning("No odds table found at %s", url)
        return []

    t1 = tables[1]
    header_cells = t1.find("tr").find_all(["td", "th"])
    header_names = [c.get_text(strip=True).lower() for c in header_cells]

    # Identify mainstream sportsbook columns (skip prediction markets + props)
    use_cols = []
    for i, h in enumerate(header_names):
        if i == 0:
            continue
        clean_h = re.sub(r"\$.*", "", h).strip()
        if not clean_h:
            continue
        if any(k in clean_h for k in _SKIP_BOOKS if k):
            continue
        use_cols.append(i)

    def best_odds(cells_list: list) -> int | None:
        for ci in use_cols:
            if ci >= len(cells_list):
                continue
            txt = re.sub(r"[^\d+-]", "", cells_list[ci].get_text(strip=True))
            if txt and re.match(r"^[+-]?\d+$", txt):
                val = int(txt)
                if abs(val) >= 100:
                    return val
        return None

    rows = t1.find_all("tr")
    matchups = []
    i = 1  # skip header
    while i < len(rows):
        row = rows[i]
        if "pr" in row.get("class", []):
            i += 1
            continue
        cells = row.find_all(["td", "th"])
        if not cells:
            i += 1
            continue

        name_a = re.sub(r"^\d+", "", cells[0].get_text(strip=True)).strip()
        if not name_a or any(w in name_a.lower() for w in _SKIP_WORDS):
            i += 1
            continue

        odds_a = best_odds(cells)

        # Find fighter B (next non-pr row)
        j = i + 1
        while j < len(rows) and "pr" in rows[j].get("class", []):
            j += 1
        if j >= len(rows):
            break

        cells_b = rows[j].find_all(["td", "th"])
        if not cells_b:
            i = j + 1
            continue

        name_b = re.sub(r"^\d+", "", cells_b[0].get_text(strip=True)).strip()
        if not name_b or any(w in name_b.lower() for w in _SKIP_WORDS):
            i = j + 1
            continue

        odds_b = best_odds(cells_b)
        matchups.append({
            "bfo_a": name_a,
            "bfo_b": name_b,
            "odds_a": odds_a,
            "odds_b": odds_b,
        })
        i = j + 1

    return matchups


def _scrape_all_matchups(date: str) -> list[dict]:
    """Scrape all BFO matchups for a date (combining main + extra pages if needed)."""
    urls = [BFO_EVENT_URLS[date]] + BFO_EXTRA_URLS.get(date, [])
    all_matchups = []
    seen_pairs = set()
    for url in urls:
        log.info("Fetching BFO page: %s", url)
        matchups = _parse_bfo_page(url)
        for m in matchups:
            key = (_name_key(m["bfo_a"]), _name_key(m["bfo_b"]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_matchups.append(m)
    return all_matchups


def backfill_odds(dry_run: bool = False) -> None:
    conn = sqlite3.connect(str(DB_UFCSTATS_PATH))

    # Track all updates for CSV backfill
    csv_updates: list[dict] = []

    total_updated = 0

    for date, bfo_url in sorted(BFO_EVENT_URLS.items()):
        # Get DB fights for this date that are missing odds
        fights = conn.execute(
            """
            SELECT f.fight_id, r.name AS r_name, b.name AS b_name
            FROM fights f
            JOIN fighters r ON f.r_fighter_id = r.fighter_id
            JOIN fighters b ON f.b_fighter_id = b.fighter_id
            WHERE f.date = ? AND f.odds_red IS NULL
            """,
            (date,),
        ).fetchall()

        if not fights:
            log.info("%s: no fights with missing odds, skipping", date)
            continue

        log.info("%s: %d fights need odds", date, len(fights))

        bfo_matchups = _scrape_all_matchups(date)
        if not bfo_matchups:
            log.warning("%s: no matchups scraped from BFO", date)
            continue

        # Build lookup: bfo_name_key -> (odds_a, odds_b, bfo_a, bfo_b)
        bfo_by_name: dict[str, dict] = {}
        for m in bfo_matchups:
            bfo_by_name[_name_key(m["bfo_a"])] = m
            bfo_by_name[_name_key(m["bfo_b"])] = m

        bfo_all_names = list(bfo_by_name.keys())

        updated = 0
        for fight_id, r_name, b_name in fights:
            r_key = _name_key(r_name)
            b_key = _name_key(b_name)

            # Try exact match first
            match = bfo_by_name.get(r_key) or bfo_by_name.get(b_key)

            # Fuzzy fallback
            if match is None:
                fuzzy = _best_match(r_name, bfo_all_names)
                if fuzzy:
                    match = bfo_by_name[_name_key(fuzzy)]
            if match is None:
                fuzzy = _best_match(b_name, bfo_all_names)
                if fuzzy:
                    match = bfo_by_name[_name_key(fuzzy)]

            if match is None:
                log.warning("  No BFO match for: %s vs %s", r_name, b_name)
                continue

            # Determine which BFO position (A or B) matches red/blue
            bfo_a_key = _name_key(match["bfo_a"])
            bfo_b_key = _name_key(match["bfo_b"])

            # Check if red fighter matches BFO position A or B
            red_is_bfo_a = (
                r_key == bfo_a_key
                or difflib.SequenceMatcher(None, r_key, bfo_a_key).ratio() > 0.8
            )

            if red_is_bfo_a:
                odds_red = match["odds_a"]
                odds_blue = match["odds_b"]
            else:
                odds_red = match["odds_b"]
                odds_blue = match["odds_a"]

            log.info(
                "  %s vs %s -> odds_red=%s odds_blue=%s (BFO: %s / %s)",
                r_name, b_name, odds_red, odds_blue, match["bfo_a"], match["bfo_b"],
            )

            if not dry_run:
                conn.execute(
                    "UPDATE fights SET odds_red=?, odds_blue=? WHERE fight_id=?",
                    (odds_red, odds_blue, fight_id),
                )

            csv_updates.append({
                "date": date,
                "r_name": r_name,
                "b_name": b_name,
                "odds_red": odds_red,
                "odds_blue": odds_blue,
            })
            updated += 1

        if not dry_run:
            conn.commit()
        log.info("  %s: updated %d fights", date, updated)
        total_updated += updated

    conn.close()
    log.info("Total DB fights updated: %d", total_updated)

    # Update ufc-master.csv
    csv_path = ROOT_DIR / "raw_data" / "ufc-master.csv"
    if not csv_path.exists():
        log.warning("ufc-master.csv not found, skipping CSV update")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    # Ensure odds columns exist
    if "R_odds" not in df.columns:
        df["R_odds"] = None
    if "B_odds" not in df.columns:
        df["B_odds"] = None

    name_key_fn = _name_key
    csv_updated = 0
    for upd in csv_updates:
        if upd["odds_red"] is None and upd["odds_blue"] is None:
            continue
        mask = (
            (df["date"].astype(str) == upd["date"])
            & (
                df["R_fighter"].apply(name_key_fn) == name_key_fn(upd["r_name"])
            )
        )
        if mask.sum() == 0:
            # Try fuzzy
            date_rows = df[df["date"].astype(str) == upd["date"]]
            if not date_rows.empty:
                for idx, row in date_rows.iterrows():
                    if (
                        difflib.SequenceMatcher(
                            None, name_key_fn(str(row.get("R_fighter", ""))), name_key_fn(upd["r_name"])
                        ).ratio() > 0.8
                    ):
                        mask = df.index == idx
                        break
        if mask.sum() > 0 and not dry_run:
            df.loc[mask, "R_odds"] = upd["odds_red"]
            df.loc[mask, "B_odds"] = upd["odds_blue"]
            csv_updated += 1

    if not dry_run:
        df.to_csv(csv_path, index=False)
        log.info("CSV updated: %d rows written to %s", csv_updated, csv_path)
    else:
        log.info("[dry-run] Would update %d CSV rows", csv_updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    backfill_odds(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
