"""
bestfightodds.py -- Scrape American moneyline odds from bestfightodds.com.

Usage (called from refresh_data.py):
    from scrapers.bestfightodds import scrape_odds
    odds_map = scrape_odds(fights)   # {fight_id: (odds_red, odds_blue)}

Non-blocking: any scraping failure returns (None, None) for that fight.
The odds columns in the DB are nullable so missing data is fine.
"""

import difflib
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.logger import get_logger

log = get_logger(__name__)

BASE    = "https://www.bestfightodds.com"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ufc-predictor-scraper/1.0)"}
_DELAY  = 1.5


def _get(url: str) -> BeautifulSoup | None:
    try:
        time.sleep(_DELAY)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.warning("BFO fetch failed for %s: %s", url, exc)
        return None


def _parse_american_odds(text: str) -> int | None:
    """Parse an American moneyline string like '-150' or '+130' to int."""
    text = text.strip().replace("−", "-")  # unicode minus sign
    m = re.match(r"([+-]?\d+)", text)
    return int(m.group(1)) if m else None


def _name_key(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def _best_match(target: str, candidates: list[str], cutoff: float = 0.80) -> str | None:
    key = _name_key(target)
    keys = [_name_key(c) for c in candidates]
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if not matches:
        return None
    return candidates[keys.index(matches[0])]


# ── BFO event list ────────────────────────────────────────────────────────────

def _fetch_bfo_events() -> list[dict]:
    """Return list of {name, url, date_text} from the BFO homepage."""
    soup = _get(f"{BASE}/")
    if soup is None:
        return []
    events = []
    for a in soup.select("table.event-table a[href*='/events/']"):
        href = a["href"].strip()
        if not href.startswith("/events/"):
            continue
        name = a.get_text(strip=True)
        url  = BASE + href if href.startswith("/") else href
        # Date is often a nearby sibling <td>
        td   = a.find_parent("td")
        row  = td.find_parent("tr") if td else None
        date_td = row.find("td", class_=re.compile(r"date")) if row else None
        date_text = date_td.get_text(strip=True) if date_td else ""
        events.append({"name": name, "url": url, "date_text": date_text})
    return events


# ── BFO event page ────────────────────────────────────────────────────────────

def _scrape_event_odds(event_url: str) -> list[dict]:
    """
    Scrape one BFO event page.

    Returns list of {r_name, b_name, odds_red, odds_blue} dicts.
    Uses the closing moneyline (last column) if available, else opening.
    """
    soup = _get(event_url)
    if soup is None:
        return []
    matchups = []
    for row in soup.select("tr.table-header + tr, tr.odd, tr.even"):
        fighter_cells = row.select("td.fighter-name a")
        if len(fighter_cells) < 2:
            continue
        r_name = fighter_cells[0].get_text(strip=True)
        b_name = fighter_cells[1].get_text(strip=True)

        # Odds cells — grab all odds values, take the last pair (closing line)
        odds_cells = row.select("td.moneyline span.best-odds-td, td.moneyline")
        raw_vals = [c.get_text(strip=True) for c in odds_cells if c.get_text(strip=True)]
        if len(raw_vals) >= 2:
            odds_red  = _parse_american_odds(raw_vals[0])
            odds_blue = _parse_american_odds(raw_vals[1])
        else:
            odds_red = odds_blue = None

        if r_name or b_name:
            matchups.append({
                "r_name":    r_name,
                "b_name":    b_name,
                "odds_red":  odds_red,
                "odds_blue": odds_blue,
            })
    return matchups


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_odds(fights: list[dict]) -> dict[str, tuple[int | None, int | None]]:
    """
    Look up moneyline odds on bestfightodds.com for each fight in *fights*.

    *fights* is the list of fight dicts returned by scrapers.ufcstats.scrape_new_data().
    Each dict must have keys: fight_id, r_fighter_id, b_fighter_id, and optionally
    r_name / b_name (used for matching).

    Returns {fight_id: (odds_red, odds_blue)}.
    Missing or unmatched fights map to (None, None).
    """
    if not fights:
        return {}

    result: dict[str, tuple[int | None, int | None]] = {
        f["fight_id"]: (None, None) for f in fights
    }

    try:
        bfo_events = _fetch_bfo_events()
    except Exception as exc:
        log.warning("Could not fetch BFO event list: %s", exc)
        return result

    if not bfo_events:
        log.warning("BFO event list was empty — skipping odds scrape")
        return result

    # Build a name->fight_id lookup from the fights list
    # fights should carry r_name / b_name if available
    name_to_fight: dict[str, str] = {}
    for f in fights:
        for key in ("r_name", "b_name"):
            if f.get(key):
                name_to_fight[_name_key(f[key])] = f["fight_id"]

    # Group fights by event date so we only scrape relevant BFO event pages
    dates_needed: set[str] = {f.get("date", "")[:10] for f in fights}
    bfo_event_names = [e["name"] for e in bfo_events]

    for event_date in dates_needed:
        # Try to find the BFO event matching this date
        # BFO event names contain "UFC NNN" or "UFC Fight Night" style strings
        # Match any BFO event whose date_text contains the same date fragment
        matching = [
            e for e in bfo_events
            if event_date.replace("-", "/") in e["date_text"]
            or event_date.replace("-", " ") in e["date_text"]
        ]
        if not matching:
            # Fallback: match by UFC event name pattern
            ufc_fights = [f for f in fights if f.get("date", "")[:10] == event_date]
            continue

        for bfo_event in matching:
            matchups = _scrape_event_odds(bfo_event["url"])
            for mu in matchups:
                # Match by fighter name
                r_key = _name_key(mu["r_name"])
                b_key = _name_key(mu["b_name"])
                fight_id = name_to_fight.get(r_key) or name_to_fight.get(b_key)
                if not fight_id:
                    # Try fuzzy match
                    matched = _best_match(
                        mu["r_name"],
                        [f.get("r_name", "") or "" for f in fights],
                    )
                    if matched:
                        fight_id = name_to_fight.get(_name_key(matched))
                if fight_id:
                    result[fight_id] = (mu["odds_red"], mu["odds_blue"])

    matched = sum(1 for v in result.values() if v != (None, None))
    log.info("BFO odds: matched %d / %d fights", matched, len(fights))
    return result
