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
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote_plus

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
    words = re.sub(r"[^a-z ]", "", name.lower()).split()
    # BFO repeats mononyms as first + last name ("Sumudaerji Sumudaerji")
    if len(words) == 2 and words[0] == words[1]:
        words = words[:1]
    return " ".join(words)


def _best_match(target: str, candidates: list[str], cutoff: float = 0.80) -> str | None:
    key = _name_key(target)
    keys = [_name_key(c) for c in candidates]
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if not matches:
        return None
    return candidates[keys.index(matches[0])]


_ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


def _parse_bfo_date(text: str, ref_year: int | None = None) -> date | None:
    """
    Parse a BFO date string into a date.

    Homepage dates omit the year (e.g. 'July 25th'); archive dates include it
    (e.g. 'Jul 18th 2026'). When the year is missing, *ref_year* is used
    (defaults to the current year).
    """
    if not text:
        return None
    cleaned = _ORDINAL_RE.sub(r"\1", text).strip()
    for fmt in ("%b %d %Y", "%B %d %Y", "%b %d", "%B %d"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        year = parsed.year if "%Y" in fmt else (ref_year or date.today().year)
        return date(year, parsed.month, parsed.day)
    return None


# ── BFO event list ────────────────────────────────────────────────────────────

def _parse_bfo_event_divs(soup: BeautifulSoup) -> list[dict]:
    """Parse the `div.table-div` event blocks shared by the homepage and event pages."""
    events = []
    for div in soup.select("div.table-div"):
        header = div.select_one("div.table-header")
        if header is None:
            continue
        a = header.select_one("a[href*='/events/']")
        if a is None:
            continue
        href = a["href"].strip()
        url  = BASE + href if href.startswith("/") else href
        name_tag = a.select_one("h1")
        name = name_tag.get_text(strip=True) if name_tag else a.get_text(strip=True)
        name = re.sub(r"\s+Odds$", "", name).strip()
        date_span = header.select_one("span.table-header-date")
        date_text = date_span.get_text(strip=True) if date_span else ""
        events.append({"name": name, "url": url, "date_text": date_text})
    return events


def _fetch_bfo_events() -> list[dict]:
    """Return list of {name, url, date_text} for upcoming events from the BFO homepage."""
    soup = _get(f"{BASE}/")
    if soup is None:
        return []
    return _parse_bfo_event_divs(soup)


def _fetch_bfo_archive_events() -> list[dict]:
    """Return list of {name, url, date_text} for recently completed events from the BFO archive."""
    soup = _get(f"{BASE}/archive")
    if soup is None:
        return []
    events = []
    for row in soup.select("table.content-list tr"):
        a = row.select_one("td.content-list-title a")
        if a is None:
            continue
        href = a["href"].strip()
        url  = BASE + href if href.startswith("/") else href
        date_td = row.select_one("td.content-list-date")
        events.append({
            "name":      a.get_text(strip=True),
            "url":       url,
            "date_text": date_td.get_text(strip=True) if date_td else "",
        })
    return events


def _parse_fighter_events(soup: BeautifulSoup) -> list[dict]:
    """Parse the {name, url, date_text} event headers from a BFO fighter page."""
    events = []
    for tr in soup.select("table.team-stats-table tr.event-header"):
        a = tr.select_one("a[href*='/events/']")
        if a is None:
            continue
        href = a["href"].strip()
        name = a.get_text(strip=True)
        events.append({
            "name":      name,
            "url":       BASE + href if href.startswith("/") else href,
            "date_text": tr.get_text(" ", strip=True)[len(name):].strip(),
        })
    return events


def _fetch_fighter_events(fighter_name: str) -> list[dict]:
    """
    Return {name, url, date_text} for every BFO event *fighter_name* appears on.

    The archive page only lists the ~20 most recent events (all orgs), so
    older events are reached through a fighter's page instead. BFO's search
    redirects straight to the fighter page on an exact single match;
    otherwise it returns a result list, fuzzy-matched here.
    """
    soup = _get(f"{BASE}/search?query={quote_plus(fighter_name)}")
    if soup is None:
        return []
    if soup.select_one("table.team-stats-table") is None:
        links = {a.get_text(strip=True): a["href"].strip()
                 for a in soup.select("table.content-list a[href^='/fighters/']")}
        match = _best_match(fighter_name, list(links))
        if match is None:
            return []
        soup = _get(BASE + links[match])
        if soup is None:
            return []
    return _parse_fighter_events(soup)


# ── BFO event page ────────────────────────────────────────────────────────────

#: Prediction-market exchanges BFO lists in the same table as sportsbooks
#: (matched against the column header text, lowercased). Their prices are
#: last trades on thin books and drift toward ~0%/100% once a fight is
#: decided -- score_event.py scrapes after the event, so these columns
#: produced "closing odds" like -3487 / +2236 for a -205 / +163 fight.
_EXCHANGE_BOOKS = ("polymarket", "kalshi", "prophetx", "novig", "sporttrade")


def _scrape_event_odds(event_url: str) -> list[dict]:
    """
    Scrape one BFO event page.

    Returns list of {r_name, b_name, odds_red, odds_blue} dicts. Both sides'
    odds come from the same sportsbook -- the first one in BFO's column order
    that prices both fighters -- with prediction-market exchanges skipped.
    Fights only an exchange prices get (None, None).
    """
    soup = _get(event_url)
    if soup is None:
        return []

    matchups: list[dict] = []
    for scroller in soup.select("div.table-scroller"):
        table = scroller.select_one("table.odds-table")
        if table is None:
            continue

        # Sportsbook ids in column order. Header <th data-b="21"> and row
        # <td data-li="[21,1,45055]"> share the book id, so odds are keyed by
        # book rather than by position (empty cells would otherwise shift
        # columns and pair different books' prices).
        book_order = [
            th["data-b"] for th in table.select("thead th[data-b]")
            if not any(x in th.get_text(strip=True).lower() for x in _EXCHANGE_BOOKS)
        ]
        if not book_order:
            log.warning("BFO odds table on %s has no sportsbook headers -- markup changed?",
                        event_url)

        # Taking each side's best price independently mixes books and produces
        # incoherent pairs, so both sides come from the same book.
        fighter_rows: list[tuple[str, dict[str, int | None]]] = []
        for row in table.select("tbody tr"):
            name_span = row.select_one("th span.t-b-fcc")
            if name_span is None:
                continue  # prop-bet row, not a fighter row
            name = name_span.get_text(strip=True)
            book_odds: dict[str, int | None] = {}
            for td in row.select("td.but-sg[data-li]"):
                span = td.select_one("span[id]")
                if span is None:
                    continue
                book_id = td["data-li"].strip("[]").split(",")[0].strip()
                book_odds[book_id] = _parse_american_odds(span.get_text(strip=True))
            fighter_rows.append((name, book_odds))

        for i in range(0, len(fighter_rows) - 1, 2):
            r_name, r_odds = fighter_rows[i]
            b_name, b_odds = fighter_rows[i + 1]
            odds_red = odds_blue = None
            for book_id in book_order:
                r_val, b_val = r_odds.get(book_id), b_odds.get(book_id)
                if r_val is not None and b_val is not None:
                    odds_red, odds_blue = r_val, b_val
                    break
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
        bfo_events = []

    try:
        archive_events = _fetch_bfo_archive_events()
    except Exception as exc:
        log.warning("Could not fetch BFO archive events: %s", exc)
        archive_events = []

    all_events = bfo_events + archive_events
    if not all_events:
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
    dates_needed: set[str] = {f.get("date", "")[:10] for f in fights if f.get("date")}

    for event_date_str in dates_needed:
        try:
            target_date = date.fromisoformat(event_date_str)
        except ValueError:
            continue

        # Match BFO events whose parsed date is within a day of the fight date
        # (BFO sometimes lists events under the local venue date).
        matching = []
        for e in all_events:
            parsed = _parse_bfo_date(e["date_text"], ref_year=target_date.year)
            if parsed and abs((parsed - target_date).days) <= 1:
                matching.append(e)
        if not matching:
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
