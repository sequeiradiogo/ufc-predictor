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


# -- Fight matching ------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _name_key(a), _name_key(b)).ratio()


_ODDS_SIM_FLOOR = 0.75  # per-fighter similarity floor for odds fuzzy fallback


def _fuzzy_odds_lookup(
    red: str, blue: str, matchups: list[dict],
) -> tuple[int | None, int | None] | None:
    """Fuzzy-match (red, blue) against BFO matchups when exact keys don't align."""
    best_score, best = 0.0, None
    for m in matchups:
        s1, s2 = _sim(red, m["r_name"]), _sim(blue, m["b_name"])
        s3, s4 = _sim(blue, m["r_name"]), _sim(red, m["b_name"])
        if s1 >= _ODDS_SIM_FLOOR and s2 >= _ODDS_SIM_FLOOR:
            score, odds = s1 + s2, (m["odds_red"], m["odds_blue"])
        elif s3 >= _ODDS_SIM_FLOOR and s4 >= _ODDS_SIM_FLOOR:
            score, odds = s3 + s4, (m["odds_blue"], m["odds_red"])
        else:
            continue
        if score > best_score:
            best_score, best = score, odds
    return best


def _fetch_listed_events() -> list[dict]:
    """BFO homepage (upcoming) + /archive (~20 most recent completed) events."""
    events = []
    for fetch, what in ((_fetch_bfo_events, "event list"),
                        (_fetch_bfo_archive_events, "archive events")):
        try:
            events += fetch()
        except Exception as exc:
            log.warning("Could not fetch BFO %s: %s", what, exc)
    return events


def odds_for_pairs(
    event_date: date,
    fighter_pairs: list[tuple[str, str]],
    listed_events: list[dict] | None = None,
    label: str | None = None,
) -> dict[tuple[str, str], tuple[int | None, int | None]]:
    """
    Closing odds for each (red, blue) pair fought on *event_date*.

    Returns {(red, blue): (odds_red, odds_blue)} oriented to *our* corners --
    BFO lists the two fighters in its own order, so a matchup is looked up
    both ways round and flipped when BFO has Blue first. Fights BFO doesn't
    price map to (None, None); {} when no BFO event matched at all.

    BFO event names are often location-based (e.g. "UFC Oklahoma") rather than
    fighter-based like UFCStats, so events are matched by date (+/-1 day) and
    confirmed by their card actually containing our fighters. The archive only
    covers the ~20 most recent events (all orgs), so any fight still uncovered
    is looked up through the fighter's own BFO page. *listed_events* lets a
    caller covering several dates fetch the homepage + archive only once.
    """
    if listed_events is None:
        listed_events = _fetch_listed_events()

    def _on_date(events: list[dict]) -> list[dict]:
        out = []
        for e in events:
            parsed = _parse_bfo_date(e["date_text"], ref_year=event_date.year)
            if parsed and abs((parsed - event_date).days) <= 1:
                out.append(e)
        return out

    fighter_keys = {_name_key(n) for pair in fighter_pairs for n in pair}
    matched_events, matchups = [], []
    seen_urls: set[str] = set()

    def _try_event(ev: dict) -> None:
        # Scrape a candidate and keep it only if its card contains our
        # fighters -- BFO covers many orgs on the same date. BFO also splits
        # a single UFC card across several listings under different names
        # (e.g. "UFC 332" for the main card + a bare "UFC" page for the
        # prelims), so every matching candidate is kept, not just the first.
        if ev["url"] in seen_urls:
            return
        seen_urls.add(ev["url"])
        try:
            ev_matchups = _scrape_event_odds(ev["url"])
        except Exception as exc:
            log.warning("BFO odds scrape failed for '%s': %s", ev["name"], exc)
            return
        ev_keys = {_name_key(m["r_name"]) for m in ev_matchups} | \
                  {_name_key(m["b_name"]) for m in ev_matchups}
        if ev_keys & fighter_keys:
            matched_events.append(ev)
            matchups.extend(ev_matchups)

    for cand in _on_date(listed_events):
        _try_event(cand)

    # Fights still uncovered (event older than the archive window, or a card
    # listing the archive missed): find the event via the fighter's BFO page.
    for red, blue in fighter_pairs:
        if _fuzzy_odds_lookup(red, blue, matchups) is not None:
            continue
        for name in (red, blue):
            try:
                fighter_events = _fetch_fighter_events(name)
            except Exception as exc:
                log.warning("BFO fighter lookup failed for '%s': %s", name, exc)
                continue
            for ev in _on_date(fighter_events):
                _try_event(ev)
            if _fuzzy_odds_lookup(red, blue, matchups) is not None:
                break

    if not matched_events:
        log.warning("No BFO event matched '%s' (%s)", label or "", event_date)
        return {}

    log.info("BFO event(s) matched: %s", ", ".join(e["name"] for e in matched_events))

    bfo_lookup: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for m in matchups:
        rk = _name_key(m["r_name"])
        bk = _name_key(m["b_name"])
        bfo_lookup[(rk, bk)] = (m["odds_red"], m["odds_blue"])
        bfo_lookup[(bk, rk)] = (m["odds_blue"], m["odds_red"])

    out: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for red, blue in fighter_pairs:
        odds = bfo_lookup.get((_name_key(red), _name_key(blue)))
        if odds is None:
            # Exact key match failed -- fuzzy-match names (hyphenation like
            # "Saint-Denis" vs "Saint Denis", suffixes like "Kai Kamaka III",
            # nicknames like "Zach" vs "Zachary" all break the exact lookup).
            odds = _fuzzy_odds_lookup(red, blue, matchups)
        out[(red, blue)] = odds if odds is not None else (None, None)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_odds(fights: list[dict]) -> dict[str, tuple[int | None, int | None]]:
    """
    Look up moneyline odds on bestfightodds.com for each fight in *fights*.

    *fights* is the list of fight dicts returned by scrapers.ufcstats.scrape_new_data().
    Each dict must have keys: fight_id, date, r_name, b_name.

    Returns {fight_id: (odds_red, odds_blue)}.
    Missing or unmatched fights map to (None, None).

    Each date's fights go through odds_for_pairs() as a whole pair, so both
    fighters must match and the odds follow our corners. Until 2026-11 this
    matched a BFO matchup on either fighter's name across every date in the
    batch and took BFO's row order as Red/Blue: Oct 2026 had 10 of 26 checked
    fights with the corners swapped, and a fighter booked twice in one batch
    (Jacobe Smith, Oct 3 + Oct 31) got the other fight's line.
    """
    if not fights:
        return {}

    result: dict[str, tuple[int | None, int | None]] = {
        f["fight_id"]: (None, None) for f in fights
    }

    by_date: dict[date, list[dict]] = {}
    for f in fights:
        if not (f.get("r_name") and f.get("b_name")):
            continue
        try:
            by_date.setdefault(date.fromisoformat(str(f.get("date", ""))[:10]), []).append(f)
        except ValueError:
            continue

    listed_events = _fetch_listed_events() if by_date else []
    for event_date, day_fights in sorted(by_date.items()):
        odds = odds_for_pairs(event_date, [(f["r_name"], f["b_name"]) for f in day_fights],
                              listed_events)
        for f in day_fights:
            result[f["fight_id"]] = odds.get((f["r_name"], f["b_name"]), (None, None))

    matched = sum(1 for v in result.values() if v != (None, None))
    log.info("BFO odds: matched %d / %d fights", matched, len(fights))
    return result
