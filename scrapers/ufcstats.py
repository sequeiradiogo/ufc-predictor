"""
ufcstats.py -- Scrape fight results, per-fight stats, and fighter bios from ufcstats.com.

Uses Playwright (headless Chromium) to bypass the Cloudflare JS challenge.
First-time setup: run `python -m playwright install chromium` once.

Returns data structured for direct insertion into the project SQLite DB:
  {
    "fighters":    [{fighter_id, name, height, reach, stance, dob}, ...],
    "fights":      [{fight_id, event_id, date, division, r_fighter_id,
                     b_fighter_id, winner_id, method, title_fight,
                     odds_red, odds_blue}, ...],
    "fight_stats": [{fight_id, fighter_id, corner, kd, sig_str_landed, ...}, ...],
  }

Fighter/fight/event IDs are the hex strings from ufcstats.com URL path segments,
matching the convention already used in the DB.
"""

import re
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.logger import get_logger

log = get_logger(__name__)

BASE   = "http://www.ufcstats.com"
_DELAY = 1.0  # seconds between page navigations


# ── Playwright session ────────────────────────────────────────────────────────

@contextmanager
def _browser_session():
    """Yield a callable get(url) -> BeautifulSoup backed by a single Playwright page."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
        })

        def get(url: str) -> BeautifulSoup:
            time.sleep(_DELAY)
            page.goto(url, wait_until="networkidle", timeout=30_000)
            return BeautifulSoup(page.content(), "lxml")

        try:
            yield get
        finally:
            browser.close()


# ── String / value parsers ────────────────────────────────────────────────────

def _parse_of(text: str) -> tuple[int, int]:
    """'X of Y' -> (X, Y). Returns (0, 0) on failure."""
    m = re.match(r"(\d+)\s+of\s+(\d+)", text.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _ctrl_to_seconds(text: str) -> int:
    """'M:SS' control time -> integer seconds."""
    m = re.match(r"(\d+):(\d{2})", text.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _fight_seconds(round_num: int, time_str: str) -> int:
    """Total elapsed seconds at finish: (round-1)*300 + minutes*60 + secs."""
    round_num = max(1, round_num)  # guard against unparsed zero
    m = re.match(r"(\d+):(\d{2})", time_str.strip())
    if not m:
        return 0
    return (round_num - 1) * 300 + int(m.group(1)) * 60 + int(m.group(2))


_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def _parse_date(text: str) -> str | None:
    """Parse UFCStats date strings ('May 30, 2026') to ISO 'YYYY-MM-DD'."""
    text = text.replace(".", "").strip().lower()
    m = re.match(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group(1)[:3])
    if not mon:
        return None
    return f"{m.group(3)}-{mon}-{int(m.group(2)):02d}"


def _id_from_url(url: str) -> str:
    """Extract the trailing hex ID from a ufcstats.com URL."""
    return url.rstrip("/").split("/")[-1]


def _normalize_division(text: str) -> str:
    """Map UFCStats weight-class strings to lowercase project format."""
    t = text.lower()
    for div in (
        "women's strawweight", "women's flyweight", "women's bantamweight",
        "women's featherweight", "flyweight", "bantamweight", "featherweight",
        "lightweight", "welterweight", "middleweight", "light heavyweight", "heavyweight",
    ):
        if div in t:
            return div
    return re.sub(r"\s*bout\s*$", "", t).strip()


def _normalize_method(text: str) -> str:
    t = text.strip().lower()
    if "ko" in t or "tko" in t:
        return "KO/TKO"
    if "sub" in t:
        return "Submission"
    if "s-dec" in t or "split" in t:
        return "Decision - Split"
    if "m-dec" in t or "majority" in t:
        return "Decision - Majority"
    if "u-dec" in t or "dec" in t or "unanimous" in t:
        return "Decision - Unanimous"
    return text.strip()


# ── Table-cell helpers ────────────────────────────────────────────────────────
# UFCStats stats tables use ONE <tr> per fight with TWO <p> per <td>
# (p[0] = first/red fighter, p[1] = second/blue fighter).

def _p_text(tds: list, td_idx: int, p_idx: int) -> str:
    if td_idx >= len(tds):
        return ""
    ps = tds[td_idx].find_all("p")
    return ps[p_idx].get_text(strip=True) if p_idx < len(ps) else ""


def _p_int(tds: list, td_idx: int, p_idx: int) -> int:
    t = _p_text(tds, td_idx, p_idx)
    m = re.search(r"\d+", t)
    return int(m.group()) if m else 0


def _p_of(tds: list, td_idx: int, p_idx: int) -> tuple[int, int]:
    return _parse_of(_p_text(tds, td_idx, p_idx))


# ── Height / reach ────────────────────────────────────────────────────────────

def _parse_height_cm(text: str) -> float | None:
    m = re.search(r"(\d+)'\s*(\d+)", text)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    return None


def _parse_reach_cm(text: str) -> float | None:
    m = re.search(r"([\d.]+)", text)
    if m:
        return round(float(m.group(1)) * 2.54, 1)
    return None


# ── Events list ───────────────────────────────────────────────────────────────

def _fetch_completed_events(since: date, get) -> list[dict]:
    """Return events newer than *since* from the completed-events listing."""
    soup = get(f"{BASE}/statistics/events/completed?page=all")
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        a = row.select_one("a.b-link")
        if not a:
            continue
        date_span = row.select_one("span.b-statistics__date")
        if not date_span:
            continue
        iso = _parse_date(date_span.get_text(strip=True))
        if not iso:
            continue
        try:
            event_date = date.fromisoformat(iso)
        except ValueError:
            continue
        if event_date > since:
            event_url = a["href"].strip()
            # Location is in the second td of each event row
            tds = row.find_all("td")
            raw_location = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            parts = [p.strip() for p in raw_location.split(",") if p.strip()]
            country = parts[-1] if parts else ""
            location = ", ".join(parts[:-1]) if len(parts) > 1 else raw_location
            events.append({
                "event_id": _id_from_url(event_url),
                "name":     a.get_text(strip=True),
                "date":     iso,
                "url":      event_url,
                "location": location,
                "country":  country,
            })
    log.info("Found %d new events since %s", len(events), since)
    return events


# ── Event detail ──────────────────────────────────────────────────────────────

def _scrape_event(event: dict, seen_fids: set[str], get) -> dict:
    """Scrape one event page. Returns {fighters, fights, fight_stats}."""
    try:
        soup = get(event["url"])
    except Exception as exc:
        log.warning("Failed to load event %s: %s", event["name"], exc)
        return {"fighters": [], "fights": [], "fight_stats": []}

    fighters: list[dict] = []
    fights:   list[dict] = []
    fight_stats: list[dict] = []

    for row in soup.select("tr.b-fight-details__table-row__hover"):
        fight_url = row.get("data-link", "").strip()
        if not fight_url:
            continue
        fight_id = _id_from_url(fight_url)

        # Fighter links are in column 1
        fighter_links = row.select("a[href*='fighter-details']")
        if len(fighter_links) < 2:
            continue
        r_url = fighter_links[0]["href"].strip()
        b_url = fighter_links[1]["href"].strip()
        r_fid = _id_from_url(r_url)
        b_fid = _id_from_url(b_url)

        # Column layout: [0]win/loss [1]fighters [2]kd [3]sig [4]td [5]sub
        #                [6]division [7]method [8]round [9]time
        tds = row.find_all("td", class_="b-fight-details__table-col")
        division   = _normalize_division(_col_ps_text(tds, 6, 0))
        method     = _normalize_method(_col_ps_text(tds, 7, 0))
        try:
            round_num = int(_col_ps_text(tds, 8, 0) or "0")
        except ValueError:
            round_num = 0
        time_str   = _col_ps_text(tds, 9, 0) or "0:00"
        title_fight = 1 if row.select("img[src*='belt'], img[alt*='belt']") else 0

        fight_data = _scrape_fight(
            fight_id=fight_id, fight_url=fight_url,
            event_id=event["event_id"], event_date=event["date"],
            r_fid=r_fid, b_fid=b_fid,
            division=division, method=method,
            round_num=round_num, time_str=time_str,
            title_fight=title_fight,
            location=event.get("location", ""),
            country=event.get("country", ""),
            get=get,
        )
        if fight_data is None:
            continue
        # Skip fights with no result yet (results page not yet updated)
        if fight_data["fight"]["winner_id"] is None and not fight_data["stats"]:
            log.debug("Skipping fight %s -- no result posted yet", fight_id)
            continue
        fights.append(fight_data["fight"])
        fight_stats.extend(fight_data["stats"])

        for fid, furl, fa in (
            (r_fid, r_url, fighter_links[0]),
            (b_fid, b_url, fighter_links[1]),
        ):
            if fid not in seen_fids:
                fname = fa.get_text(strip=True)
                fighters.append(_scrape_fighter_bio(fid, furl, fname, get))
                seen_fids.add(fid)

    return {"fighters": fighters, "fights": fights, "fight_stats": fight_stats}


def _col_ps_text(tds: list, td_idx: int, p_idx: int) -> str:
    """Get text from p[p_idx] inside tds[td_idx]."""
    if td_idx >= len(tds):
        return ""
    ps = tds[td_idx].find_all("p")
    return ps[p_idx].get_text(strip=True) if p_idx < len(ps) else tds[td_idx].get_text(strip=True)


# ── Fight detail ──────────────────────────────────────────────────────────────

def _scrape_fight(
    fight_id: str, fight_url: str,
    event_id: str, event_date: str,
    r_fid: str, b_fid: str,
    division: str, method: str,
    round_num: int, time_str: str,
    title_fight: int,
    location: str = "",
    country: str = "",
    get=None,
) -> dict | None:
    """Scrape fight detail page. Returns {fight: dict, stats: list[dict]}."""
    try:
        soup = get(fight_url)
    except Exception as exc:
        log.warning("Failed to scrape fight %s: %s", fight_id, exc)
        return None

    # ---- Red/Blue corner assignment and winner ----
    # The fight detail page lists Red corner first, Blue corner second in
    # b-fight-details__person divs. The event listing page lists winner first,
    # so we override r_fid/b_fid here with the actual corner assignment.
    winner_id = None
    person_divs = soup.select("div.b-fight-details__person")
    if len(person_divs) >= 2:
        red_link  = person_divs[0].select_one("a[href*='fighter-details']")
        blue_link = person_divs[1].select_one("a[href*='fighter-details']")
        if red_link and blue_link:
            r_fid = _id_from_url(red_link["href"])
            b_fid = _id_from_url(blue_link["href"])
    for person_div in person_divs:
        status = person_div.select_one("i.b-fight-details__person-status")
        flink  = person_div.select_one("a[href*='fighter-details']")
        if status and flink and "W" in status.get_text():
            winner_id = _id_from_url(flink["href"])
            break
    if winner_id not in (r_fid, b_fid):
        winner_id = None  # draw / no-contest

    total_secs = _fight_seconds(round_num, time_str)

    # ---- finish_details from fight info content block ----
    # Details: is in the second p.b-fight-details__text; text is a direct node in p
    finish_details = ""
    for p in soup.select("p.b-fight-details__text"):
        label = p.find("i", class_="b-fight-details__label")
        if label and "detail" in label.get_text(strip=True).lower():
            finish_details = " ".join(p.get_text(separator=" ").split())
            label_text = label.get_text(strip=True)
            if finish_details.lower().startswith(label_text.lower()):
                finish_details = finish_details[len(label_text):].strip()
            break

    fight = {
        "fight_id":              fight_id,
        "event_id":              event_id,
        "date":                  event_date,
        "location":              location,
        "country":               country,
        "division":              division,
        "r_fighter_id":          r_fid,
        "b_fighter_id":          b_fid,
        "winner_id":             winner_id,
        "method":                method,
        "title_fight":           title_fight,
        "odds_red":              None,
        "odds_blue":             None,
        "total_fight_time_secs": total_secs,
        "finish_round":          round_num,
        "finish_round_time":     time_str,
        "finish_details":        finish_details,
    }

    # ---- Stat sections ----
    # UFCStats fight page section layout (js-fight-section):
    #   0: fight info (no table)
    #   1: Overall totals -- ONE <tbody> with ONE <tr>, each <td> has 2 <p>
    #   2: Per-round totals (multiple <tbody> blocks)
    #   3: empty/decorative
    #   4: Sig-strike breakdown (per-round <tbody> blocks; first tbody often empty)
    sections = soup.select("section.b-fight-details__section.js-fight-section")

    # Overall totals (section 1)
    tds: list = []
    if len(sections) > 1:
        tbody = sections[1].find("tbody", class_="b-fight-details__table-body")
        if tbody:
            row = tbody.find("tr")
            if row:
                tds = row.find_all("td")

    if not tds:
        log.warning("No totals data for fight %s", fight_id)
        return {"fight": fight, "stats": []}

    # Determine fighter order: td[0] p[0] href -> compare to r_fid
    td0_ps = tds[0].find_all("p") if tds else []
    td0_fid0 = ""
    if td0_ps:
        link0 = td0_ps[0].find("a")
        if link0:
            td0_fid0 = _id_from_url(link0["href"])
    # r_idx = 0 if red is listed first, else 1
    r_idx = 0 if (td0_fid0 == r_fid or td0_fid0 == "") else 1
    b_idx = 1 - r_idx

    # Sig-strike totals (sum across per-round tbody blocks in section 4)
    sig: dict[str, list[tuple[int, int]]] = {
        "head":   [(0, 0), (0, 0)],
        "body":   [(0, 0), (0, 0)],
        "leg":    [(0, 0), (0, 0)],
        "dist":   [(0, 0), (0, 0)],
        "clinch": [(0, 0), (0, 0)],
        "ground": [(0, 0), (0, 0)],
    }
    # td indices in sig-strike table: Head=3, Body=4, Leg=5, Distance=6, Clinch=7, Ground=8
    _SIG_COL = {3: "head", 4: "body", 5: "leg", 6: "dist", 7: "clinch", 8: "ground"}

    if len(sections) > 4:
        for tb in sections[4].find_all("tbody"):
            row4 = tb.find("tr")
            if not row4:
                continue
            tds4 = row4.find_all("td")
            for td_idx, stat in _SIG_COL.items():
                for pi in range(2):
                    l, a = _p_of(tds4, td_idx, pi)
                    old_l, old_a = sig[stat][pi]
                    sig[stat][pi] = (old_l + l, old_a + a)

    # ---- Build per-fighter stats ----
    stats = []
    for i, (fid, corner) in enumerate(((r_fid, "r"), (b_fid, "b"))):
        pi = r_idx if corner == "r" else b_idx

        kd                         = _p_int(tds, 1, pi)
        sig_str_l, sig_str_a       = _p_of(tds, 2, pi)
        total_str_l, total_str_a   = _p_of(tds, 4, pi)
        td_l, td_a                 = _p_of(tds, 5, pi)
        sub_att                    = _p_int(tds, 7, pi)
        ctrl                       = _ctrl_to_seconds(_p_text(tds, 9, pi))

        head_l,   head_a   = sig["head"][pi]
        body_l,   body_a   = sig["body"][pi]
        leg_l,    leg_a    = sig["leg"][pi]
        dist_l,   dist_a   = sig["dist"][pi]
        clinch_l, clinch_a = sig["clinch"][pi]
        ground_l, ground_a = sig["ground"][pi]

        stats.append({
            "fight_id":          fight_id,
            "fighter_id":        fid,
            "corner":            corner,
            "kd":                kd,
            "sig_str_landed":    sig_str_l,
            "sig_str_atmpted":   sig_str_a,
            "total_str_landed":  total_str_l,
            "total_str_atmpted": total_str_a,
            "td_landed":         td_l,
            "td_atmpted":        td_a,
            "sub_att":           sub_att,
            "ctrl":              ctrl,
            "head_landed":       head_l,
            "head_atmpted":      head_a,
            "body_landed":       body_l,
            "body_atmpted":      body_a,
            "leg_landed":        leg_l,
            "leg_atmpted":       leg_a,
            "dist_landed":       dist_l,
            "dist_atmpted":      dist_a,
            "clinch_landed":     clinch_l,
            "clinch_atmpted":    clinch_a,
            "ground_landed":     ground_l,
            "ground_atmpted":    ground_a,
            "total_fight_time":  total_secs,
        })

    return {"fight": fight, "stats": stats}


# ── Fighter bio ───────────────────────────────────────────────────────────────

def _scrape_fighter_bio(fighter_id: str, fighter_url: str, name: str, get) -> dict:
    bio = {
        "fighter_id": fighter_id,
        "name":       name,
        "height":     None,
        "reach":      None,
        "stance":     None,
        "dob":        None,
    }
    try:
        soup = get(fighter_url)
        for li in soup.select("li.b-list__box-list-item"):
            text = li.get_text(" ", strip=True)
            key  = text.split(":")[0].strip().lower()
            val  = text.split(":", 1)[-1].strip()
            if not val or val == "--":
                continue
            if key == "height":
                bio["height"] = _parse_height_cm(val)
            elif key == "reach":
                bio["reach"] = _parse_reach_cm(val)
            elif key == "stance":
                bio["stance"] = val
            elif key == "dob":
                bio["dob"] = _parse_date(val)
    except Exception as exc:
        log.warning("Failed to scrape fighter bio %s (%s): %s", name, fighter_id, exc)
    return bio


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_upcoming_event() -> dict | None:
    """
    Scrape the next upcoming event from UFCStats.

    Returns a dict:
        {
            "name":   "UFC 317",
            "date":   "2026-06-07",
            "url":    "http://www.ufcstats.com/event-details/...",
            "fights": [
                {
                    "r_id": "<hex>", "r_name": "Fighter A",
                    "b_id": "<hex>", "b_name": "Fighter B",
                    "division": "lightweight",
                    "title_fight": 0,
                },
                ...
            ],
        }
    Returns None if no upcoming events are found.

    Upcoming event pages list fighters but contain no fight stats, so only
    fighter links and division are extracted -- no visit to individual fight
    detail pages is needed.
    """
    with _browser_session() as get:
        soup = get(f"{BASE}/statistics/events/upcoming?page=all")

        events = []
        for row in soup.select("tr.b-statistics__table-row"):
            a = row.select_one("a.b-link")
            if not a:
                continue
            date_span = row.select_one("span.b-statistics__date")
            if not date_span:
                continue
            iso = _parse_date(date_span.get_text(strip=True))
            if not iso:
                continue
            event_url = a["href"].strip()
            events.append({
                "name":     a.get_text(strip=True),
                "date":     iso,
                "url":      event_url,
                "event_id": _id_from_url(event_url),
            })

        if not events:
            log.warning("No upcoming events found on UFCStats.")
            return None

        # Take the earliest upcoming event
        events.sort(key=lambda e: e["date"])
        event = events[0]
        log.info("Next upcoming event: %s (%s)", event["name"], event["date"])

        soup = get(event["url"])
        fights = []

        # Upcoming events use tr.b-fight-details__table-row (no __hover suffix)
        # Completed events use __hover; try both to be safe.
        rows = soup.select("tr.b-fight-details__table-row")

        for row in rows:
            fighter_links = row.select("a[href*='fighter-details']")
            if len(fighter_links) < 2:
                continue

            r_url  = fighter_links[0]["href"].strip()
            b_url  = fighter_links[1]["href"].strip()
            r_id   = _id_from_url(r_url)
            b_id   = _id_from_url(b_url)
            r_name = fighter_links[0].get_text(strip=True)
            b_name = fighter_links[1].get_text(strip=True)

            if not r_id or not b_id or r_id == b_id:
                continue

            tds        = row.find_all("td", class_="b-fight-details__table-col")
            division   = _normalize_division(_col_ps_text(tds, 6, 0)) if len(tds) > 6 else ""
            title_fight = 1 if row.select("img[src*='belt'], img[alt*='belt']") else 0

            fights.append({
                "r_id":       r_id,
                "r_name":     r_name,
                "b_id":       b_id,
                "b_name":     b_name,
                "division":   division,
                "title_fight": title_fight,
            })

        log.info("Found %d fights on the card.", len(fights))
        event["fights"] = fights
        return event


def scrape_events_iter(
    since: date,
    existing_fighter_ids: set[str] | None = None,
    skip_event_ids: set[str] | None = None,
):
    """
    Generator that scrapes events one at a time and yields (event_meta, data) per event.

    event_meta -- dict with event_id, name, date, url
    data       -- dict with "fighters", "fights", "fight_stats" for that event only

    Use this instead of scrape_new_data() when you want to checkpoint incrementally
    (e.g. write to DB after every N events) so a crash does not lose all progress.

    Pass *skip_event_ids* to resume after a partial run -- events whose event_id is
    already in that set are logged and skipped without an HTTP request.
    """
    seen_fids: set[str] = set(existing_fighter_ids or [])
    skip: set[str] = set(skip_event_ids or [])

    with _browser_session() as get:
        events = _fetch_completed_events(since, get)
        if not events:
            return

        for event in events:
            if event["event_id"] in skip:
                log.info("Skipping already-ingested event: %s (%s)", event["name"], event["date"])
                continue
            log.info("Scraping event: %s (%s)", event["name"], event["date"])
            data = _scrape_event(event, seen_fids, get)
            yield event, data


def scrape_new_data(since: date, existing_fighter_ids: set[str] | None = None) -> dict:
    """
    Scrape all UFC events completed after *since* from ufcstats.com.

    Returns {"fighters": [...], "fights": [...], "fight_stats": [...]}.
    Pass *existing_fighter_ids* to skip bio scraping for fighters already in DB.
    """
    all_fighters: list[dict] = []
    all_fights:   list[dict] = []
    all_stats:    list[dict] = []

    for _event, data in scrape_events_iter(since, existing_fighter_ids):
        all_fighters.extend(data["fighters"])
        all_fights.extend(data["fights"])
        all_stats.extend(data["fight_stats"])

    log.info(
        "UFCStats: %d fighters, %d fights, %d fight_stats rows",
        len(all_fighters), len(all_fights), len(all_stats),
    )
    return {"fighters": all_fighters, "fights": all_fights, "fight_stats": all_stats}
