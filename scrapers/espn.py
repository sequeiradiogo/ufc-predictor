"""
espn.py -- Dana White's Contender Series fights + ESPN <-> UFCStats fighter ids, from ESPN's JSON API.

Endpoints (public, no key):
  calendar   site.api.espn.com/.../mma/ufc/scoreboard?dates=YYYY0601 -> leagues[0].calendar
             (every UFC and DWCS event of that year)
  event      site.web.api.espn.com/.../mma/ufc/fightcenter/{event_id}
             (fighters, winner, method, round/clock, summary stats -- one call per event)
  stats      sports.core.api.espn.com/.../events/{eid}/competitions/{cid}/competitors/{aid}/statistics
             (full per-fight stats: head/body/leg x distance/clinch/ground, TD, control, reversals)
  career     sports.core.api.espn.com/.../athletes/{aid}/eventlog -> every pro bout in any promotion
             (league slug, competition -> date/opponent/winner, status -> method); no stats outside
             the UFC. A 2026-11 check of 100 fighters found it complete against the pro record, and
             never short of UFCStats' record (extras: post-UFC bouts, TUF exhibition bouts)

ESPN and UFCStats share the UFC's official stats feed: on 74 UFC fights (2024-25) every
stat column, the corner order and the round/time matched exactly, so DWCS stats are in
the same format as fight_stats. The DWCS isn't on UFCStats at all.

Do NOT use a competitor's `displayRecord` (or the competitor `record` ref): it is the
fighter's *current* record, not the record at the time of the fight (it never changed
across 8852 consecutive UFC fights of the same fighter) -- reading it for a past fight
leaks every later result.

Fighters are matched across the two sources by id, not by name: every UFC fight on
ESPN is paired with its UFCStats row (same date +/-1 day, both names), which links
the two ids. Name-only fuzzy matching gave false positives ("Tony Johnson" ->
Anthony Johnson). A UFC debutant has no completed UFC fight yet, so predict_event.py
pairs the upcoming card the same way (map_card_fighters).
"""

import difflib
import hashlib
import json
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import NAME_ALIASES  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates={}"
_FIGHTCENTER = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/fightcenter/{}"
_STATS = ("https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events/{}/competitions/{}"
          "/competitors/{}/statistics")
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ufc-predictor)", "Accept": "application/json"}
_DELAY = 0.3

DWCS_RE = re.compile(r"contender series", re.I)
# Not UFC cards: the DWCS, plus TUF exhibition-style listings
_NON_UFC_RE = re.compile(r"contender series|road to ufc|ultimate fighter", re.I)

# fight_stats columns, in UFCStats naming
STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_atmpted", "total_str_landed", "total_str_atmpted",
    "td_landed", "td_atmpted", "sub_att", "reversals", "ctrl",
    "head_landed", "head_atmpted", "body_landed", "body_atmpted", "leg_landed", "leg_atmpted",
    "dist_landed", "dist_atmpted", "clinch_landed", "clinch_atmpted", "ground_landed", "ground_atmpted",
]


# -- HTTP ----------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(_DELAY)
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            log.warning("ESPN request failed (%s), retrying: %s", exc, url)
            time.sleep(2 * (attempt + 1))
    return {}


class CachedFetcher:
    """Thread-safe GET with an on-disk JSON cache (one file per URL) -- makes long scrapes resumable.

    Completed fights never change, so a cached response is reused forever; delete the
    cache directory to force a refresh. 404s are cached as {} too.
    """

    def __init__(self, cache_dir: Path, delay: float = _DELAY):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._local = threading.local()

    def _path(self, url: str) -> Path:
        key = hashlib.md5(url.split("?")[0].replace("http://", "https://").encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, url: str) -> dict:
        url = url.replace("http://", "https://")
        path = self._path(url)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update(_HEADERS)
        data: dict = {}
        for attempt in range(4):
            try:
                time.sleep(self.delay)
                resp = self._local.session.get(url, timeout=30)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as exc:
                if attempt == 3:
                    raise
                log.warning("ESPN request failed (%s), retrying: %s", exc, url)
                time.sleep(3 * (attempt + 1))
        path.parent.mkdir(exist_ok=True)
        # one tmp per thread: two threads fetching the same URL (shared bouts) raced on one name
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            tmp.replace(path)
        except PermissionError:   # Windows: the other thread's replace holds the file; same content
            tmp.unlink(missing_ok=True)
        return data


_ATHLETE_EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/athletes/{}/eventlog?limit=200&page={}"


def athlete_eventlog(fetch: CachedFetcher, athlete_id: str) -> list[dict]:
    """Played bouts of an athlete's whole pro career: [{league, event_ref, competition_ref}]."""
    out, page = [], 1
    while True:
        data = fetch.get(_ATHLETE_EVENTLOG.format(athlete_id, page))
        ev = data.get("events") or {}
        for it in ev.get("items", []):
            if not it.get("played"):
                continue
            ref = (it.get("competition") or {}).get("$ref", "")
            m = re.search(r"/leagues/([^/]+)/events/(\d+)/competitions/(\d+)", ref)
            if m:
                out.append({"league": m.group(1), "event_id": m.group(2), "competition_id": m.group(3),
                            "event_ref": (it.get("event") or {}).get("$ref", ""), "competition_ref": ref})
        if page >= int(ev.get("pageCount") or 1):
            return out
        page += 1


def career_bout(fetch: CachedFetcher, competition_ref: str) -> dict:
    """Date, both athletes (order, winner), method and round/clock of one bout ({} if missing)."""
    c = fetch.get(competition_ref)
    comp = c.get("competitors") or []
    if not c or len(comp) != 2:
        return {}
    st_ref = (c.get("status") or {}).get("$ref")
    st = fetch.get(st_ref) if st_ref else {}
    return {
        "date": _local_date(c["date"]) if c.get("date") else "",
        "athletes": [(str(x.get("id")), bool(x.get("winner"))) for x in sorted(comp, key=lambda x: x.get("order", 0))],
        "method": (st.get("result") or {}).get("displayName") or "",
        "finish_round": int(st.get("period") or 0),
        "match_time_sec": float(st.get("clock") or 0),
        "sched_rounds": ((c.get("format") or {}).get("regulation") or {}).get("periods"),
    }


def bout_outcome(fetch: CachedFetcher, competition_ref: str) -> dict:
    """Date + both athletes' winner flags of one bout; the status (method) is only fetched when
    nobody won, to tell a draw from a no contest. Lighter than career_bout ({} if missing)."""
    c = fetch.get(competition_ref)
    comp = c.get("competitors") or []
    if not c or len(comp) != 2 or not c.get("date"):
        return {}
    athletes = [(str(x.get("id")), bool(x.get("winner"))) for x in sorted(comp, key=lambda x: x.get("order", 0))]
    method = ""
    if not any(w for _, w in athletes):
        st_ref = (c.get("status") or {}).get("$ref")
        method = ((fetch.get(st_ref) if st_ref else {}).get("result") or {}).get("displayName") or ""
    return {"date": _local_date(c["date"]), "athletes": athletes, "method": method}


# -- Names / dates -------------------------------------------------------------

def name_key(name: str) -> str:
    """ASCII-folded, lowercase, punctuation-free; NAME_ALIASES applied."""
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = " ".join(re.sub(r"[^a-z ]", "", n.lower().replace("-", " ")).split())
    return NAME_ALIASES.get(n, n).lower()


def _local_date(iso: str) -> str:
    """ESPN times are UTC; US cards start after midnight UTC, so shift back 7 h."""
    dt = datetime.strptime(iso[:16], "%Y-%m-%dT%H:%M")
    return (dt - timedelta(hours=7)).date().isoformat()


# -- Calendar / events ---------------------------------------------------------

def calendar(years: range | list[int]) -> list[dict]:
    """[{event_id, name, start}] for every UFC/DWCS event ESPN lists in *years*."""
    out, seen = [], set()
    for yr in years:
        data = _get(_SCOREBOARD.format(f"{yr}0601"))
        for e in (data.get("leagues") or [{}])[0].get("calendar", []):
            m = re.search(r"/events/(\d+)", e.get("event", {}).get("$ref", ""))
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                out.append({"event_id": m.group(1), "name": e.get("label", ""),
                            "start": e.get("startDate", "")})
    return out


def is_dwcs(event: dict) -> bool:
    return bool(DWCS_RE.search(event["name"]))


def is_ufc_card(event: dict) -> bool:
    return not _NON_UFC_RE.search(event["name"])


def _summary_stats(stats: list[dict]) -> dict:
    """fightcenter's summary stats ('sigStrikes' -> '46/101') in UFCStats naming."""
    raw = {}
    for s in stats or []:
        dv = str(s.get("displayValue", ""))
        if "/" in dv:
            landed, att = dv.split("/", 1)
            raw[s["name"] + "Landed"], raw[s["name"] + "Attempted"] = landed, att
        else:
            raw[s["name"]] = s.get("value")
    g = lambda k: float(raw.get(k) or 0)  # noqa: E731
    return {
        "kd": g("knockDowns"), "sig_str_landed": g("sigStrikesLanded"),
        "sig_str_atmpted": g("sigStrikesAttempted"), "total_str_landed": g("totalStrikesLanded"),
        "total_str_atmpted": g("totalStrikesAttempted"), "td_landed": g("takedownsLanded"),
        "td_atmpted": g("takedownsAttempted"), "sub_att": g("submissions"), "ctrl": g("timeInControl"),
        "head_landed": g("headStrikesLanded"), "head_atmpted": g("headStrikesAttempted"),
        "body_landed": g("bodyStrikesLanded"), "body_atmpted": g("bodyStrikesAttempted"),
        "leg_landed": g("legStrikesLanded"), "leg_atmpted": g("legStrikesAttempted"),
    }


def full_stats(event_id: str, competition_id: str, athlete_id: str) -> dict:
    """Every STAT_COLS value for one fighter in one fight ({} if ESPN has none)."""
    data = _get(_STATS.format(event_id, competition_id, athlete_id))
    raw = {st["name"]: st.get("value") for cat in data.get("splits", {}).get("categories", [])
           for st in cat.get("stats", [])}
    if not raw:
        return {}
    g = lambda k: float(raw.get(k) or 0)  # noqa: E731
    out = {
        "kd": g("knockDowns"),
        "sig_str_landed": g("sigStrikesLanded"), "sig_str_atmpted": g("sigStrikesAttempted"),
        "total_str_landed": g("totalStrikesLanded"), "total_str_atmpted": g("totalStrikesAttempted"),
        "td_landed": g("takedownsLanded"), "td_atmpted": g("takedownsAttempted"),
        "sub_att": g("submissions"), "reversals": g("reversals"), "ctrl": g("timeInControl"),
    }
    for la, col in (("Landed", "landed"), ("Attempted", "atmpted")):
        for zone in ("Head", "Body", "Leg"):
            out[f"{zone.lower()}_{col}"] = sum(
                g(f"sig{pos}{zone}Strikes{la}") for pos in ("Distance", "Clinch", "Ground"))
        for pos, short in (("Distance", "dist"), ("Clinch", "clinch"), ("Ground", "ground")):
            out[f"{short}_{col}"] = sum(
                g(f"sig{pos}{zone}Strikes{la}") for zone in ("Head", "Body", "Leg"))
    return out


def event_fights(event_id: str) -> list[dict]:
    """Every bout of an event, completed or scheduled, with fightcenter's summary stats.

    Corner order follows ESPN's `order` (1 = Red on UFC cards).
    """
    data = _get(_FIGHTCENTER.format(event_id))
    ev_date = (data.get("event") or {}).get("date", "")
    rows = []
    for card in (data.get("cards") or {}).values():
        for c in card.get("competitions", []):
            comp = sorted(c.get("competitors", []), key=lambda x: x.get("order", 0))
            if len(comp) != 2:
                continue
            st = c.get("status") or {}
            res = st.get("result") or {}
            period = int(st.get("period") or 0)
            row = {
                "event_id": event_id, "competition_id": c.get("id"),
                "date": _local_date(c.get("date") or ev_date),
                "completed": bool((st.get("type") or {}).get("completed")),
                "method": res.get("displayName") or "", "method_detail": res.get("description") or "",
                "finish_round": period, "match_time_sec": float(st.get("clock") or 0),
                "sched_rounds": ((c.get("format") or {}).get("regulation") or {}).get("periods"),
            }
            for side, x in zip(("r", "b"), comp):
                a = x.get("athlete") or {}
                row[f"{side}_espn_id"] = str(a.get("id") or x.get("id"))
                row[f"{side}_name"] = a.get("displayName", "")
                row[f"{side}_winner"] = bool(x.get("winner"))
                row[f"{side}_stats"] = _summary_stats(x.get("stats"))
            rows.append(row)
    return rows


def result_of(row: dict, side: str) -> str:
    """W / L / D (draw) / NC for *side* ('r' or 'b') of an event_fights row."""
    other = "b" if side == "r" else "r"
    if row[f"{side}_winner"]:
        return "W"
    if row[f"{other}_winner"]:
        return "L"
    return "NC" if "no contest" in row["method"].lower() else "D"


# -- ESPN id <-> UFCStats id ---------------------------------------------------

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, name_key(a), name_key(b)).ratio()


def pair_fight(r_name: str, b_name: str, candidates: list[tuple[str, str, str, str]],
               floor: float = 0.75) -> tuple[str, str] | None:
    """Match one ESPN bout to (red_id, blue_id) among (r_id, r_name, b_id, b_name) candidates.

    Both names must clear *floor*; corners are taken from whichever orientation scores higher.
    """
    best, best_score = None, 0.0
    for rid, rn, bid, bn in candidates:
        straight = (_sim(r_name, rn), _sim(b_name, bn))
        swapped = (_sim(r_name, bn), _sim(b_name, rn))
        for (s1, s2), ids in ((straight, (rid, bid)), (swapped, (bid, rid))):
            if min(s1, s2) >= floor and s1 + s2 > best_score:
                best, best_score = ids, s1 + s2
    return best


def build_fighter_map(ufc_events: list[dict], ufc_conn) -> dict[str, str]:
    """ESPN athlete id -> UFCStats fighter_id from completed UFC fights in both sources.

    Each paired fight is one vote; an ESPN id keeps its most-voted UFCStats id.
    """
    names = dict(ufc_conn.execute("SELECT fighter_id, name FROM fighters"))
    votes: dict[str, Counter] = defaultdict(Counter)
    n_fights = n_unpaired = 0
    for ev in ufc_events:
        try:
            rows = event_fights(ev["event_id"])
        except requests.RequestException as exc:
            log.warning("ESPN event %s (%s) skipped: %s", ev["event_id"], ev["name"], exc)
            continue
        for r in rows:
            if not r["completed"]:
                continue
            n_fights += 1
            cands = [(rid, names.get(rid, ""), bid, names.get(bid, "")) for rid, bid in ufc_conn.execute(
                "SELECT r_fighter_id, b_fighter_id FROM fights "
                "WHERE date BETWEEN date(?, '-1 day') AND date(?, '+1 day')", (r["date"], r["date"]))]
            ids = pair_fight(r["r_name"], r["b_name"], cands)
            if ids is None:
                n_unpaired += 1
                continue
            votes[r["r_espn_id"]][ids[0]] += 1
            votes[r["b_espn_id"]][ids[1]] += 1
    conflicts = sum(len(c) > 1 for c in votes.values())
    log.info("ESPN fighter map: %d UFC fights, %d unpaired, %d athletes mapped (%d with conflicting votes)",
             n_fights, n_unpaired, len(votes), conflicts)
    return {aid: c.most_common(1)[0][0] for aid, c in votes.items()}


def map_card_fighters(card_date: str, card: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """UFCStats fighter_id -> ESPN athlete id for an upcoming card.

    *card* is [(r_id, r_name, b_id, b_name)] from the UFCStats scrape. Debutants have
    no completed UFC fight to pair, so the scheduled bouts on ESPN are paired instead.
    Fails soft (returns what it found) -- the caller falls back to the stored map.
    """
    out: dict[str, str] = {}
    try:
        d = datetime.fromisoformat(card_date[:10])
        events = [e for e in calendar([d.year]) if is_ufc_card(e)
                  and abs((datetime.fromisoformat(_local_date(e["start"])) - d).days) <= 1]
        for ev in events:
            for r in event_fights(ev["event_id"]):
                ids = pair_fight(r["r_name"], r["b_name"], card)
                if ids:
                    out[ids[0]], out[ids[1]] = r["r_espn_id"], r["b_espn_id"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("ESPN card mapping failed for %s: %s", card_date, exc)
    return out
