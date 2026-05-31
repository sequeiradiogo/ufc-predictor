"""
csv_builder.py -- Convert ufcstats-scraped per-fight data into ufc-master.csv rows.

Each row in ufc-master.csv describes one fight with BOTH fighters' pre-fight
career stats.  This module:
  1. Reads each fighter's current career state from the existing DB
  2. Processes new fights in chronological order, tracking running career totals
  3. For each fight emits a row with PRE-FIGHT stats, then updates the state
  4. Returns a DataFrame ready to append to ufc-master.csv

The returned DataFrame matches the exact column schema expected by
db/ingest_mdabbert.py so the full pipeline can be re-run without modification.
"""

import sqlite3
import sys
from copy import deepcopy
from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.logger import get_logger

log = get_logger(__name__)

# Reverse-map: DB method -> CSV finish code (used by ingest_mdabbert)
_METHOD_TO_FINISH = {
    "KO/TKO":               "KO/TKO",
    "Submission":            "SUB",
    "Decision - Unanimous":  "U-DEC",
    "Decision - Split":      "S-DEC",
    "Decision - Majority":   "M-DEC",
    "TKO - Doctor's Stoppage": "KO/TKO",
    "Could Not Continue":    "KO/TKO",
}

_DIVISION_TO_GENDER = {
    "women's strawweight": "FEMALE",
    "women's flyweight":   "FEMALE",
    "women's bantamweight": "FEMALE",
    "women's featherweight": "FEMALE",
}

_DIVISION_TO_ROUNDS = {
    # title bouts get 5 rounds; main events sometimes 5 too -- use title_fight flag
}


# ── Career state dataclass ───────────────────────────────────��────────────────

@dataclass
class _CareerState:
    """Running career statistics for one fighter, updated fight-by-fight."""
    fighter_id: str
    name: str = ""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_streak: int = 0
    lose_streak: int = 0
    longest_win_streak: int = 0
    total_rounds: int = 0
    total_title_bouts: int = 0
    win_by_ko: int = 0
    win_by_sub: int = 0
    win_by_dec_unanimous: int = 0
    win_by_dec_split: int = 0
    win_by_dec_majority: int = 0
    win_by_tko_doctor: int = 0
    # Running totals for per-minute stats (updated from ufcstats per-fight data)
    _total_sig_str_landed: float = 0.0
    _total_sig_str_atmpted: float = 0.0
    _total_td_landed: float = 0.0
    _total_td_atmpted: float = 0.0
    _total_sub_att: float = 0.0
    _total_fight_time_min: float = 0.0
    # Bio (static)
    height: float | None = None
    reach: float | None = None
    stance: str | None = None
    dob: str | None = None
    weightclass_rank: float | None = None

    # ---- Derived averages ----

    @property
    def splm(self) -> float:
        """Sig strikes per minute."""
        if self._total_fight_time_min <= 0:
            return 0.0
        return self._total_sig_str_landed / self._total_fight_time_min

    @property
    def avg_sig_str_pct(self) -> float:
        """Sig strike accuracy (0-1)."""
        if self._total_sig_str_atmpted <= 0:
            return 0.0
        return self._total_sig_str_landed / self._total_sig_str_atmpted

    @property
    def avg_td_landed(self) -> float:
        """Takedowns per 15 minutes."""
        if self._total_fight_time_min <= 0:
            return 0.0
        return self._total_td_landed / (self._total_fight_time_min / 15.0)

    @property
    def avg_td_pct(self) -> float:
        """Takedown accuracy (0-1)."""
        if self._total_td_atmpted <= 0:
            return 0.0
        return self._total_td_landed / self._total_td_atmpted

    @property
    def avg_sub_att(self) -> float:
        """Submission attempts per 15 minutes."""
        if self._total_fight_time_min <= 0:
            return 0.0
        return self._total_sub_att / (self._total_fight_time_min / 15.0)

    def age_at(self, fight_date: str) -> int | None:
        """Return age in years at fight_date, or None if dob unknown."""
        if not self.dob:
            return None
        try:
            dob = date.fromisoformat(self.dob[:10])
            fd  = date.fromisoformat(fight_date[:10])
            return (fd - dob).days // 365
        except (ValueError, TypeError):
            return None

    def update_with_fight(
        self,
        won: bool,
        draw: bool,
        method: str,
        title_fight: bool,
        rounds_fought: int,
        stats: dict | None,          # per-fight stats from ufcstats scraper
        fight_time_sec: int,
    ) -> None:
        """Update career state after a fight completes."""
        time_min = fight_time_sec / 60.0

        # Win/loss/draw counts
        if draw:
            self.draws += 1
            self.win_streak = 0
            self.lose_streak = 0
        elif won:
            self.wins += 1
            self.win_streak += 1
            self.lose_streak = 0
            if self.win_streak > self.longest_win_streak:
                self.longest_win_streak = self.win_streak
            m = method.lower()
            if "ko" in m or "tko" in m:
                self.win_by_ko += 1
            elif "sub" in m:
                self.win_by_sub += 1
            elif "split" in m:
                self.win_by_dec_split += 1
            elif "majority" in m:
                self.win_by_dec_majority += 1
            elif "dec" in m:
                self.win_by_dec_unanimous += 1
        else:
            self.losses += 1
            self.lose_streak += 1
            self.win_streak = 0

        # Cumulative counters
        self.total_rounds += rounds_fought
        if title_fight:
            self.total_title_bouts += 1

        # Per-minute stats (use ufcstats per-fight data if available)
        if stats and time_min > 0:
            self._total_sig_str_landed  += stats.get("sig_str_landed",  0) or 0
            self._total_sig_str_atmpted += stats.get("sig_str_atmpted", 0) or 0
            self._total_td_landed       += stats.get("td_landed",       0) or 0
            self._total_td_atmpted      += stats.get("td_atmpted",      0) or 0
            self._total_sub_att         += stats.get("sub_att",         0) or 0
            self._total_fight_time_min  += time_min


# ── Load career state from DB ──────────────────────────────��──────────────────

def _load_state_from_db(fighter_id: str, conn: sqlite3.Connection) -> _CareerState:
    """
    Build a _CareerState from the most recent fight_stats row in the DB.
    Estimates running totals from the stored averages + total_rounds_fought.
    """
    row = conn.execute("""
        SELECT
            fi.name,
            fs.wins, fs.losses, fs.career_win_streak, fs.career_lose_streak,
            fs.longest_win_streak, fs.total_rounds_fought, fs.total_title_bouts,
            fs.avg_sig_str_pct, fs.avg_sub_att, fs.avg_td_pct,
            fs.splm, fs.td_avg,
            fs.height, fs.reach, fs.stance, fs.age,
            fs.weightclass_rank,
            fs.win_by_ko, fs.win_by_sub, fs.win_by_dec_unanimous, fs.win_by_dec_split,
            f.date
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        LEFT JOIN fighters fi ON fs.fighter_id = fi.fighter_id
        WHERE fs.fighter_id = ?
        ORDER BY f.date DESC
        LIMIT 1
    """, (fighter_id,)).fetchone()

    if row is None:
        return _CareerState(fighter_id=fighter_id)

    cols = [d[0] for d in conn.execute("SELECT * FROM fight_stats LIMIT 0").description]

    def _v(name: str, default=0):
        try:
            v = row[cols.index(name) if name in cols else -1]
            return v if v is not None else default
        except (IndexError, ValueError):
            return default

    # Use column names from the raw row (sqlite3.Row would be better but let's be safe)
    r = dict(zip(
        ["name","wins","losses","win_streak","lose_streak","longest_win_streak",
         "total_rounds","total_title_bouts","avg_sig_str_pct","avg_sub_att","avg_td_pct",
         "splm","td_avg","height","reach","stance","age","weightclass_rank",
         "win_by_ko","win_by_sub","win_by_dec_unanimous","win_by_dec_split","date"],
        row
    ))

    def _f(k, d=0.0): return float(r.get(k) or d)
    def _i(k, d=0):   return int(r.get(k) or d)

    state = _CareerState(fighter_id=fighter_id, name=r.get("name") or "")
    state.wins                = _i("wins")
    state.losses              = _i("losses")
    state.win_streak          = _i("win_streak")
    state.lose_streak         = _i("lose_streak")
    state.longest_win_streak  = _i("longest_win_streak")
    state.total_rounds        = _i("total_rounds")
    state.total_title_bouts   = _i("total_title_bouts")
    state.win_by_ko           = _i("win_by_ko")
    state.win_by_sub          = _i("win_by_sub")
    state.win_by_dec_unanimous = _i("win_by_dec_unanimous")
    state.win_by_dec_split    = _i("win_by_dec_split")
    state.height              = _f("height") or None
    state.reach               = _f("reach") or None
    state.stance              = r.get("stance") or None
    state.weightclass_rank    = _f("weightclass_rank") or None

    # Back-compute running totals from stored averages
    total_time_min = _i("total_rounds") * 5.0 * 0.8   # rough proxy (rounds ~80% full)
    splm_val    = _f("splm")
    sig_pct     = _f("avg_sig_str_pct")
    td_avg_val  = _f("td_avg")
    td_pct      = _f("avg_td_pct")
    sub_att_val = _f("avg_sub_att")

    state._total_fight_time_min  = total_time_min
    state._total_sig_str_landed  = splm_val * total_time_min
    state._total_sig_str_atmpted = (splm_val / sig_pct * total_time_min) if sig_pct > 0 else 0.0
    state._total_td_landed       = td_avg_val * (total_time_min / 15.0)
    state._total_td_atmpted      = (td_avg_val / td_pct * (total_time_min / 15.0)) if td_pct > 0 else 0.0
    state._total_sub_att         = sub_att_val * (total_time_min / 15.0)

    return state


# ── Build one CSV row from pre-fight states ───────────────────────────────────

def _make_row(
    fight: dict,
    r_state: _CareerState,
    b_state: _CareerState,
    r_name: str,
    b_name: str,
) -> dict:
    """Produce a single ufc-master.csv row (pre-fight states for both fighters)."""
    fd = fight["date"]
    method = fight.get("method") or ""
    finish_code = _METHOD_TO_FINISH.get(method, "U-DEC")

    winner_id = fight.get("winner_id")
    if winner_id == fight["r_fighter_id"]:
        winner = "Red"
    elif winner_id == fight["b_fighter_id"]:
        winner = "Blue"
    else:
        winner = "Draw"

    division = fight.get("division") or ""
    gender = _DIVISION_TO_GENDER.get(division.lower(), "MALE")
    title_fight = bool(fight.get("title_fight"))
    no_rounds = 5 if title_fight else 3

    total_secs = fight.get("total_fight_time_secs") or 0

    def _stats(s: _CareerState, prefix: str) -> dict:
        return {
            f"{prefix}fighter":                 s.name,
            f"{prefix}wins":                    s.wins,
            f"{prefix}losses":                  s.losses,
            f"{prefix}current_win_streak":      s.win_streak,
            f"{prefix}current_lose_streak":     s.lose_streak,
            f"{prefix}longest_win_streak":      s.longest_win_streak,
            f"{prefix}draw":                    s.draws,
            f"{prefix}total_rounds_fought":     s.total_rounds,
            f"{prefix}total_title_bouts":       s.total_title_bouts,
            f"{prefix}win_by_KO/TKO":           s.win_by_ko,
            f"{prefix}win_by_Submission":        s.win_by_sub,
            f"{prefix}win_by_Decision_Unanimous": s.win_by_dec_unanimous,
            f"{prefix}win_by_Decision_Split":    s.win_by_dec_split,
            f"{prefix}win_by_Decision_Majority": s.win_by_dec_majority,
            f"{prefix}win_by_TKO_Doctor_Stoppage": s.win_by_tko_doctor,
            f"{prefix}avg_SIG_STR_pct":          round(s.avg_sig_str_pct, 4),
            f"{prefix}avg_SUB_ATT":              round(s.avg_sub_att, 4),
            f"{prefix}avg_TD_pct":               round(s.avg_td_pct, 4),
            f"{prefix}avg_SIG_STR_landed":       round(s.splm, 4),
            f"{prefix}avg_TD_landed":            round(s.avg_td_landed, 4),
            f"{prefix}Height_cms":               s.height,
            f"{prefix}Reach_cms":                s.reach,
            f"{prefix}Stance":                   s.stance,
            f"{prefix}age":                      s.age_at(fd),
            f"{prefix}match_weightclass_rank":   s.weightclass_rank,
        }

    row: dict[str, Any] = {
        "date":                fight["date"],
        "weight_class":        division,
        "title_bout":          title_fight,
        "Winner":              winner,
        "finish":              finish_code,
        "finish_round":        None,
        "finish_round_time":   None,
        "total_fight_time_secs": total_secs,
        "no_of_rounds":        no_rounds,
        "gender":              gender,
        "R_odds":              fight.get("odds_red"),
        "B_odds":              fight.get("odds_blue"),
        "location":            None,
    }
    row.update(_stats(r_state, "R_"))
    row.update(_stats(b_state, "B_"))

    # Pre-computed diff columns (ingest_mdabbert doesn't use these for fight_stats
    # but they exist in the CSV for completeness)
    row["lose_streak_dif"]       = b_state.lose_streak   - r_state.lose_streak
    row["win_streak_dif"]        = b_state.win_streak    - r_state.win_streak
    row["longest_win_streak_dif"]= b_state.longest_win_streak - r_state.longest_win_streak
    row["win_dif"]               = b_state.wins          - r_state.wins
    row["loss_dif"]              = b_state.losses        - r_state.losses
    row["total_round_dif"]       = b_state.total_rounds  - r_state.total_rounds
    row["total_title_bout_dif"]  = b_state.total_title_bouts - r_state.total_title_bouts
    row["ko_dif"]                = b_state.win_by_ko     - r_state.win_by_ko
    row["sub_dif"]               = b_state.win_by_sub    - r_state.win_by_sub
    row["height_dif"]            = ((b_state.height or 0) - (r_state.height or 0)) if r_state.height and b_state.height else None
    row["reach_dif"]             = ((b_state.reach or 0)  - (r_state.reach or 0))  if r_state.reach  and b_state.reach  else None
    r_age = r_state.age_at(fd)
    b_age = b_state.age_at(fd)
    row["age_dif"]               = (b_age - r_age) if r_age and b_age else None
    row["sig_str_dif"]           = round(b_state.splm - r_state.splm, 4)
    row["avg_sub_att_dif"]       = round(b_state.avg_sub_att - r_state.avg_sub_att, 4)
    row["avg_td_dif"]            = round(b_state.avg_td_landed - r_state.avg_td_landed, 4)

    return row


# ── Public API ────────────────────────────────���───────────────────────────────

def build_csv_rows(data: dict, db_path: Path) -> pd.DataFrame:
    """
    Convert scraped ufcstats data into ufc-master.csv format rows.

    *data* is the dict returned by scrapers.ufcstats.scrape_new_data().
    *db_path* is the existing DB used to initialise career stats.

    Returns a DataFrame ready to pd.concat onto ufc-master.csv.
    """
    fights     = data.get("fights",     [])
    all_stats  = data.get("fight_stats", [])
    scraped_fighters = {f["fighter_id"]: f for f in data.get("fighters", [])}

    if not fights:
        return pd.DataFrame()

    # Index per-fight stats: {fight_id: {fighter_id: stats_dict}}
    stats_index: dict[str, dict[str, dict]] = {}
    for s in all_stats:
        fid, pid = s["fight_id"], s["fighter_id"]
        stats_index.setdefault(fid, {})[pid] = s

    # Sort fights chronologically
    fights_sorted = sorted(fights, key=lambda f: f.get("date") or "")

    # Load career states from DB for all fighters in the batch
    all_fighter_ids = {f["r_fighter_id"] for f in fights} | {f["b_fighter_id"] for f in fights}
    conn = sqlite3.connect(str(db_path))

    states: dict[str, _CareerState] = {}
    for fid in all_fighter_ids:
        state = _load_state_from_db(fid, conn)
        # Fill bio from scraped data if DB has nothing
        if fid in scraped_fighters:
            bio = scraped_fighters[fid]
            if not state.name:
                state.name = bio.get("name") or ""
            if state.height is None:
                state.height = bio.get("height")
            if state.reach is None:
                state.reach = bio.get("reach")
            if state.stance is None:
                state.stance = bio.get("stance")
            if state.dob is None:
                state.dob = bio.get("dob")
        states[fid] = state

    conn.close()

    rows = []
    for fight in fights_sorted:
        r_fid = fight["r_fighter_id"]
        b_fid = fight["b_fighter_id"]

        r_state = states.get(r_fid, _CareerState(fighter_id=r_fid))
        b_state = states.get(b_fid, _CareerState(fighter_id=b_fid))
        r_name = r_state.name or scraped_fighters.get(r_fid, {}).get("name", r_fid)
        b_name = b_state.name or scraped_fighters.get(b_fid, {}).get("name", b_fid)

        # Fill names into fight for row building
        fight_with_names = {**fight, "r_name": r_name, "b_name": b_name}

        # Emit pre-fight row (snapshot BEFORE this fight)
        row = _make_row(fight_with_names, r_state, b_state, r_name, b_name)
        rows.append(row)

        # Update both fighters' career states with this fight's outcome
        fight_stats = stats_index.get(fight["fight_id"], {})
        winner_id   = fight.get("winner_id")
        method      = fight.get("method") or ""
        title_fight = bool(fight.get("title_fight"))
        total_secs  = int(fight.get("total_fight_time_secs") or 0)
        round_secs  = total_secs or 0
        # Approximate rounds_fought from finish time
        rounds_fought = max(1, round_secs // 300 + (1 if round_secs % 300 > 0 else 0))

        r_state.update_with_fight(
            won=winner_id == r_fid,
            draw=winner_id is None,
            method=method,
            title_fight=title_fight,
            rounds_fought=rounds_fought,
            stats=fight_stats.get(r_fid),
            fight_time_sec=round_secs,
        )
        b_state.update_with_fight(
            won=winner_id == b_fid,
            draw=winner_id is None,
            method=method,
            title_fight=title_fight,
            rounds_fought=rounds_fought,
            stats=fight_stats.get(b_fid),
            fight_time_sec=round_secs,
        )

    df = pd.DataFrame(rows)
    log.info("Built %d new CSV rows from scraped data", len(df))
    return df
