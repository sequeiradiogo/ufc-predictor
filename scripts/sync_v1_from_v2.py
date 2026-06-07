"""
sync_v1_from_v2.py -- Rebuild ufc_v2.db (mdabbert career-aggregate DB) from
ufc_ufcstats.db (UFCStats raw per-fight DB).

Reads all fights chronologically, computes running career averages with shift(1)
leakage prevention, and writes a fresh ufc_v2.db.  This replaces the manual
step of updating ufc-master.csv and running db/ingest_mdabbert.py.

ACCURACY NOTE: The current ufc_v2.db and models_v1/ were built from the Kaggle
ufc-master.csv pipeline (68.8% backtest accuracy).  Running this script rebuilds
ufc_v2.db from UFCStats rolling stats, which produces correct per-minute splm
values but currently yields ~60% accuracy after retraining.  After running,
always run: python scripts/backtest_v1.py --from-year 2025 --model ensemble
before committing retrained model artifacts.

Post-event workflow:
    python scripts/scrape_history.py      # update UFCStats raw DB
    python scripts/sync_v1_from_v2.py    # rebuild v1 career-average DB
    python scripts/backtest_v1.py --from-year 2025 --model ensemble  # verify accuracy

Options:
    --dry-run       Print summary without writing to disk.
    --v2-db PATH    Source UFCStats DB  (default: config.DB_PATH).
    --v1-db PATH    Target mdabbert DB  (default: config.DB_V1_PATH).
"""

import argparse
import bisect
import hashlib
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, DB_V1_PATH, RAW_DIR, NAME_ALIASES
from utils.logger import get_logger

log = get_logger(__name__)


# ── ID generation (mirrors db/ingest_mdabbert.py) ─────────────────────────────

def _fid(name: str) -> str:
    """Deterministic 16-char hex fighter ID from name (mdabbert convention)."""
    return hashlib.md5(name.lower().strip().encode()).hexdigest()[:16]


def _fight_id_v1(r_name: str, b_name: str, dt: str) -> str:
    """Deterministic 16-char hex fight ID from corner names + date."""
    key = f"{r_name.lower().strip()}|{b_name.lower().strip()}|{dt}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _age_years(dob, fight_date: str) -> float | None:
    """Age in years at fight_date, or None if dob is missing."""
    if not dob:
        return None
    try:
        dob_dt = date.fromisoformat(str(dob)[:10])
        fd_dt  = date.fromisoformat(str(fight_date)[:10])
        return (fd_dt - dob_dt).days / 365.25
    except (ValueError, TypeError):
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_v2_data(
    v2_conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load fighters, fights, and raw fight_stats from the UFCStats DB.
    Returns (fighters_df, fights_df, stats_df).
    """
    fighters_df = pd.read_sql_query(
        "SELECT fighter_id, name, height, reach, stance, dob FROM fighters",
        v2_conn,
    )

    fights_df = pd.read_sql_query(
        """
        SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id,
               method, title_fight, division, odds_red, odds_blue,
               COALESCE(finish_round,   1)      AS finish_round,
               COALESCE(match_time_sec, 0)      AS round_time_sec,
               COALESCE(no_of_rounds,   3)      AS no_of_rounds,
               COALESCE(gender, 'male')         AS gender
        FROM fights
        ORDER BY date ASC, fight_id ASC
        """,
        v2_conn,
    )

    # Rolling pre-fight stats computed by rolling.py (shift(1), leakage-free).
    # These are the final values -- do NOT cumulate them again.
    stats_df = pd.read_sql_query(
        """
        SELECT fight_id, fighter_id, corner,
               CAST(COALESCE(splm,        0) AS REAL) AS splm,
               CAST(COALESCE(sig_str_acc, 0) / 100.0 AS REAL) AS avg_sig_str_pct,
               CAST(COALESCE(td_avg,      0)         AS REAL) AS td_avg,
               CAST(COALESCE(td_acc,      0) / 100.0 AS REAL) AS avg_td_pct,
               CAST(COALESCE(sub_avg,     0) AS REAL) AS avg_sub_att,
               CAST(COALESCE(sapm,        0) AS REAL) AS sapm,
               CAST(COALESCE(str_def,     0) AS REAL) AS str_def,
               CAST(COALESCE(td_def,      0) AS REAL) AS td_def,
               CAST(COALESCE(wins,        0) AS REAL) AS wins,
               CAST(COALESCE(losses,      0) AS REAL) AS losses,
               height, reach, stance, dob
        FROM fight_stats
        """,
        v2_conn,
    )

    return fighters_df, fights_df, stats_df


# ── Historical rankings ───────────────────────────────────────────────────────

_RANKINGS_CSV = RAW_DIR / "rankings_history.csv"

# Reverse alias map: canonical UFCStats name (lower) -> rankings CSV name (lower).
# Derived from NAME_ALIASES which maps alt_name -> canonical.
_ALIAS_TO_CANONICAL = {
    v.lower().strip(): v.lower().strip()          # identity pass-through
    for v in NAME_ALIASES.values()
}
_RANKINGS_NAME_MAP: dict[str, str] = {
    alt.lower().strip(): canonical.lower().strip()
    for alt, canonical in NAME_ALIASES.items()
}


def _load_rankings_history(csv_path: Path = _RANKINGS_CSV) -> dict:
    """
    Load weekly ranking snapshots from rankings_history.csv.

    Returns {(fighter_canonical_lower, division_lower): [(date_str, rank), ...]}
    sorted by date ascending.  Keys use the canonical UFCStats fighter name so
    lookups via v2id_to_name work without extra translation.
    rank is an int 0-15 (0 = champion).
    """
    if not csv_path.exists():
        log.warning("Rankings history not found at %s -- weightclass_rank will be NULL.", csv_path)
        return {}
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["date", "weightclass", "fighter", "rank"],
        )
        # Drop Pound-for-Pound entries (not a fight division)
        df = df[~df["weightclass"].str.contains("Pound-for-Pound", case=False, na=False)]
        df = df.dropna(subset=["fighter", "rank"])
        df["rank"] = df["rank"].astype(int)
        df = df.sort_values("date").reset_index(drop=True)

        lookup: dict = {}
        for row in df.itertuples(index=False):
            raw = str(row.fighter).lower().strip()
            # Normalise to canonical UFCStats name via NAME_ALIASES where available
            canonical = _RANKINGS_NAME_MAP.get(raw, raw)
            div = str(row.weightclass).lower().strip()
            key = (canonical, div)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append((str(row.date)[:10], int(row.rank)))

        log.info("Loaded rankings history: %d fighter-division series from %s.",
                 len(lookup), csv_path.name)
        return lookup
    except Exception as exc:
        log.warning("Could not load rankings history (%s) -- weightclass_rank will be NULL.", exc)
        return {}


def _lookup_rank(
    rankings: dict,
    fighter_name: str,
    division: str,
    fight_date: str,
) -> float | None:
    """
    Return the fighter's rank in division on the most recent snapshot date
    that is <= fight_date, or None if no ranking exists before that fight.
    """
    key = (fighter_name.lower().strip(), division.lower().strip())
    series = rankings.get(key)
    if not series:
        return None
    dates = [s[0] for s in series]
    pos = bisect.bisect_right(dates, fight_date) - 1
    if pos < 0:
        return None
    return float(series[pos][1])


# ── Outcome stats (sequential -- required for win/loss streaks) ───────────────

def _compute_outcome_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-fight streaks and win-by-method per (fight_id, fighter_id).

    wins/losses come from the UFCStats rolling columns -- only streaks and
    win-by-method breakdowns require sequential per-fighter processing.
    long_df must be sorted by (fighter_id, date, fight_id) beforehand.
    """
    records = []

    for fighter_id, group in long_df.groupby("fighter_id", sort=False):
        win_streak = lose_streak = longest_win_streak = 0
        win_by_ko = win_by_sub = win_by_dec_unanimous = win_by_dec_split = 0
        total_rounds_fought = total_title_bouts = 0

        for _, row in group.iterrows():
            # Snapshot before this fight
            records.append({
                "fight_id":              row["fight_id"],
                "fighter_id":            fighter_id,
                "career_win_streak":     win_streak,
                "career_lose_streak":    lose_streak,
                "longest_win_streak":    longest_win_streak,
                "win_by_ko":             win_by_ko,
                "win_by_sub":            win_by_sub,
                "win_by_dec_unanimous":  win_by_dec_unanimous,
                "win_by_dec_split":      win_by_dec_split,
                "total_rounds_fought":   total_rounds_fought,
                "total_title_bouts":     total_title_bouts,
            })

            # Advance state
            won  = bool(row["won"])
            draw = bool(row["draw"])
            m    = str(row.get("method") or "").lower()

            if draw:
                win_streak  = 0
                lose_streak = 0
            elif won:
                win_streak += 1
                lose_streak = 0
                if win_streak > longest_win_streak:
                    longest_win_streak = win_streak
                if "ko" in m or "tko" in m:
                    win_by_ko += 1
                elif "sub" in m:
                    win_by_sub += 1
                elif "split" in m:
                    win_by_dec_split += 1
                else:
                    win_by_dec_unanimous += 1
            else:
                lose_streak += 1
                win_streak  = 0

            total_rounds_fought += int(row.get("rounds_fought") or 1)
            if row.get("title_fight"):
                total_title_bouts += 1

    return pd.DataFrame(records)



# ── Schema ────────────────────────────────────────────────────────────────────

def _create_v1_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fighters (
            fighter_id  TEXT PRIMARY KEY,
            name        TEXT,
            height      REAL,
            reach       REAL,
            stance      TEXT,
            dob         TEXT
        );

        CREATE TABLE IF NOT EXISTS fights (
            fight_id      TEXT PRIMARY KEY,
            date          TEXT,
            r_fighter_id  TEXT,
            b_fighter_id  TEXT,
            winner_id     TEXT,
            method        TEXT,
            division      TEXT,
            title_fight   INTEGER DEFAULT 0,
            odds_red      REAL,
            odds_blue     REAL,
            match_time_sec INTEGER,
            no_of_rounds  INTEGER,
            gender        TEXT
        );

        CREATE TABLE IF NOT EXISTS fight_stats (
            fight_id             TEXT,
            fighter_id           TEXT,
            corner               TEXT,
            wins                 REAL,
            losses               REAL,
            career_win_streak    REAL,
            career_lose_streak   REAL,
            longest_win_streak   REAL,
            total_rounds_fought  REAL,
            total_title_bouts    REAL,
            avg_sig_str_pct      REAL,
            avg_sub_att          REAL,
            avg_td_pct           REAL,
            height               REAL,
            reach                REAL,
            stance               TEXT,
            age                  REAL,
            splm                 REAL,
            td_avg               REAL,
            weightclass_rank     REAL,
            win_by_ko            REAL,
            win_by_sub           REAL,
            win_by_dec_unanimous REAL,
            win_by_dec_split     REAL,
            total_fight_time     REAL,
            PRIMARY KEY (fight_id, fighter_id)
        );

        CREATE INDEX IF NOT EXISTS idx_fights_date      ON fights(date);
        CREATE INDEX IF NOT EXISTS idx_fights_rfighter  ON fights(r_fighter_id);
        CREATE INDEX IF NOT EXISTS idx_fightstats_fight ON fight_stats(fight_id);
        CREATE INDEX IF NOT EXISTS idx_fightstats_fid   ON fight_stats(fighter_id);
    """)
    conn.commit()


# ── Main rebuild ──────────────────────────────────────────────────────────────

def build_v1_db(v2_db_path: Path, v1_db_path: Path, dry_run: bool = False) -> None:
    """
    Full rebuild: read all fights from UFCStats DB, compute career averages,
    write fresh mdabbert DB.
    """
    if not v2_db_path.exists():
        log.error("UFCStats DB not found: %s", v2_db_path)
        sys.exit(1)

    # ── Load raw data ─────────────────────────────────────────────────────────
    log.info("Reading UFCStats DB: %s", v2_db_path)
    v2_conn = sqlite3.connect(str(v2_db_path))
    fighters_df, fights_df, stats_df = _load_v2_data(v2_conn)
    v2_conn.close()

    log.info("Loaded %d fighters, %d fights, %d stat rows.",
             len(fighters_df), len(fights_df), len(stats_df))

    if fights_df.empty:
        log.warning("No fights found in UFCStats DB. Nothing to sync.")
        return

    # ── ID mappings: UFCStats hex -> name -> mdabbert MD5 ─────────────────────
    v2id_to_name: dict[str, str] = {
        r.fighter_id: r.name
        for r in fighters_df.itertuples()
        if r.name
    }
    v2id_to_v1id: dict[str, str] = {
        v2id: _fid(name) for v2id, name in v2id_to_name.items()
    }

    # UFCStats fight_id -> mdabbert fight_id (needs r/b names resolved first)
    v2fid_to_v1fid: dict[str, str] = {}
    for r in fights_df.itertuples():
        r_name = v2id_to_name.get(r.r_fighter_id)
        b_name = v2id_to_name.get(r.b_fighter_id)
        if r_name and b_name:
            v2fid_to_v1fid[r.fight_id] = _fight_id_v1(r_name, b_name, r.date)

    skipped = len(fights_df) - len(v2fid_to_v1fid)
    if skipped:
        log.warning("Skipped %d fights with unresolvable fighter names.", skipped)
    log.info("Resolved %d/%d fights to v1 IDs.", len(v2fid_to_v1fid), len(fights_df))

    # ── Total fight time (seconds) per fight ──────────────────────────────────
    # round_time_sec = time elapsed in the final round; finish_round = round number.
    # Total = time_in_final_round + (finish_round - 1) full rounds.
    fights_df["fight_time_sec"] = (
        fights_df["round_time_sec"].fillna(0) +
        (fights_df["finish_round"].fillna(1) - 1) * 300
    ).clip(lower=1)

    # ── Build per-fighter-per-fight long table ────────────────────────────────
    red_rows = fights_df[[
        "fight_id", "date", "r_fighter_id", "winner_id", "method",
        "title_fight", "fight_time_sec",
    ]].rename(columns={"r_fighter_id": "fighter_id"}).copy()
    red_rows["corner"] = "r"

    blue_rows = fights_df[[
        "fight_id", "date", "b_fighter_id", "winner_id", "method",
        "title_fight", "fight_time_sec",
    ]].rename(columns={"b_fighter_id": "fighter_id"}).copy()
    blue_rows["corner"] = "b"

    long_df = pd.concat([red_rows, blue_rows], ignore_index=True)

    # NaN winner_id = draw; string equality on NaN is already False in pandas
    long_df["won"]  = long_df["winner_id"] == long_df["fighter_id"]
    long_df["draw"] = long_df["winner_id"].isna()

    # Approximate rounds fought: ceiling(total_seconds / 300)
    long_df["rounds_fought"] = np.ceil(
        long_df["fight_time_sec"] / 300
    ).clip(lower=1).astype(int)

    # Merge pre-computed rolling stats and bio from fight_stats.
    # These columns are already shift(1) leakage-free (rolling.py does the shift).
    # _load_v2_data already aliases UFCStats column names to mdabbert v1 names
    # (sig_str_acc->avg_sig_str_pct, td_acc->avg_td_pct, sub_avg->avg_sub_att).
    rolling_cols = [
        "fight_id", "fighter_id",
        "splm", "avg_sig_str_pct", "td_avg", "avg_td_pct", "avg_sub_att",
        "sapm", "str_def", "td_def", "wins", "losses",
        "height", "reach", "stance", "dob",
    ]
    long_df = long_df.merge(
        stats_df[[c for c in rolling_cols if c in stats_df.columns]],
        on=["fight_id", "fighter_id"],
        how="left",
    )
    for col in ["splm", "avg_sig_str_pct", "td_avg", "avg_td_pct", "avg_sub_att",
                "sapm", "str_def", "td_def", "wins", "losses"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce").fillna(0)

    # Sort chronologically per fighter -- outcome stats need chronological order
    long_df = long_df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    # ── Compute outcome stats ─────────────────────────────────────────────────
    log.info("Computing outcome stats (streaks, win-by-method)...")
    outcome_stats = _compute_outcome_stats(long_df)

    # ── Assemble fight_stats_v1 ───────────────────────────────────────────────
    fight_stats_v1 = long_df[[
        "fight_id", "fighter_id", "corner",
        "height", "reach", "stance", "dob",
        "splm", "avg_sig_str_pct", "td_avg", "avg_td_pct", "avg_sub_att",
        "sapm", "str_def", "td_def", "wins", "losses",
    ]].copy()
    fight_stats_v1 = fight_stats_v1.merge(outcome_stats, on=["fight_id", "fighter_id"], how="left")

    # age at fight date from bio dob
    date_map = dict(zip(fights_df["fight_id"], fights_df["date"]))
    fight_stats_v1["fight_date"] = fight_stats_v1["fight_id"].map(date_map)
    fight_stats_v1["age"] = [
        _age_years(row.dob, row.fight_date)
        for row in fight_stats_v1[["dob", "fight_date"]].itertuples()
    ]

    # total_fight_time: wins+losses proxy (mdabbert convention; used for debutant detection)
    fight_stats_v1["total_fight_time"] = (
        fight_stats_v1["wins"].fillna(0) + fight_stats_v1["losses"].fillna(0)
    )

    # Historical weightclass_rank from rankings_history.csv
    # NaN = unranked -> encoded as 16 at feature-build time
    rankings = _load_rankings_history()
    if rankings:
        rank_rows = []
        for r in fights_df.itertuples():
            r_name = v2id_to_name.get(r.r_fighter_id, "")
            b_name = v2id_to_name.get(r.b_fighter_id, "")
            div    = str(r.division or "").lower().strip()
            r_rank = _lookup_rank(rankings, r_name, div, str(r.date)[:10])
            b_rank = _lookup_rank(rankings, b_name, div, str(r.date)[:10])
            rank_rows.append({"fight_id": r.fight_id, "corner": "r", "weightclass_rank": r_rank})
            rank_rows.append({"fight_id": r.fight_id, "corner": "b", "weightclass_rank": b_rank})
        rank_df = pd.DataFrame(rank_rows)
        fight_stats_v1 = fight_stats_v1.merge(rank_df, on=["fight_id", "corner"], how="left")
        matched = fight_stats_v1["weightclass_rank"].notna().sum()
        log.info("Ranked entries: %d/%d fight_stats rows have a pre-fight rank.",
                 matched, len(fight_stats_v1))
    else:
        fight_stats_v1["weightclass_rank"] = None

    # Remap IDs to mdabbert MD5 format
    fight_stats_v1["fighter_id"] = fight_stats_v1["fighter_id"].map(v2id_to_v1id)
    fight_stats_v1["fight_id"]   = fight_stats_v1["fight_id"].map(v2fid_to_v1fid)
    fight_stats_v1 = fight_stats_v1.dropna(subset=["fighter_id", "fight_id"])
    # Drop duplicate (fight_id, fighter_id) pairs that arise when two fighters
    # meet twice on the same date (same-night tournament rematches, e.g. UFC 1997).
    fight_stats_v1 = fight_stats_v1.drop_duplicates(subset=["fight_id", "fighter_id"], keep="first")

    # ── Assemble fights_v1 ────────────────────────────────────────────────────
    fights_v1 = fights_df[fights_df["fight_id"].isin(v2fid_to_v1fid)].copy()
    fights_v1["fight_id"]      = fights_v1["fight_id"].map(v2fid_to_v1fid)
    fights_v1["r_fighter_id"]  = fights_v1["r_fighter_id"].map(v2id_to_v1id)
    fights_v1["b_fighter_id"]  = fights_v1["b_fighter_id"].map(v2id_to_v1id)
    # NaN winner_id stays NaN (draw); map only non-null values
    fights_v1["winner_id"] = fights_v1["winner_id"].map(
        lambda x: v2id_to_v1id.get(x) if pd.notna(x) else None
    )
    # match_time_sec in mdabbert convention = total fight time (not round time)
    fights_v1 = fights_v1.rename(columns={"fight_time_sec": "match_time_sec"})
    fights_v1 = fights_v1.dropna(subset=["r_fighter_id", "b_fighter_id"])

    # ── Assemble fighters_v1 ─────────────────────────────────────────────────
    fighters_v1 = fighters_df.copy()
    fighters_v1["fighter_id"] = fighters_v1["fighter_id"].map(v2id_to_v1id)
    fighters_v1 = fighters_v1.dropna(subset=["fighter_id"])
    fighters_v1 = fighters_v1.drop_duplicates(subset=["fighter_id"])

    # ── Summary ───────────────────────────────────────────────────────────────
    n_f  = len(fighters_v1)
    n_fi = len(fights_v1)
    n_s  = len(fight_stats_v1)
    date_min = fights_v1["date"].min() if n_fi else "N/A"
    date_max = fights_v1["date"].max() if n_fi else "N/A"
    log.info("Ready to write: %d fighters, %d fights (%s to %s), %d fight_stats rows.",
             n_f, n_fi, date_min, date_max, n_s)

    if dry_run:
        log.info("[DRY RUN] No changes written.")
        return

    # ── Backup existing DB ────────────────────────────────────────────────────
    if v1_db_path.exists():
        from datetime import datetime
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = v1_db_path.with_name(v1_db_path.stem + f"_backup_{ts}.db")
        shutil.copy2(v1_db_path, backup)
        log.info("Backed up existing DB -> %s", backup.name)

    # ── Write to DB ───────────────────────────────────────────────────────────
    log.info("Writing v1 DB: %s", v1_db_path)
    v1_conn = sqlite3.connect(str(v1_db_path))

    # Drop and recreate all three tables so schema is always up to date.
    fighters_v1[["fighter_id", "name", "height", "reach", "stance", "dob"]].to_sql(
        "fighters", v1_conn, if_exists="replace", index=False,
    )
    log.info("Wrote %d fighters.", n_f)

    fight_cols = [
        "fight_id", "date", "r_fighter_id", "b_fighter_id",
        "winner_id", "method", "division", "title_fight",
        "odds_red", "odds_blue", "match_time_sec", "no_of_rounds", "gender",
    ]
    fights_out = fights_v1[[c for c in fight_cols if c in fights_v1.columns]]
    fights_out = fights_out.drop_duplicates(subset=["fight_id"], keep="first")
    fights_out.to_sql("fights", v1_conn, if_exists="replace", index=False)
    log.info("Wrote %d fights.", len(fights_out))

    stat_cols = [
        "fight_id", "fighter_id", "corner",
        "wins", "losses", "career_win_streak", "career_lose_streak", "longest_win_streak",
        "total_rounds_fought", "total_title_bouts",
        "avg_sig_str_pct", "avg_sub_att", "avg_td_pct",
        "sapm", "str_def", "td_def",
        "height", "reach", "stance", "age",
        "splm", "td_avg",
        "weightclass_rank",
        "win_by_ko", "win_by_sub", "win_by_dec_unanimous", "win_by_dec_split",
        "total_fight_time",
    ]
    fight_stats_v1[[c for c in stat_cols if c in fight_stats_v1.columns]].to_sql(
        "fight_stats", v1_conn, if_exists="replace", index=False,
    )
    log.info("Wrote %d fight_stats rows.", n_s)

    # Also deduplicate fights (same-night rematches map to the same v1 fight_id)
    fights_v1_out = fights_v1[[c for c in fight_cols if c in fights_v1.columns]]
    n_fi = len(fights_v1_out.drop_duplicates(subset=["fight_id"]))

    # Recreate indexes dropped by to_sql replace
    cur = v1_conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_date         ON fights(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_rfighter     ON fights(r_fighter_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fight    ON fight_stats(fight_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fid      ON fight_stats(fighter_id);")
    v1_conn.commit()
    v1_conn.close()
    log.info("Sync complete. v1 DB: %s", v1_db_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ufc_v2.db (v1 career averages) from ufc_ufcstats.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Post-event workflow:
  python scripts/scrape_history.py       # update UFCStats raw DB
  python scripts/sync_v1_from_v2.py     # rebuild v1 career-average DB
  python ml/train_v1_models.py           # retrain if drift is significant
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without writing to disk.",
    )
    parser.add_argument(
        "--v2-db", type=Path, default=DB_PATH,
        help=f"Source UFCStats DB path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--v1-db", type=Path, default=DB_V1_PATH,
        help=f"Target mdabbert DB path (default: {DB_V1_PATH})",
    )
    args = parser.parse_args()

    log.info("UFC Predictor -- v1 DB Sync")
    log.info("Source: %s", args.v2_db)
    log.info("Target: %s", args.v1_db)
    if args.dry_run:
        log.info("Mode: dry-run (no writes)")

    build_v1_db(args.v2_db, args.v1_db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
