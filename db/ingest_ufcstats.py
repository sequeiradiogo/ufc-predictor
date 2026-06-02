"""
ingest_ufcstats.py -- Ingest UFCStats-scraped data into the per-fight granular database.

Consumes the dict returned by scrapers/ufcstats.scrape_new_data() and populates
db/ufc_ufcstats.db with a per-fight schema compatible with db/rolling.py.

Unlike ingest_mdabbert.py (which stores career averages), this script stores raw
per-fight counts (kd, strikes landed/attempted, ctrl, etc.) so that rolling.py can
compute proper pre-fight rolling windows for every fighter.

Usage:
    Called by scripts/scrape_history.py -- not normally run directly.
    For a full rebuild: python scripts/scrape_history.py
"""

import re
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_UFCSTATS_PATH, DIVISIONS
from utils.logger import get_logger

log = get_logger(__name__)

# Derive fighter weight limit (lbs) from division name for rolling.py pass-through.
_DIVISION_WEIGHT: dict[str, float] = {
    "women's strawweight":  115.0,
    "women's flyweight":    125.0,
    "women's bantamweight": 135.0,
    "women's featherweight": 145.0,
    "flyweight":            125.0,
    "bantamweight":         135.0,
    "featherweight":        145.0,
    "lightweight":          155.0,
    "welterweight":         170.0,
    "middleweight":         185.0,
    "light heavyweight":    205.0,
    "heavyweight":          265.0,
}

_WOMEN_DIVS = {d for d in DIVISIONS if d.startswith("women")}


def _round_time_secs(time_str: str) -> int:
    """Parse 'M:SS' finish-round time string to integer seconds within that round."""
    m = re.match(r"(\d+):(\d{2})", (time_str or "0:00").strip())
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def _create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript("""
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
            event_id      TEXT,
            date          TEXT,
            division      TEXT,
            r_fighter_id  TEXT,
            b_fighter_id  TEXT,
            winner_id     TEXT,
            method        TEXT,
            title_fight   INTEGER DEFAULT 0,
            odds_red      REAL,
            odds_blue     REAL,
            finish_round  INTEGER,
            match_time_sec INTEGER,
            no_of_rounds  INTEGER,
            gender        TEXT
        );

        CREATE TABLE IF NOT EXISTS fight_stats (
            fight_id          TEXT,
            fighter_id        TEXT,
            corner            TEXT,
            -- per-fight raw counts (input for rolling.py)
            kd                INTEGER,
            sig_str_landed    INTEGER,
            sig_str_atmpted   INTEGER,
            total_str_landed  INTEGER,
            total_str_atmpted INTEGER,
            td_landed         INTEGER,
            td_atmpted        INTEGER,
            sub_att           INTEGER,
            ctrl              INTEGER,
            head_landed       INTEGER,
            head_atmpted      INTEGER,
            body_landed       INTEGER,
            body_atmpted      INTEGER,
            leg_landed        INTEGER,
            leg_atmpted       INTEGER,
            dist_landed       INTEGER,
            dist_atmpted      INTEGER,
            clinch_landed     INTEGER,
            clinch_atmpted    INTEGER,
            ground_landed     INTEGER,
            ground_atmpted    INTEGER,
            total_fight_time  INTEGER,
            -- bio pass-throughs (rolling.py reads these via SELECT fs.*)
            height            REAL,
            reach             REAL,
            stance            TEXT,
            dob               TEXT,
            weight            REAL,
            PRIMARY KEY (fight_id, fighter_id)
        );

        CREATE INDEX IF NOT EXISTS idx_fights_date      ON fights(date);
        CREATE INDEX IF NOT EXISTS idx_fights_rfighter  ON fights(r_fighter_id);
        CREATE INDEX IF NOT EXISTS idx_fightstats_fight ON fight_stats(fight_id);
        CREATE INDEX IF NOT EXISTS idx_fightstats_fid   ON fight_stats(fighter_id);
    """)
    conn.commit()


def ingest(data: dict, db_path: Path = DB_UFCSTATS_PATH) -> None:
    """
    Ingest scraped UFC data into the per-fight granular database.

    data -- dict with keys "fighters", "fights", "fight_stats" as returned by
            scrapers/ufcstats.scrape_new_data().
    db_path -- destination SQLite file (created if absent).
    """
    fighters:   list[dict] = data.get("fighters", [])
    fights:     list[dict] = data.get("fights", [])
    fight_stats: list[dict] = data.get("fight_stats", [])

    log.info(
        "Ingesting: %d fighters, %d fights, %d fight_stats rows -> %s",
        len(fighters), len(fights), len(fight_stats), db_path,
    )

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    _create_schema(conn)
    cur = conn.cursor()

    # ── Fighters ──────────────────────────────────────────────────────────────
    log.info("Upserting fighters…")
    cur.executemany(
        """
        INSERT INTO fighters (fighter_id, name, height, reach, stance, dob)
        VALUES (:fighter_id, :name, :height, :reach, :stance, :dob)
        ON CONFLICT(fighter_id) DO UPDATE SET
            name   = excluded.name,
            height = COALESCE(excluded.height, fighters.height),
            reach  = COALESCE(excluded.reach,  fighters.reach),
            stance = COALESCE(excluded.stance, fighters.stance),
            dob    = COALESCE(excluded.dob,    fighters.dob)
        """,
        fighters,
    )
    conn.commit()

    # Build fighter bio lookup for denormalisation into fight_stats.
    cur.execute("SELECT fighter_id, height, reach, stance, dob FROM fighters")
    bio: dict[str, dict] = {
        row[0]: {"height": row[1], "reach": row[2], "stance": row[3], "dob": row[4]}
        for row in cur.fetchall()
    }

    # ── Fights ────────────────────────────────────────────────────────────────
    log.info("Upserting fights…")
    fights_rows = []
    for f in fights:
        div = (f.get("division") or "").lower()
        gender = "female" if div in _WOMEN_DIVS else "male"
        no_of_rounds = 5 if f.get("title_fight") else 3
        match_time_sec = _round_time_secs(f.get("finish_round_time", "0:00"))
        fights_rows.append({
            "fight_id":      f["fight_id"],
            "event_id":      f.get("event_id"),
            "date":          f["date"],
            "division":      div,
            "r_fighter_id":  f["r_fighter_id"],
            "b_fighter_id":  f["b_fighter_id"],
            "winner_id":     f.get("winner_id"),
            "method":        f.get("method"),
            "title_fight":   int(f.get("title_fight", 0)),
            "odds_red":      f.get("odds_red"),
            "odds_blue":     f.get("odds_blue"),
            "finish_round":  f.get("finish_round"),
            "match_time_sec": match_time_sec,
            "no_of_rounds":  no_of_rounds,
            "gender":        gender,
        })

    cur.executemany(
        """
        INSERT INTO fights (
            fight_id, event_id, date, division, r_fighter_id, b_fighter_id,
            winner_id, method, title_fight, odds_red, odds_blue,
            finish_round, match_time_sec, no_of_rounds, gender
        ) VALUES (
            :fight_id, :event_id, :date, :division, :r_fighter_id, :b_fighter_id,
            :winner_id, :method, :title_fight, :odds_red, :odds_blue,
            :finish_round, :match_time_sec, :no_of_rounds, :gender
        )
        ON CONFLICT(fight_id) DO UPDATE SET
            winner_id     = COALESCE(excluded.winner_id, fights.winner_id),
            odds_red      = COALESCE(excluded.odds_red,  fights.odds_red),
            odds_blue     = COALESCE(excluded.odds_blue, fights.odds_blue)
        """,
        fights_rows,
    )
    conn.commit()

    # Build a division lookup per fight for weight derivation.
    cur.execute("SELECT fight_id, division from fights")
    fight_division: dict[str, str] = {row[0]: row[1] for row in cur.fetchall()}

    # ── Fight stats ───────────────────────────────────────────────────────────
    log.info("Upserting fight_stats…")
    stats_rows = []
    for s in fight_stats:
        fid  = s["fighter_id"]
        b    = bio.get(fid, {})
        div  = fight_division.get(s["fight_id"], "")
        stats_rows.append({
            "fight_id":          s["fight_id"],
            "fighter_id":        fid,
            "corner":            s["corner"],
            "kd":                s.get("kd", 0),
            "sig_str_landed":    s.get("sig_str_landed", 0),
            "sig_str_atmpted":   s.get("sig_str_atmpted", 0),
            "total_str_landed":  s.get("total_str_landed", 0),
            "total_str_atmpted": s.get("total_str_atmpted", 0),
            "td_landed":         s.get("td_landed", 0),
            "td_atmpted":        s.get("td_atmpted", 0),
            "sub_att":           s.get("sub_att", 0),
            "ctrl":              s.get("ctrl", 0),
            "head_landed":       s.get("head_landed", 0),
            "head_atmpted":      s.get("head_atmpted", 0),
            "body_landed":       s.get("body_landed", 0),
            "body_atmpted":      s.get("body_atmpted", 0),
            "leg_landed":        s.get("leg_landed", 0),
            "leg_atmpted":       s.get("leg_atmpted", 0),
            "dist_landed":       s.get("dist_landed", 0),
            "dist_atmpted":      s.get("dist_atmpted", 0),
            "clinch_landed":     s.get("clinch_landed", 0),
            "clinch_atmpted":    s.get("clinch_atmpted", 0),
            "ground_landed":     s.get("ground_landed", 0),
            "ground_atmpted":    s.get("ground_atmpted", 0),
            "total_fight_time":  s.get("total_fight_time", 0),
            "height":            b.get("height"),
            "reach":             b.get("reach"),
            "stance":            b.get("stance"),
            "dob":               b.get("dob"),
            "weight":            _DIVISION_WEIGHT.get(div),
        })

    cur.executemany(
        """
        INSERT INTO fight_stats (
            fight_id, fighter_id, corner,
            kd, sig_str_landed, sig_str_atmpted, total_str_landed, total_str_atmpted,
            td_landed, td_atmpted, sub_att, ctrl,
            head_landed, head_atmpted, body_landed, body_atmpted,
            leg_landed, leg_atmpted, dist_landed, dist_atmpted,
            clinch_landed, clinch_atmpted, ground_landed, ground_atmpted,
            total_fight_time, height, reach, stance, dob, weight
        ) VALUES (
            :fight_id, :fighter_id, :corner,
            :kd, :sig_str_landed, :sig_str_atmpted, :total_str_landed, :total_str_atmpted,
            :td_landed, :td_atmpted, :sub_att, :ctrl,
            :head_landed, :head_atmpted, :body_landed, :body_atmpted,
            :leg_landed, :leg_atmpted, :dist_landed, :dist_atmpted,
            :clinch_landed, :clinch_atmpted, :ground_landed, :ground_atmpted,
            :total_fight_time, :height, :reach, :stance, :dob, :weight
        )
        ON CONFLICT(fight_id, fighter_id) DO UPDATE SET
            kd                = excluded.kd,
            sig_str_landed    = excluded.sig_str_landed,
            sig_str_atmpted   = excluded.sig_str_atmpted,
            total_str_landed  = excluded.total_str_landed,
            total_str_atmpted = excluded.total_str_atmpted,
            td_landed         = excluded.td_landed,
            td_atmpted        = excluded.td_atmpted,
            sub_att           = excluded.sub_att,
            ctrl              = excluded.ctrl,
            head_landed       = excluded.head_landed,
            head_atmpted      = excluded.head_atmpted,
            body_landed       = excluded.body_landed,
            body_atmpted      = excluded.body_atmpted,
            leg_landed        = excluded.leg_landed,
            leg_atmpted       = excluded.leg_atmpted,
            dist_landed       = excluded.dist_landed,
            dist_atmpted      = excluded.dist_atmpted,
            clinch_landed     = excluded.clinch_landed,
            clinch_atmpted    = excluded.clinch_atmpted,
            ground_landed     = excluded.ground_landed,
            ground_atmpted    = excluded.ground_atmpted,
            total_fight_time  = excluded.total_fight_time,
            height            = COALESCE(excluded.height, fight_stats.height),
            reach             = COALESCE(excluded.reach,  fight_stats.reach),
            stance            = COALESCE(excluded.stance, fight_stats.stance),
            dob               = COALESCE(excluded.dob,    fight_stats.dob),
            weight            = COALESCE(excluded.weight, fight_stats.weight)
        """,
        stats_rows,
    )
    conn.commit()

    n_f  = cur.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
    n_fi = cur.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    n_s  = cur.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    date_min, date_max = cur.execute("SELECT MIN(date), MAX(date) FROM fights").fetchone()
    log.info("Ingest complete:")
    log.info("  fighters:    %d", n_f)
    log.info("  fights:      %d  (%s to %s)", n_fi, date_min, date_max)
    log.info("  fight_stats: %d", n_s)
    conn.close()
