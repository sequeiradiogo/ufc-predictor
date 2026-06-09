"""
ingest_mdabbert.py — Ingest the mdabbert-format UFC CSV into the project database.

Replaces pipeline steps 1-3 (raw_sql_database, keys, rolling) when the source
data uses mdabbert column names (R_fighter, B_fighter, R_avg_SIG_STR_landed …)
rather than the original UFCStats hex-ID format.

The mdabbert CSV already contains career-average stats going into each fight so
rolling.py is not needed.  After running this, continue with steps 4-7:
    python run_pipeline.py --steps 4,5,6,7

Usage:
    python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv
"""

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from config import DB_PATH
    from logger import get_logger
except ImportError:
    DB_PATH = Path(__file__).resolve().parent / "ufc_v2.db"
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

log = get_logger(__name__)


# ── ID generation ─────────────────────────────────────────────────────────────

def _fid(name: str) -> str:
    """Deterministic 16-char hex ID from a fighter name."""
    return hashlib.md5(name.lower().strip().encode()).hexdigest()[:16]


def _fight_id(r_name: str, b_name: str, date: str) -> str:
    """Deterministic 16-char hex ID from the fight key."""
    key = f"{r_name.lower().strip()}|{b_name.lower().strip()}|{date}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ── Finish method mapping ─────────────────────────────────────────────────────

_FINISH_MAP = {
    "KO/TKO": "KO/TKO",
    "SUB":     "Submission",
    "U-DEC":   "Decision - Unanimous",
    "S-DEC":   "Decision - Split",
    "M-DEC":   "Decision - Majority",
    "DQ":      "Decision - Unanimous",
    "CNC":     "Could Not Continue",
}


def _method(finish: str) -> str | None:
    return _FINISH_MAP.get(str(finish).strip(), None)


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest(csv_path: Path, db_path: Path) -> None:
    log.info("Loading %s…", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    log.info("CSV loaded: %d rows x %d columns.", *df.shape)

    # ── Generate IDs ──────────────────────────────────────────────────────────
    df["r_fighter_id"] = df["R_fighter"].apply(_fid)
    df["b_fighter_id"] = df["B_fighter"].apply(_fid)
    df["fight_id"]     = df.apply(
        lambda row: _fight_id(row["R_fighter"], row["B_fighter"], row["date"]), axis=1
    )

    # winner_id: None for draws / NCs (filtered out during ML prep)
    def _winner_id(row) -> str | None:
        if row["Winner"] == "Red":
            return row["r_fighter_id"]
        if row["Winner"] == "Blue":
            return row["b_fighter_id"]
        return None

    df["winner_id"] = df.apply(_winner_id, axis=1)
    df["method"]    = df["finish"].apply(_method)

    log.info("IDs generated. Connecting to %s…", db_path)
    conn = sqlite3.connect(str(db_path))

    # ── fighters ──────────────────────────────────────────────────────────────
    log.info("Building fighters table…")
    r_fighters = df[["r_fighter_id", "R_fighter", "R_Height_cms", "R_Reach_cms", "R_Stance"]].rename(
        columns={
            "r_fighter_id": "fighter_id",
            "R_fighter":    "name",
            "R_Height_cms": "height",
            "R_Reach_cms":  "reach",
            "R_Stance":     "stance",
        }
    )
    b_fighters = df[["b_fighter_id", "B_fighter", "B_Height_cms", "B_Reach_cms", "B_Stance"]].rename(
        columns={
            "b_fighter_id": "fighter_id",
            "B_fighter":    "name",
            "B_Height_cms": "height",
            "B_Reach_cms":  "reach",
            "B_Stance":     "stance",
        }
    )
    fighters = (
        pd.concat([r_fighters, b_fighters], ignore_index=True)
        .drop_duplicates(subset=["fighter_id"])
        .reset_index(drop=True)
    )
    fighters.to_sql("fighters", conn, if_exists="replace", index=False)
    log.info("Inserted %d fighters.", len(fighters))

    # ── fights ────────────────────────────────────────────────────────────────
    log.info("Building fights table…")

    # Pre-computed fight-level features (from add_computed_features_to_csv.py)
    _DIV_COLS = [
        c for c in df.columns
        if c.startswith("div_") or c in (
            "grapple_ratio_diff", "striker_vs_wrestler", "wrestler_vs_striker",
            "southpaw_adv_diff", "both_southpaw", "weightclass_rank_diff",
        )
    ]

    fights = df[[
        "fight_id", "date", "r_fighter_id", "b_fighter_id",
        "winner_id", "method", "weight_class", "title_bout",
        "R_odds", "B_odds", "finish_round", "total_fight_time_secs",
        "no_of_rounds", "gender",
    ] + _DIV_COLS].rename(columns={
        "weight_class":          "division",
        "title_bout":            "title_fight",
        "R_odds":                "odds_red",
        "B_odds":                "odds_blue",
        "total_fight_time_secs": "match_time_sec",
    }).copy()

    fights["division"]    = fights["division"].str.lower()
    fights["title_fight"] = fights["title_fight"].astype(int)
    fights.to_sql("fights", conn, if_exists="replace", index=False)
    log.info("Inserted %d fights.", len(fights))

    # ── fight_stats ───────────────────────────────────────────────────────────
    log.info("Building fight_stats table…")

    # Column mapping: (mdabbert_prefix) -> (our column name)
    # These are career averages going INTO each fight (pre-fight stats).
    stat_map = {
        "wins":                  ("R_wins",                "B_wins"),
        "losses":                ("R_losses",              "B_losses"),
        # Prefixed 'career_' to avoid collision with compute_recent_form's win_streak column
        "career_win_streak":     ("R_current_win_streak",  "B_current_win_streak"),
        "career_lose_streak":    ("R_current_lose_streak", "B_current_lose_streak"),
        "longest_win_streak":    ("R_longest_win_streak",  "B_longest_win_streak"),
        "total_rounds_fought":  ("R_total_rounds_fought","B_total_rounds_fought"),
        "total_title_bouts":    ("R_total_title_bouts",  "B_total_title_bouts"),
        "avg_sig_str_pct":      ("R_avg_SIG_STR_pct",    "B_avg_SIG_STR_pct"),
        "avg_sub_att":          ("R_avg_SUB_ATT",        "B_avg_SUB_ATT"),
        "avg_td_pct":           ("R_avg_TD_pct",         "B_avg_TD_pct"),
        "height":               ("R_Height_cms",         "B_Height_cms"),
        "reach":                ("R_Reach_cms",          "B_Reach_cms"),
        "stance":               ("R_Stance",             "B_Stance"),
        "age":                  ("R_age",                "B_age"),
        # used by add_style_features (splm = strikes/min proxy, td_avg = td/fight proxy)
        "splm":                 ("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
        "td_avg":               ("R_avg_TD_landed",      "B_avg_TD_landed"),
        # ranking at time of fight
        "weightclass_rank":     ("R_match_weightclass_rank", "B_match_weightclass_rank"),
        # win-by-method breakdown
        "win_by_ko":            ("R_win_by_KO/TKO",      "B_win_by_KO/TKO"),
        "win_by_sub":           ("R_win_by_Submission",  "B_win_by_Submission"),
        "win_by_dec_unanimous": ("R_win_by_Decision_Unanimous", "B_win_by_Decision_Unanimous"),
        "win_by_dec_split":     ("R_win_by_Decision_Split",     "B_win_by_Decision_Split"),
        # Defensive metrics (from UFCStats, now stored in CSV)
        "sapm":    ("R_sapm",    "B_sapm"),
        "str_def": ("R_str_def", "B_str_def"),
        "td_def":  ("R_td_def",  "B_td_def"),
        # Pre-computed features (from add_computed_features_to_csv.py)
        "elo":               ("R_elo",               "B_elo"),
        "glicko":            ("R_glicko",            "B_glicko"),
        "glicko_rd":         ("R_glicko_rd",         "B_glicko_rd"),
        "recent_win_rate":   ("R_recent_win_rate",   "B_recent_win_rate"),
        "recent_finish_rate":("R_recent_finish_rate","B_recent_finish_rate"),
        "sos":               ("R_sos",               "B_sos"),
        "str_acc_slope":     ("R_str_acc_slope",     "B_str_acc_slope"),
        "splm_slope":        ("R_splm_slope",        "B_splm_slope"),
        "td_acc_slope":      ("R_td_acc_slope",      "B_td_acc_slope"),
        "ko_rate":           ("R_ko_rate",           "B_ko_rate"),
        "sub_rate":          ("R_sub_rate",          "B_sub_rate"),
        "dec_rate":          ("R_dec_rate",          "B_dec_rate"),
        "days_since_last":   ("R_days_since_last",   "B_days_since_last"),
        "ko_vuln":           ("R_ko_vuln",           "B_ko_vuln"),
    }

    r_stats = df[["fight_id", "r_fighter_id"]].rename(columns={"r_fighter_id": "fighter_id"}).copy()
    b_stats = df[["fight_id", "b_fighter_id"]].rename(columns={"b_fighter_id": "fighter_id"}).copy()
    r_stats["corner"] = "r"
    b_stats["corner"] = "b"

    for col, (r_src, b_src) in stat_map.items():
        if r_src in df.columns:
            r_stats[col] = df[r_src].values
        if b_src in df.columns:
            b_stats[col] = df[b_src].values

    # total_fight_time: use wins+losses as a proxy (0 for debutants → is_debutant_diff works)
    r_stats["total_fight_time"] = (df["R_wins"].fillna(0) + df["R_losses"].fillna(0)).values
    b_stats["total_fight_time"] = (df["B_wins"].fillna(0) + df["B_losses"].fillna(0)).values

    stats_df = pd.concat([r_stats, b_stats], ignore_index=True)
    stats_df.to_sql("fight_stats", conn, if_exists="replace", index=False)
    log.info("Inserted %d fight_stats rows.", len(stats_df))

    # ── indexes ───────────────────────────────────────────────────────────────
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_date      ON fights(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fights_rfighter  ON fights(r_fighter_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fight ON fight_stats(fight_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fightstats_fid   ON fight_stats(fighter_id);")
    conn.commit()

    # ── summary ───────────────────────────────────────────────────────────────
    n_f  = cur.execute("SELECT COUNT(*) FROM fighters").fetchone()[0]
    n_fi = cur.execute("SELECT COUNT(*) FROM fights").fetchone()[0]
    n_s  = cur.execute("SELECT COUNT(*) FROM fight_stats").fetchone()[0]
    date_min, date_max = cur.execute("SELECT MIN(date), MAX(date) FROM fights").fetchone()
    log.info("Database complete:")
    log.info("  fighters:   %d", n_f)
    log.info("  fights:     %d  (%s to %s)", n_fi, date_min, date_max)
    log.info("  fight_stats: %d", n_s)
    conn.close()
    log.info("Saved to %s", db_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest mdabbert-format UFC CSV into the project database.",
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Path to mdabbert-format CSV (e.g. raw_data/ufc-master.csv)",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"Output SQLite database path (default: {DB_PATH})",
    )
    args = parser.parse_args()
    ingest(args.csv, args.db)


if __name__ == "__main__":
    main()
