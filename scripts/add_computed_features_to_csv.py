"""
scripts/add_computed_features_to_csv.py

Pre-compute ELO, Glicko-2, recent form, SOS, slope features, finish rates,
inactivity, and KO vulnerability for every fight and write them into ufc-master.csv.

These are all deterministic per-fight values (no leakage -- each uses shift(1)
or fights before that date). Storing them in the CSV makes the pipeline
self-contained and simplifies ML_data_preparation_v1.py to pure diff-building.

Run after ingest_mdabbert.py rebuilds the DB:
    python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv
    python scripts/add_computed_features_to_csv.py
    python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv   (re-ingest with new cols)

Usage:
    python scripts/add_computed_features_to_csv.py
    python scripts/add_computed_features_to_csv.py --dry-run
"""
import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DB_V1_PATH, DB_PATH, EWMA_SPAN
from ml.ELO_calculator import build_elo_features, build_glicko_features
from ml.ML_data_preparation import (
    compute_finish_rates,
    compute_inactivity,
    compute_ko_vulnerability,
    compute_sos_features,
)
from ml.ML_data_preparation_v1 import compute_slope_features_v1, _compute_recent_form_v1

KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"

# Maps (wide column) -> CSV column name
_FIGHT_COLS = {
    "elo_red":    "R_elo",
    "elo_blue":   "B_elo",
    "glicko_red":    "R_glicko",
    "glicko_blue":   "B_glicko",
    "glicko_rd_red":  "R_glicko_rd",
    "glicko_rd_blue": "B_glicko_rd",
}

# Per-fighter features: (db_col, red_csv_col, blue_csv_col)
_PER_FIGHTER_COLS = [
    ("recent_win_rate",    "R_recent_win_rate",    "B_recent_win_rate"),
    ("recent_finish_rate", "R_recent_finish_rate", "B_recent_finish_rate"),
    ("sos",                "R_sos",                "B_sos"),
    ("str_acc_slope",      "R_str_acc_slope",      "B_str_acc_slope"),
    ("splm_slope",         "R_splm_slope",         "B_splm_slope"),
    ("td_acc_slope",       "R_td_acc_slope",       "B_td_acc_slope"),
    ("ko_rate",            "R_ko_rate",            "B_ko_rate"),
    ("sub_rate",           "R_sub_rate",           "B_sub_rate"),
    ("dec_rate",           "R_dec_rate",           "B_dec_rate"),
    ("days_since_last",    "R_days_since_last",     "B_days_since_last"),
    ("ko_vuln",            "R_ko_vuln",            "B_ko_vuln"),
    ("kd_received",        "R_kd_received",        "B_kd_received"),
]


def _fight_id(r_name: str, b_name: str, date: str) -> str:
    key = f"{r_name.lower().strip()}|{b_name.lower().strip()}|{date}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _assign_per_fighter(
    base: pd.DataFrame,
    feat_df: pd.DataFrame,
    db_col: str,
    red_col: str,
    blue_col: str,
    r_fid: str = "r_fighter_id",
    b_fid: str = "b_fighter_id",
) -> pd.DataFrame:
    """Merge a per-fighter feature into the base DataFrame for red and blue corners."""
    red = feat_df[["fight_id", "fighter_id", db_col]].rename(columns={db_col: red_col})
    blue = feat_df[["fight_id", "fighter_id", db_col]].rename(columns={db_col: blue_col})
    base = base.merge(red,  left_on=["fight_id", r_fid], right_on=["fight_id", "fighter_id"], how="left").drop(columns=["fighter_id"])
    base = base.merge(blue, left_on=["fight_id", b_fid], right_on=["fight_id", "fighter_id"], how="left").drop(columns=["fighter_id"])
    return base


def main(dry_run: bool = False) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    df["_fight_id"] = df.apply(
        lambda r: _fight_id(r["R_fighter"], r["B_fighter"], str(r["date"])[:10]),
        axis=1,
    )
    print(f"  {len(df)} rows, fight_id computed.")

    print(f"Computing features from mdabbert DB ({DB_V1_PATH.name}) ...")
    conn = sqlite3.connect(str(DB_V1_PATH))

    fights_meta = pd.read_sql_query(
        "SELECT fight_id, date, r_fighter_id, b_fighter_id FROM fights ORDER BY date ASC",
        conn,
    )
    fights_meta["date"] = pd.to_datetime(fights_meta["date"])

    # ── ELO ───────────────────────────────────────────────────────────────────
    print("  ELO ...")
    elo_df = build_elo_features(conn)

    # ── Glicko-2 ──────────────────────────────────────────────────────────────
    print("  Glicko-2 ...")
    glicko_df = build_glicko_features(conn)

    # ── SOS (needs ELO) ───────────────────────────────────────────────────────
    print("  SOS ...")
    elo_for_sos = fights_meta.merge(elo_df, on="fight_id", how="left")
    sos_df = compute_sos_features(
        elo_for_sos[["fight_id", "date", "r_fighter_id", "b_fighter_id", "elo_red", "elo_blue"]]
    )

    # ── Recent form ───────────────────────────────────────────────────────────
    print("  Recent form ...")
    form_df = _compute_recent_form_v1(conn)

    # ── Finish rates ──────────────────────────────────────────────────────────
    print("  Finish rates ...")
    fin_df = compute_finish_rates(conn)

    # ── Inactivity ────────────────────────────────────────────────────────────
    print("  Inactivity ...")
    inact_df = compute_inactivity(conn)

    # ── KO vulnerability + kd received (needs UFCStats DB for per-fight kd) ───
    print("  KO vulnerability ...")
    ufc_conn = sqlite3.connect(str(DB_PATH))
    kovuln_raw = compute_ko_vulnerability(ufc_conn)

    # Map UFCStats (fight_id, fighter_id) -> mdabbert (fight_id, fighter_id)
    ufc_fighters = pd.read_sql_query(
        "SELECT fighter_id, name FROM fighters", ufc_conn
    )
    ufc_fights = pd.read_sql_query(
        "SELECT f.fight_id, f.date, r.name r_name, b.name b_name "
        "FROM fights f "
        "JOIN fighters r ON f.r_fighter_id = r.fighter_id "
        "JOIN fighters b ON f.b_fighter_id = b.fighter_id",
        ufc_conn,
    )

    # Item 8: EWMA striking/TD accuracy -- requires per-fight raw counts from UFCStats
    print("  EWMA accuracy ...")
    _ewma_raw = pd.read_sql_query(
        """
        SELECT fs.fighter_id, f.fight_id, f.date,
               CAST(fs.sig_str_landed  AS REAL) AS str_land,
               CAST(fs.sig_str_atmpted AS REAL) AS str_att,
               CAST(fs.td_landed       AS REAL) AS td_land,
               CAST(fs.td_atmpted      AS REAL) AS td_att,
               CAST(fs.head_landed     AS REAL) AS head_land,
               CAST(fs.head_atmpted    AS REAL) AS head_att,
               CAST(fs.body_landed     AS REAL) AS body_land,
               CAST(fs.body_atmpted    AS REAL) AS body_att,
               CAST(fs.dist_landed     AS REAL) AS dist_land,
               CAST(fs.dist_atmpted    AS REAL) AS dist_att
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        ufc_conn,
    )

    # Opponent-adjusted SPLM, TD avg, and zone accuracy: normalize offensive
    # stats by the defensive quality of prior opponents.
    # opp.str_def/td_def/zone_def are pre-fight rolling values (rolling.py shift(1) applied).
    print("  Opponent-adjusted stats ...")
    _oadj_raw = pd.read_sql_query(
        """
        SELECT
            fs.fighter_id,
            f.fight_id,
            f.date,
            CAST(opp.str_def    AS REAL) AS opp_str_def,
            CAST(opp.td_def     AS REAL) AS opp_td_def,
            CAST(opp.head_def   AS REAL) AS opp_head_def,
            CAST(opp.body_def   AS REAL) AS opp_body_def,
            CAST(opp.dist_def   AS REAL) AS opp_dist_def,
            CAST(fs.splm        AS REAL) AS own_splm,
            CAST(fs.td_avg      AS REAL) AS own_td_avg,
            CAST(fs.head_acc    AS REAL) AS own_head_acc,
            CAST(fs.body_acc    AS REAL) AS own_body_acc,
            CAST(fs.dist_acc    AS REAL) AS own_dist_acc
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        JOIN fight_stats opp
            ON f.fight_id = opp.fight_id
            AND opp.fighter_id != fs.fighter_id
        ORDER BY fs.fighter_id, f.date ASC, f.fight_id ASC
        """,
        ufc_conn,
    )

    import hashlib as _hl
    def _md5id(s: str) -> str:
        return _hl.md5(s.lower().strip().encode()).hexdigest()[:16]

    ufc_fid_to_name = dict(zip(ufc_fighters["fighter_id"], ufc_fighters["name"]))
    ufc_fights["mab_fight_id"] = ufc_fights.apply(
        lambda r: _fight_id(r["r_name"], r["b_name"], str(r["date"])[:10]), axis=1
    )
    ufc_fight_map    = dict(zip(ufc_fights["fight_id"], ufc_fights["mab_fight_id"]))

    kovuln_df = kovuln_raw.copy()
    kovuln_df["fight_id"]   = kovuln_df["fight_id"].map(ufc_fight_map)
    kovuln_df["fighter_id"] = kovuln_df["fighter_id"].map(
        lambda fid: _md5id(ufc_fid_to_name[fid]) if fid in ufc_fid_to_name else None
    )
    kovuln_df = kovuln_df.dropna(subset=["fight_id", "fighter_id"])[
        ["fight_id", "fighter_id", "ko_vuln", "kd_received"]
    ]

    # Compute EWMA of per-fight striking/TD accuracy (shift(1) for leakage prevention)
    _eps = 1e-6
    _ewma_raw["date"] = pd.to_datetime(_ewma_raw["date"])
    _ewma_raw = _ewma_raw.sort_values(["fighter_id", "date", "fight_id"]).copy()

    _ewma_raw["pf_str_acc"]  = (_ewma_raw["str_land"]  / (_ewma_raw["str_att"]  + _eps)).where(_ewma_raw["str_att"]  > 0, 0.0)
    _ewma_raw["pf_td_acc"]   = (_ewma_raw["td_land"]   / (_ewma_raw["td_att"]   + _eps)).where(_ewma_raw["td_att"]   > 0, 0.0)
    _ewma_raw["pf_head_acc"] = (_ewma_raw["head_land"] / (_ewma_raw["head_att"] + _eps)).where(_ewma_raw["head_att"] > 0, 0.0)
    _ewma_raw["pf_body_acc"] = (_ewma_raw["body_land"] / (_ewma_raw["body_att"] + _eps)).where(_ewma_raw["body_att"] > 0, 0.0)
    _ewma_raw["pf_dist_acc"] = (_ewma_raw["dist_land"] / (_ewma_raw["dist_att"] + _eps)).where(_ewma_raw["dist_att"] > 0, 0.0)

    _grp = _ewma_raw.groupby("fighter_id")
    _ewma_acc_cols = [
        ("pf_str_acc",  "ewma_str_acc"),
        ("pf_td_acc",   "ewma_td_acc"),
        ("pf_head_acc", "ewma_head_acc"),
        ("pf_body_acc", "ewma_body_acc"),
        ("pf_dist_acc", "ewma_dist_acc"),
    ]
    for _pf_col, _out_col in _ewma_acc_cols:
        _ewma_raw[_out_col] = (
            _grp[_pf_col].transform(lambda s: s.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean())
            .fillna(0)
        )
    _ewma_raw["str_acc_var"] = (
        _grp["pf_str_acc"].transform(lambda s: s.shift(1).rolling(EWMA_SPAN, min_periods=2).std())
        .fillna(0)
    )
    _ewma_acc_out_cols = [c for _, c in _ewma_acc_cols] + ["str_acc_var"]
    ewma_computed = _ewma_raw[["fight_id", "fighter_id"] + _ewma_acc_out_cols].copy()
    ewma_df = ewma_computed.copy()
    ewma_df["fight_id"]   = ewma_df["fight_id"].map(ufc_fight_map)
    ewma_df["fighter_id"] = ewma_df["fighter_id"].map(
        lambda fid: _md5id(ufc_fid_to_name[fid]) if fid in ufc_fid_to_name else None
    )
    ewma_df = ewma_df.dropna(subset=["fight_id", "fighter_id"])[
        ["fight_id", "fighter_id"] + _ewma_acc_out_cols
    ]

    # --- Opponent-adjusted stats ---
    # Zero def values mean no prior data; replace with NaN before averaging
    # so first-fight opponents don't pull down the quality estimate.
    _oadj_def_cols = ["opp_str_def", "opp_td_def", "opp_head_def", "opp_body_def", "opp_dist_def"]
    for _dc in _oadj_def_cols:
        _oadj_raw[_dc] = _oadj_raw[_dc].where(_oadj_raw[_dc] > 0)

    _league_str_def  = _oadj_raw["opp_str_def"].mean()
    _league_td_def   = _oadj_raw["opp_td_def"].mean()
    _league_head_def = _oadj_raw["opp_head_def"].mean()
    _league_body_def = _oadj_raw["opp_body_def"].mean()
    _league_dist_def = _oadj_raw["opp_dist_def"].mean()
    _league_str_allowed  = 1.0 - _league_str_def  / 100.0
    _league_td_allowed   = 1.0 - _league_td_def   / 100.0
    _league_head_allowed = 1.0 - _league_head_def  / 100.0
    _league_body_allowed = 1.0 - _league_body_def  / 100.0
    _league_dist_allowed = 1.0 - _league_dist_def  / 100.0

    _oadj_raw = _oadj_raw.sort_values(["fighter_id", "date", "fight_id"]).copy()
    _oadj_grp = _oadj_raw.groupby("fighter_id")

    # shift(1) so fight i uses opponents from fights 0..i-1 (leakage-free)
    _oadj_avg_map = {
        "opp_str_def":  (_league_str_def,  "avg_opp_str_def"),
        "opp_td_def":   (_league_td_def,   "avg_opp_td_def"),
        "opp_head_def": (_league_head_def,  "avg_opp_head_def"),
        "opp_body_def": (_league_body_def,  "avg_opp_body_def"),
        "opp_dist_def": (_league_dist_def,  "avg_opp_dist_def"),
    }
    for _src_col, (_fill, _avg_col) in _oadj_avg_map.items():
        _shifted = _oadj_grp[_src_col].shift(1)
        _oadj_raw[_avg_col] = (
            _oadj_grp[_src_col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
            .fillna(_fill)
        )

    # opp_adj = own_stat * (league_avg_allowed / this_fighter_avg_opp_allowed)
    def _allowed(avg_col):
        return (1.0 - _oadj_raw[avg_col] / 100.0).clip(lower=0.05)

    _oadj_raw["opp_adj_splm"]     = (_oadj_raw["own_splm"]    * (_league_str_allowed  / _allowed("avg_opp_str_def"))).fillna(0)
    _oadj_raw["opp_adj_td_avg"]   = (_oadj_raw["own_td_avg"]  * (_league_td_allowed   / _allowed("avg_opp_td_def"))).fillna(0)
    _oadj_raw["opp_adj_head_acc"] = (_oadj_raw["own_head_acc"] * (_league_head_allowed / _allowed("avg_opp_head_def"))).fillna(0)
    _oadj_raw["opp_adj_body_acc"] = (_oadj_raw["own_body_acc"] * (_league_body_allowed / _allowed("avg_opp_body_def"))).fillna(0)
    _oadj_raw["opp_adj_dist_acc"] = (_oadj_raw["own_dist_acc"] * (_league_dist_allowed / _allowed("avg_opp_dist_def"))).fillna(0)

    _oadj_out_cols = ["opp_adj_splm", "opp_adj_td_avg", "opp_adj_head_acc", "opp_adj_body_acc", "opp_adj_dist_acc"]
    oadj_df = _oadj_raw[["fight_id", "fighter_id"] + _oadj_out_cols].copy()
    oadj_df["fight_id"]   = oadj_df["fight_id"].map(ufc_fight_map)
    oadj_df["fighter_id"] = oadj_df["fighter_id"].map(
        lambda fid: _md5id(ufc_fid_to_name[fid]) if fid in ufc_fid_to_name else None
    )
    oadj_df = oadj_df.dropna(subset=["fight_id", "fighter_id"])[
        ["fight_id", "fighter_id"] + _oadj_out_cols
    ]

    # --- EWMA of per-fight output rates (splm, td_avg, sapm) ---
    print("  EWMA output rates (splm, td_avg, sapm) ...")
    _ewma_rates_raw = pd.read_sql_query(
        """
        SELECT fs.fighter_id, f.fight_id, f.date,
               CAST(fs.sig_str_landed  AS REAL) AS str_land,
               CAST(fs.td_landed       AS REAL) AS td_land,
               CAST(fs.clinch_landed   AS REAL) AS clinch_land,
               CAST(fs.sub_att         AS REAL) AS sub_att,
               CAST(opp.sig_str_landed AS REAL) AS opp_str_land,
               CAST(opp.kd             AS REAL) AS opp_kd,
               CAST(f.match_time_sec   AS REAL) AS match_sec,
               CAST(f.finish_round     AS REAL) AS finish_round
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        JOIN fight_stats opp
            ON f.fight_id = opp.fight_id
            AND opp.fighter_id != fs.fighter_id
        ORDER BY f.date ASC, f.fight_id ASC
        """,
        ufc_conn,
    )
    ufc_conn.close()

    _ewma_rates_raw["date"] = pd.to_datetime(_ewma_rates_raw["date"])
    _ewma_rates_raw = _ewma_rates_raw.sort_values(["fighter_id", "date", "fight_id"]).copy()

    _fight_secs = (
        _ewma_rates_raw["match_sec"].fillna(0) +
        (_ewma_rates_raw["finish_round"].fillna(1) - 1) * 300
    )
    _fight_min = (_fight_secs / 60.0).clip(lower=1e-6)
    _has_time = _fight_secs > 0

    _ewma_rates_raw["pf_splm"]        = (_ewma_rates_raw["str_land"]     / _fight_min).where(_has_time, 0.0)
    _ewma_rates_raw["pf_td_avg"]      = (_ewma_rates_raw["td_land"] * 900.0 / _fight_secs.clip(lower=1e-6)).where(_has_time, 0.0)
    _ewma_rates_raw["pf_sapm"]        = (_ewma_rates_raw["opp_str_land"] / _fight_min).where(_has_time, 0.0)
    _ewma_rates_raw["pf_clinch_per"]  = (_ewma_rates_raw["clinch_land"]  / _fight_min).where(_has_time, 0.0)
    _ewma_rates_raw["pf_sub_att"]     = (_ewma_rates_raw["sub_att"] * 900.0 / _fight_secs.clip(lower=1e-6)).where(_has_time, 0.0)
    _ewma_rates_raw["pf_kd_received"] = (_ewma_rates_raw["opp_kd"].fillna(0) / _fight_min).where(_has_time, 0.0)

    _rates_grp = _ewma_rates_raw.groupby("fighter_id")
    _rate_cols = [
        ("pf_splm",        "ewma_splm"),
        ("pf_td_avg",      "ewma_td_avg"),
        ("pf_sapm",        "ewma_sapm"),
        ("pf_clinch_per",  "ewma_clinch_per"),
        ("pf_sub_att",     "ewma_sub_att"),
        ("pf_kd_received", "ewma_kd_received"),
    ]
    for _pf_col, _out_col in _rate_cols:
        _ewma_rates_raw[_out_col] = (
            _rates_grp[_pf_col].transform(lambda s: s.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean())
            .fillna(0)
        )

    _rate_out_cols = [c for _, c in _rate_cols]
    ewma_rates_df = _ewma_rates_raw[["fight_id", "fighter_id"] + _rate_out_cols].copy()
    ewma_rates_df["fight_id"]   = ewma_rates_df["fight_id"].map(ufc_fight_map)
    ewma_rates_df["fighter_id"] = ewma_rates_df["fighter_id"].map(
        lambda fid: _md5id(ufc_fid_to_name[fid]) if fid in ufc_fid_to_name else None
    )
    ewma_rates_df = ewma_rates_df.dropna(subset=["fight_id", "fighter_id"])[
        ["fight_id", "fighter_id"] + _rate_out_cols
    ]

    # ── Slope features ────────────────────────────────────────────────────────
    print("  Slope features ...")
    slope_df = compute_slope_features_v1(conn)

    conn.close()

    # ── Build wide table: fight_id + all per-fighter features ─────────────────
    wide = fights_meta[["fight_id", "r_fighter_id", "b_fighter_id"]].copy()

    wide = wide.merge(elo_df,    on="fight_id", how="left")
    wide = wide.merge(glicko_df, on="fight_id", how="left")

    per_fighter_data = [
        (form_df,      "recent_win_rate",    "R_recent_win_rate",    "B_recent_win_rate"),
        (form_df,      "recent_finish_rate", "R_recent_finish_rate", "B_recent_finish_rate"),
        (sos_df,       "sos",                "R_sos",                "B_sos"),
        (slope_df,     "str_acc_slope",      "R_str_acc_slope",      "B_str_acc_slope"),
        (slope_df,     "splm_slope",         "R_splm_slope",         "B_splm_slope"),
        (slope_df,     "td_acc_slope",       "R_td_acc_slope",       "B_td_acc_slope"),
        (fin_df,       "ko_rate",            "R_ko_rate",            "B_ko_rate"),
        (fin_df,       "sub_rate",           "R_sub_rate",           "B_sub_rate"),
        (fin_df,       "dec_rate",           "R_dec_rate",           "B_dec_rate"),
        (inact_df,     "days_since_last",    "R_days_since_last",    "B_days_since_last"),
        (kovuln_df,    "ko_vuln",            "R_ko_vuln",            "B_ko_vuln"),
        (kovuln_df,    "kd_received",        "R_kd_received",        "B_kd_received"),
        (ewma_df,      "ewma_str_acc",       "R_ewma_str_acc",       "B_ewma_str_acc"),
        (ewma_df,      "ewma_td_acc",        "R_ewma_td_acc",        "B_ewma_td_acc"),
        (ewma_df,      "ewma_head_acc",      "R_ewma_head_acc",      "B_ewma_head_acc"),
        (ewma_df,      "ewma_body_acc",      "R_ewma_body_acc",      "B_ewma_body_acc"),
        (ewma_df,      "ewma_dist_acc",      "R_ewma_dist_acc",      "B_ewma_dist_acc"),
        (ewma_df,      "str_acc_var",        "R_str_acc_var",        "B_str_acc_var"),
        (oadj_df,          "opp_adj_splm",       "R_opp_adj_splm",       "B_opp_adj_splm"),
        (oadj_df,          "opp_adj_td_avg",     "R_opp_adj_td_avg",     "B_opp_adj_td_avg"),
        (oadj_df,          "opp_adj_head_acc",   "R_opp_adj_head_acc",   "B_opp_adj_head_acc"),
        (oadj_df,          "opp_adj_body_acc",   "R_opp_adj_body_acc",   "B_opp_adj_body_acc"),
        (oadj_df,          "opp_adj_dist_acc",   "R_opp_adj_dist_acc",   "B_opp_adj_dist_acc"),
        (ewma_rates_df,    "ewma_splm",          "R_ewma_splm",          "B_ewma_splm"),
        (ewma_rates_df,    "ewma_td_avg",        "R_ewma_td_avg",        "B_ewma_td_avg"),
        (ewma_rates_df,    "ewma_sapm",          "R_ewma_sapm",          "B_ewma_sapm"),
        (ewma_rates_df,    "ewma_clinch_per",    "R_ewma_clinch_per",    "B_ewma_clinch_per"),
        (ewma_rates_df,    "ewma_sub_att",       "R_ewma_sub_att",       "B_ewma_sub_att"),
        (ewma_rates_df,    "ewma_kd_received",   "R_ewma_kd_received",   "B_ewma_kd_received"),
    ]

    for feat_df, db_col, red_col, blue_col in per_fighter_data:
        wide = _assign_per_fighter(wide, feat_df, db_col, red_col, blue_col)

    # Rename fight-level ELO/Glicko columns to CSV convention
    wide = wide.rename(columns=_FIGHT_COLS)

    # ── Build unique list of all new per-fighter columns ──────────────────────
    seen: set[str] = set()
    csv_feature_cols: list[str] = []
    for _, _, rc, bc in per_fighter_data:
        for c in (rc, bc):
            if c not in seen:
                csv_feature_cols.append(c)
                seen.add(c)
    all_new_cols = list(_FIGHT_COLS.values()) + csv_feature_cols

    wide_slim = wide[["fight_id"] + all_new_cols].copy()
    num_cols = wide_slim.select_dtypes(include="number").columns
    wide_slim[num_cols] = wide_slim[num_cols].round(4)

    before = df.shape[1]
    df = df.drop(columns=[c for c in all_new_cols if c in df.columns], errors="ignore")
    df = df.merge(wide_slim, left_on="_fight_id", right_on="fight_id", how="left")
    df = df.drop(columns=["_fight_id", "fight_id"], errors="ignore")

    # ── Fight-level derived features (computed from existing CSV columns) ──────
    print("  Style matchup, stance, division, rank features ...")
    _EPS = 1e-6
    r_splm   = pd.to_numeric(df["R_avg_SIG_STR_landed"], errors="coerce").fillna(0)
    r_td_avg = pd.to_numeric(df["R_avg_TD_landed"],      errors="coerce").fillna(0)
    b_splm   = pd.to_numeric(df["B_avg_SIG_STR_landed"], errors="coerce").fillna(0)
    b_td_avg = pd.to_numeric(df["B_avg_TD_landed"],      errors="coerce").fillna(0)
    r_denom  = r_splm + r_td_avg + _EPS
    b_denom  = b_splm + b_td_avg + _EPS
    r_strike  = r_splm   / r_denom
    r_grapple = r_td_avg / r_denom
    b_strike  = b_splm   / b_denom
    b_grapple = b_td_avg / b_denom
    df["grapple_ratio_diff"]  = (r_grapple - b_grapple).round(4)
    df["striker_vs_wrestler"] = (r_strike  * b_grapple).round(4)
    df["wrestler_vs_striker"] = (r_grapple * b_strike).round(4)

    stance_r = df["R_Stance"].fillna("Orthodox").str.strip().str.lower()
    stance_b = df["B_Stance"].fillna("Orthodox").str.strip().str.lower()
    red_sp   = stance_r == "southpaw"
    blue_sp  = stance_b == "southpaw"
    red_or   = stance_r == "orthodox"
    blue_or  = stance_b == "orthodox"
    df["southpaw_adv_diff"] = ((red_sp & blue_or).astype(int) - (red_or & blue_sp).astype(int))
    df["both_southpaw"]     = (red_sp & blue_sp).astype(int)

    _UNRANKED = 16.0
    r_rank = pd.to_numeric(df["R_match_weightclass_rank"], errors="coerce").fillna(0)
    b_rank = pd.to_numeric(df["B_match_weightclass_rank"], errors="coerce").fillna(0)
    df["weightclass_rank_diff"] = (r_rank.where(r_rank > 0, _UNRANKED) - b_rank.where(b_rank > 0, _UNRANKED))

    from config import DIVISIONS
    div_lower = df["weight_class"].str.lower().str.strip().fillna("")
    for div in DIVISIONS:
        col = "div_" + div.replace(" ", "_").replace("'", "")
        df[col] = (div_lower == div).astype(int)

    matched = df["R_elo"].notna().sum()
    print(f"\n  Matched {matched}/{len(df)} rows ({matched/len(df):.1%})")
    print(f"  Columns: {before} -> {df.shape[1]} (+{df.shape[1]-before})")

    if dry_run:
        print("\nSample (first 3 matched rows):")
        sample = df[df["R_elo"].notna()].head(3)[["date", "R_fighter", "B_fighter", "R_elo", "B_elo", "R_recent_win_rate", "R_sos"]]
        print(sample.to_string(index=False))
        print("\nDry run -- no changes written.")
        return

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(KAGGLE_CSV, index=False)
    print(f"\nSaved {KAGGLE_CSV.name}: {len(df)} rows, {df.shape[1]} columns")
    print(f"  Added {df.shape[1]-before} computed feature columns")
    print("\nNext steps:")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
