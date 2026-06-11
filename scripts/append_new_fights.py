"""
scripts/append_new_fights.py

Append UFC fights from the UFCStats DB that are missing from ufc-master.csv.

For each missing fight this script:
  1. Identifies the canonical fighter name to use (matching existing CSV rows)
  2. Pulls pre-fight career averages from UFCStats rolling fight_stats
  3. Replays fight history for streaks, method counts, and title bout tally
  4. Formats a row matching the ufc-master.csv schema
  5. Appends to ufc-master.csv (or previews with --dry-run)

splm is taken directly from UFCStats rolling stats (already per-minute scale).
No fix script needed for new fights -- the era split only affected the old
Kaggle-compiled rows, which are already patched in the committed CSV.

Usage:
    python scripts/append_new_fights.py
    python scripts/append_new_fights.py --dry-run
    python scripts/append_new_fights.py --from-date 2026-04-01
"""
import argparse
import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"
DB_PATH = ROOT / "db" / "ufc_ufcstats.db"

# UFCStats division (lowercase) -> Kaggle weight_class string
DIVISION_MAP: dict[str, str] = {
    "heavyweight":            "Heavyweight",
    "light heavyweight":      "Light Heavyweight",
    "middleweight":           "Middleweight",
    "welterweight":           "Welterweight",
    "lightweight":            "Lightweight",
    "featherweight":          "Featherweight",
    "bantamweight":           "Bantamweight",
    "flyweight":              "Flyweight",
    "women's strawweight":    "Women's Strawweight",
    "women's flyweight":      "Women's Flyweight",
    "women's bantamweight":   "Women's Bantamweight",
    "women's featherweight":  "Women's Featherweight",
    "catch weight":           "Catch Weight",
    "super heavyweight":      "Super Heavyweight",
    "open weight":            "Open Weight",
}

WOMENS_DIVISIONS = {
    "women's strawweight", "women's flyweight",
    "women's bantamweight", "women's featherweight",
}

# Division -> weight limit in lbs
DIVISION_WEIGHT_LBS: dict[str, float] = {
    "women's strawweight":   115.0,
    "women's flyweight":     125.0,
    "women's bantamweight":  135.0,
    "women's featherweight": 145.0,
    "flyweight":             125.0,
    "bantamweight":          135.0,
    "featherweight":         145.0,
    "lightweight":           155.0,
    "welterweight":          170.0,
    "middleweight":          185.0,
    "light heavyweight":     205.0,
    "heavyweight":           265.0,
}

# UFCStats method -> Kaggle finish code
METHOD_TO_FINISH: dict[str, str] = {
    "KO/TKO":                "KO/TKO",
    "Submission":            "SUB",
    "Decision - Unanimous":  "U-DEC",
    "Decision - Split":      "S-DEC",
    "Decision - Majority":   "M-DEC",
    "Could Not Continue":    "CNC",
    "Decision - Technical":  "U-DEC",
    "Overturned":            "U-DEC",
}

# Kaggle name (lowercase) -> UFCStats canonical name
# (same dict as fix_splm_in_csv.py / fix_td_sub_in_csv.py)
NAME_ALIASES: dict[str, str] = {
    "joanne calderwood":            "Joanne Wood",
    "tecia torres":                 "Tecia Pennington",
    "michelle waterson":            "Michelle Waterson-Gomez",
    "katlyn chookagian":            "Katlyn Cerminara",
    "yana kunitskaya":              "Yana Santos",
    "cheyanne buys":                "Cheyanne Vlismas",
    "cris cyborg":                  "Cristiane Justino",
    "mirko cro cop":                "Mirko Filipovic",
    "rampage jackson":              "Quinton Jackson",
    "minotauro nogueira":           "Antonio Rodrigo Nogueira",
    "polo reyes":                   "Marco Polo Reyes",
    "rafael feijao":                "Rafael Cavalcante",
    "tiago trator":                 "Tiago dos Santos e Silva",
    "bubba mcdaniel":               "Robert McDaniel",
    "bobby green":                  "King Green",
    "patricio freire":              "Patricio Pitbull",
    "weili zhang":                  "Zhang Weili",
    "ning guangyou":                "Guangyou Ning",
    "liu pingyuan":                 "Pingyuan Liu",
    "an ying wang":                 "Anying Wang",
    "aori qileng":                  "Aoriqileng",
    "wuliji buren":                 "Wulijiburen",
    "chanmi jeon":                  "Chan-Mi Jeon",
    "seohee ham":                   "Seo Hee Ham",
    "da un jung":                   "Da Woon Jung",
    "da-un jung":                   "Da Woon Jung",
    "danaa batgerel":               "Batgerel Danaa",
    "na liang":                     "Liang Na",
    "tiequan zhang":                "Zhang Tiequan",
    "rong zhu":                     "Rongzhu",
    "su mudaerji":                  "Sumudaerji",
    "heili alateng":                "Alatengheili",
    "rick glenn":                   "Ricky Glenn",
    "bradley scott":                "Brad Scott",
    "jimmy wallhead":               "Jim Wallhead",
    "nico musoke":                  "Nicholas Musoke",
    "costas philippou":             "Constantinos Philippou",
    "rob whiteford":                "Robert Whiteford",
    "benny alloway":                "Ben Alloway",
    "joe gigliotti":                "Joseph Gigliotti",
    "philip rowe":                  "Phil Rowe",
    "juan puig":                    "Juan Manuel Puig",
    "kai kamaka":                   "Kai Kamaka III",
    "tim johnson":                  "Timothy Johnson",
    "jim crute":                    "Jimmy Crute",
    "joshua culibao":               "Josh Culibao",
    "phillip hawes":                "Phil Hawes",
    "zachary reese":                "Zach Reese",
    "luci pudilova":                "Lucie Pudilova",
    "montserrat rendon":            "Montse Rendon",
    "zu anyanwu":                   "Azunna Anyanwu",
    "roldan sangcha-an":            "Roldan Sangcha'an",
    "kai kara france":              "Kai Kara-France",
    "waldo cortes-acosta":          "Waldo Cortes Acosta",
    "heather jo clark":             "Heather Clark",
    "emily peters kagan":           "Emily Kagan",
    "carlo pedersoli":              "Carlo Pedersoli Jr.",
    "glaico franca":                "Glaico Franca Moreira",
    "joshua sampo":                 "Josh Sampo",
    "alvaro herrera":               "Alvaro Herrera Mendoza",
    "wendell oliveira":             "Wendell Oliveira Marques",
    "elizeu dos santos":            "Elizeu Zaleski dos Santos",
    "humberto brown":               "Humberto Brown Morrison",
    "raphael pessoa nunes":         "Raphael Pessoa",
    "rodolfo rubio":                "Rodolfo Rubio Perez",
    "montserrat conejo":            "Montserrat Conejo Ruiz",
    "rocco martin":                 "Anthony Rocco Martin",
    "vernon ramos":                 "Vernon Ramos Ho",
    "omar antonio morales ferrer":  "Omar Morales",
    "aleksandra albu":              "Aleksandra Albu",
    "alexandra albu":               "Aleksandra Albu",
    "william patolino":             "William Macario",
    "alekander volkov":             "Alexander Volkov",
    "alex munoz":                   "Alexander Munoz",
    "ali qaisi":                    "Ali AlQaisi",
    "caludia gadelha":              "Claudia Gadelha",
    "caludio puelles":              "Claudio Puelles",
    "grigorii popov":               "Grigory Popov",
    "ian garry":                    "Ian Machado Garry",
    "isabela de pauda":             "Isabela de Padua",
    "jun yong park":                "JunYong Park",
    "kalinn williams":              "Khaos Williams",
    "krzystof jotko":               "Krzysztof Jotko",
    "mizuki inoue":                 "Mizuki",
    "ode obsourne":                 "Ode Osbourne",
    "peter yan":                    "Petr Yan",
    "vincente luque":               "Vicente Luque",
    "youssef zalel":                "Youssef Zalal",
    "zhalgas zhamagulov":           "Zhalgas Zhumagulov",
    "nina ansaroff":                "Nina Nunes",
    "ariane lipski":                "Ariane da Silva",
    "brianna van buren":            "Brianna Fortino",
    "ulka sasaki":                  "Yuta Sasaki",
    "roberto sanchez":              "Robert Sanchez",
}

# Reverse: UFCStats_name_lower -> Kaggle name
_REVERSE_ALIASES: dict[str, str] = {
    v.lower(): k for k, v in NAME_ALIASES.items()
}


def _kaggle_name(ufcstats_name: str, kaggle_lower: set[str]) -> str:
    """Return the Kaggle CSV name to use for a UFCStats fighter."""
    lower = ufcstats_name.lower().strip()
    if lower in kaggle_lower:
        return ufcstats_name
    if lower in _REVERSE_ALIASES:
        # Reconstruct Kaggle name from the alias key (title-case the stored key)
        alias_key = _REVERSE_ALIASES[lower]
        # Find the actual Kaggle CSV entry to preserve original capitalisation
        return alias_key
    return ufcstats_name


def _history_stats(fighter_id: str, fight_date: str, conn: sqlite3.Connection) -> dict:
    """Compute career stats for fighter_id before fight_date by replaying history."""
    rows = conn.execute(
        """
        SELECT f.method, f.winner_id, f.title_fight, f.finish_round, f.no_of_rounds,
               CAST(opp.kd AS INTEGER) AS opp_kd
        FROM fight_stats fs
        JOIN fights f ON fs.fight_id = f.fight_id
        JOIN fight_stats opp ON opp.fight_id = f.fight_id AND opp.fighter_id != ?
        WHERE fs.fighter_id = ? AND f.date < ?
        ORDER BY f.date
        """,
        (fighter_id, fighter_id, fight_date),
    ).fetchall()

    cur_win = cur_lose = longest_win = 0
    ko_w = sub_w = udec_w = sdec_w = mdec_w = 0
    title_bouts = total_rounds = kd_received = 0

    for method, winner_id, title_fight, finish_round, no_of_rounds, opp_kd in rows:
        is_win = winner_id == fighter_id
        if is_win:
            cur_win += 1
            cur_lose = 0
            longest_win = max(longest_win, cur_win)
            m = method or ""
            if m == "KO/TKO":
                ko_w += 1
            elif m == "Submission":
                sub_w += 1
            elif m == "Decision - Unanimous":
                udec_w += 1
            elif m == "Decision - Split":
                sdec_w += 1
            elif m == "Decision - Majority":
                mdec_w += 1
        else:
            cur_lose += 1
            cur_win = 0
        if title_fight:
            title_bouts += 1
        total_rounds += finish_round or no_of_rounds or 3
        kd_received += opp_kd or 0

    return {
        "current_win_streak":  cur_win,
        "current_lose_streak": cur_lose,
        "longest_win_streak":  longest_win,
        "win_by_ko":           ko_w,
        "win_by_sub":          sub_w,
        "win_by_dec_unanimous": udec_w,
        "win_by_dec_split":    sdec_w,
        "total_title_bouts":   title_bouts,
        "total_rounds_fought": total_rounds,
        "kd_received":         kd_received,
    }


def _rolling_stats(
    fighter_id: str, fight_id: str, conn: sqlite3.Connection
) -> dict:
    """Get pre-fight rolling stats for fighter_id in fight fight_id."""
    row = conn.execute(
        """
        SELECT CAST(fs.wins        AS REAL),
               CAST(fs.losses      AS REAL),
               CAST(fs.splm        AS REAL),
               CAST(fs.td_avg      AS REAL),
               CAST(fs.sub_avg     AS REAL),
               CAST(fs.sig_str_acc AS REAL),
               CAST(fs.td_acc      AS REAL),
               CAST(fs.sapm        AS REAL),
               CAST(fs.str_def     AS REAL),
               CAST(fs.td_def      AS REAL),
               fi.height, fi.reach, fi.stance, fi.dob
        FROM fight_stats fs
        JOIN fighters fi ON fs.fighter_id = fi.fighter_id
        WHERE fs.fighter_id = ? AND fs.fight_id = ?
        """,
        (fighter_id, fight_id),
    ).fetchone()
    if row is None:
        return {}
    (wins, losses, splm, td_avg, sub_avg, sig_str_acc, td_acc,
     sapm, str_def, td_def, height, reach, stance, dob) = row
    return {
        "wins":         wins or 0,
        "losses":       losses or 0,
        "splm":         splm or np.nan,
        "td_avg":       td_avg or np.nan,
        "sub_avg":      sub_avg or np.nan,
        "sig_str_acc":  sig_str_acc or np.nan,
        "td_acc":       td_acc or np.nan,
        "sapm":         sapm or np.nan,
        "str_def":      str_def or np.nan,
        "td_def":       td_def or np.nan,
        "height":       height,
        "reach":        reach,
        "stance":       stance,
        "dob":          dob,
    }


def _age(dob: str | None, fight_date: str) -> float | None:
    if not dob:
        return None
    try:
        d = pd.to_datetime(dob)
        f = pd.to_datetime(fight_date)
        return round((f - d).days / 365.25, 2)
    except Exception:
        return None


def _finish_round_time(total_secs: int | None, finish_round: int | None) -> str:
    if total_secs is None or finish_round is None or finish_round < 1:
        return ""
    secs_per_round = 300
    elapsed_complete = (finish_round - 1) * secs_per_round
    secs_in_round = max(0, (total_secs or 0) - elapsed_complete)
    m, s = divmod(secs_in_round, 60)
    return f"{m}:{s:02d}"


def _winner_str(winner_id: str | None, r_id: str, b_id: str) -> str:
    if winner_id == r_id:
        return "Red"
    if winner_id == b_id:
        return "Blue"
    return "Draw"


def build_rows(
    new_fights: pd.DataFrame,
    conn: sqlite3.Connection,
    kaggle_lower: set[str],
    kaggle_name_map: dict[str, str],
) -> list[dict]:
    """Build Kaggle-format dicts for each new fight."""
    rows: list[dict] = []

    for _, fight in new_fights.iterrows():
        fid     = fight["fight_id"]
        date    = fight["date"]
        r_id    = fight["r_fighter_id"]
        b_id    = fight["b_fighter_id"]
        r_name  = fight["r_name"]
        b_name  = fight["b_name"]

        # Resolve names to use in CSV
        r_csv = kaggle_name_map.get(r_name.lower().strip(), _kaggle_name(r_name, kaggle_lower))
        b_csv = kaggle_name_map.get(b_name.lower().strip(), _kaggle_name(b_name, kaggle_lower))

        # Rolling pre-fight stats
        r_roll = _rolling_stats(r_id, fid, conn)
        b_roll = _rolling_stats(b_id, fid, conn)
        if not r_roll or not b_roll:
            continue  # skip if fight_stats row missing

        # History-based stats (streaks, methods, title bouts, rounds)
        r_hist = _history_stats(r_id, date, conn)
        b_hist = _history_stats(b_id, date, conn)

        division = (fight.get("division") or "").lower().strip()
        weight_class = DIVISION_MAP.get(division, division.title())
        weight_lbs = DIVISION_WEIGHT_LBS.get(division, np.nan)
        gender = "FEMALE" if division in WOMENS_DIVISIONS else "MALE"
        title_bout = int(fight.get("title_fight") or 0)
        method = fight.get("method") or ""
        finish = METHOD_TO_FINISH.get(method, "U-DEC")
        finish_round = fight.get("finish_round") or None
        total_secs = fight.get("match_time_sec") or None
        no_of_rounds = fight.get("no_of_rounds") or 3
        winner_str = _winner_str(fight.get("winner_id"), r_id, b_id)
        location = fight.get("location") or ""
        country = fight.get("country") or ""
        finish_details = fight.get("finish_details") or ""

        r_age = _age(r_roll.get("dob"), date)
        b_age = _age(b_roll.get("dob"), date)

        row: dict = {
            "R_fighter":  r_csv,
            "B_fighter":  b_csv,
            "R_odds":     fight.get("odds_red") or np.nan,
            "B_odds":     fight.get("odds_blue") or np.nan,
            "R_ev":       np.nan,
            "B_ev":       np.nan,
            "date":       date,
            "location":   location,
            "country":    country,
            "Winner":     winner_str,
            "title_bout": title_bout,
            "weight_class": weight_class,
            "gender":     gender,
            "no_of_rounds": no_of_rounds,
            # Blue fighter career stats
            "B_current_lose_streak":        b_hist["current_lose_streak"],
            "B_current_win_streak":         b_hist["current_win_streak"],
            "B_draw":                       0,
            "B_avg_SIG_STR_landed":         b_roll.get("splm", np.nan),
            "B_avg_SIG_STR_pct":            b_roll.get("sig_str_acc", np.nan) / 100,
            "B_avg_SUB_ATT":                b_roll.get("sub_avg", np.nan),
            "B_avg_TD_landed":              b_roll.get("td_avg", np.nan),
            "B_avg_TD_pct":                 b_roll.get("td_acc", np.nan) / 100,
            "B_longest_win_streak":         b_hist["longest_win_streak"],
            "B_losses":                     b_roll.get("losses", np.nan),
            "B_total_rounds_fought":        b_hist["total_rounds_fought"],
            "B_total_title_bouts":          b_hist["total_title_bouts"],
            "B_win_by_Decision_Majority":   0,
            "B_win_by_Decision_Split":      b_hist["win_by_dec_split"],
            "B_win_by_Decision_Unanimous":  b_hist["win_by_dec_unanimous"],
            "B_win_by_KO/TKO":             b_hist["win_by_ko"],
            "B_win_by_Submission":          b_hist["win_by_sub"],
            "B_win_by_TKO_Doctor_Stoppage": 0,
            "B_wins":                       b_roll.get("wins", np.nan),
            "B_Stance":                     b_roll.get("stance"),
            "B_Height_cms":                 b_roll.get("height"),
            "B_Reach_cms":                  b_roll.get("reach"),
            "B_Weight_lbs":                 weight_lbs,
            # Red fighter career stats
            "R_current_lose_streak":        r_hist["current_lose_streak"],
            "R_current_win_streak":         r_hist["current_win_streak"],
            "R_draw":                       0,
            "R_avg_SIG_STR_landed":         r_roll.get("splm", np.nan),
            "R_avg_SIG_STR_pct":            r_roll.get("sig_str_acc", np.nan) / 100,
            "R_avg_SUB_ATT":                r_roll.get("sub_avg", np.nan),
            "R_avg_TD_landed":              r_roll.get("td_avg", np.nan),
            "R_avg_TD_pct":                 r_roll.get("td_acc", np.nan) / 100,
            "R_longest_win_streak":         r_hist["longest_win_streak"],
            "R_losses":                     r_roll.get("losses", np.nan),
            "R_total_rounds_fought":        r_hist["total_rounds_fought"],
            "R_total_title_bouts":          r_hist["total_title_bouts"],
            "R_win_by_Decision_Majority":   0,
            "R_win_by_Decision_Split":      r_hist["win_by_dec_split"],
            "R_win_by_Decision_Unanimous":  r_hist["win_by_dec_unanimous"],
            "R_win_by_KO/TKO":             r_hist["win_by_ko"],
            "R_win_by_Submission":          r_hist["win_by_sub"],
            "R_win_by_TKO_Doctor_Stoppage": 0,
            "R_wins":                       r_roll.get("wins", np.nan),
            "R_Stance":                     r_roll.get("stance"),
            "R_Height_cms":                 r_roll.get("height"),
            "R_Reach_cms":                  r_roll.get("reach"),
            "R_Weight_lbs":                 weight_lbs,
            "R_age":                        r_age,
            "B_age":                        b_age,
            # Difference columns (convenience, not used by model directly)
            "lose_streak_dif":    r_hist["current_lose_streak"]   - b_hist["current_lose_streak"],
            "win_streak_dif":     r_hist["current_win_streak"]    - b_hist["current_win_streak"],
            "longest_win_streak_dif": r_hist["longest_win_streak"] - b_hist["longest_win_streak"],
            "win_dif":    (r_roll.get("wins") or 0) - (b_roll.get("wins") or 0),
            "loss_dif":   (r_roll.get("losses") or 0) - (b_roll.get("losses") or 0),
            "total_round_dif":   r_hist["total_rounds_fought"] - b_hist["total_rounds_fought"],
            "total_title_bout_dif": r_hist["total_title_bouts"] - b_hist["total_title_bouts"],
            "ko_dif":    r_hist["win_by_ko"]  - b_hist["win_by_ko"],
            "sub_dif":   r_hist["win_by_sub"] - b_hist["win_by_sub"],
            "height_dif":  (r_roll.get("height") or 0) - (b_roll.get("height") or 0),
            "reach_dif":   (r_roll.get("reach")  or 0) - (b_roll.get("reach")  or 0),
            "age_dif":     (r_age or 0) - (b_age or 0),
            "sig_str_dif":   (r_roll.get("splm")    or 0) - (b_roll.get("splm")    or 0),
            "avg_sub_att_dif": (r_roll.get("sub_avg") or 0) - (b_roll.get("sub_avg") or 0),
            "avg_td_dif":    (r_roll.get("td_avg")  or 0) - (b_roll.get("td_avg")  or 0),
            "empty_arena":  0,
            # Ranking columns -- not available from UFCStats; leave as NaN
            "B_match_weightclass_rank": np.nan,
            "R_match_weightclass_rank": np.nan,
            "R_Women's Flyweight_rank":     np.nan,
            "R_Women's Featherweight_rank": np.nan,
            "R_Women's Strawweight_rank":   np.nan,
            "R_Women's Bantamweight_rank":  np.nan,
            "R_Heavyweight_rank":           np.nan,
            "R_Light Heavyweight_rank":     np.nan,
            "R_Middleweight_rank":          np.nan,
            "R_Welterweight_rank":          np.nan,
            "R_Lightweight_rank":           np.nan,
            "R_Featherweight_rank":         np.nan,
            "R_Bantamweight_rank":          np.nan,
            "R_Flyweight_rank":             np.nan,
            "R_Pound-for-Pound_rank":       np.nan,
            "B_Women's Flyweight_rank":     np.nan,
            "B_Women's Featherweight_rank": np.nan,
            "B_Women's Strawweight_rank":   np.nan,
            "B_Women's Bantamweight_rank":  np.nan,
            "B_Heavyweight_rank":           np.nan,
            "B_Light Heavyweight_rank":     np.nan,
            "B_Middleweight_rank":          np.nan,
            "B_Welterweight_rank":          np.nan,
            "B_Lightweight_rank":           np.nan,
            "B_Featherweight_rank":         np.nan,
            "B_Bantamweight_rank":          np.nan,
            "B_Flyweight_rank":             np.nan,
            "B_Pound-for-Pound_rank":       np.nan,
            "better_rank":  np.nan,
            # Fight outcome columns
            "finish":             finish,
            "finish_details":     finish_details,
            "finish_round":       finish_round,
            "finish_round_time":  _finish_round_time(total_secs, finish_round),
            "total_fight_time_secs": total_secs,
            # Odds columns
            "r_dec_odds": np.nan,
            "b_dec_odds": np.nan,
            "r_sub_odds": np.nan,
            "b_sub_odds": np.nan,
            "r_ko_odds":  np.nan,
            "b_ko_odds":  np.nan,
            # Defensive metrics (from UFCStats rolling stats)
            "R_sapm":    round(r_roll.get("sapm",    np.nan) or np.nan, 2),
            "B_sapm":    round(b_roll.get("sapm",    np.nan) or np.nan, 2),
            "R_str_def": round(r_roll.get("str_def", np.nan) or np.nan, 2),
            "B_str_def": round(b_roll.get("str_def", np.nan) or np.nan, 2),
            "R_td_def":  round(r_roll.get("td_def",  np.nan) or np.nan, 2),
            "B_td_def":  round(b_roll.get("td_def",  np.nan) or np.nan, 2),
            # kd_received is filled by add_computed_features_to_csv.py; leave NaN here
            "R_kd_received": np.nan,
            "B_kd_received": np.nan,
        }
        rows.append(row)

    return rows


def main(dry_run: bool = False, from_date: str | None = None) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Build lookup structures for existing fighters
    all_kaggle_names = set(
        list(df["R_fighter"].str.strip()) + list(df["B_fighter"].str.strip())
    )
    kaggle_lower: set[str] = {n.lower() for n in all_kaggle_names}

    # Map UFCStats_lower -> exact Kaggle CSV name (for fighters in both systems)
    # Covers both exact matches and reverse aliases
    kaggle_name_map: dict[str, str] = {}
    for name in all_kaggle_names:
        kaggle_name_map[name.lower().strip()] = name
    # Also map UFCStats canonical names (alias values) -> Kaggle alias keys
    for kaggle_key, ufcstats_val in NAME_ALIASES.items():
        if ufcstats_val.lower() not in kaggle_name_map:
            # Find canonical Kaggle form: look for the Kaggle key in the CSV
            for n in all_kaggle_names:
                if n.lower() == kaggle_key:
                    kaggle_name_map[ufcstats_val.lower()] = n
                    break

    # Set of existing fights: (date, r_lower, b_lower)
    existing_keys: set[tuple[str, str, str]] = set(
        zip(
            df["date"],
            df["R_fighter"].str.lower().str.strip(),
            df["B_fighter"].str.lower().str.strip(),
        )
    )

    print("Connecting to UFCStats DB ...")
    conn = sqlite3.connect(str(DB_PATH))

    # Load all UFCStats fights with fighter names
    ufcstats = pd.read_sql_query(
        """
        SELECT f.fight_id, f.date, f.division, f.title_fight, f.method,
               f.winner_id, f.odds_red, f.odds_blue, f.finish_round,
               f.match_time_sec, f.no_of_rounds, f.gender,
               f.r_fighter_id, f.b_fighter_id,
               COALESCE(f.location, '') AS location,
               COALESCE(f.country, '')  AS country,
               COALESCE(f.finish_details, '') AS finish_details,
               r.name AS r_name, b.name AS b_name
        FROM fights f
        JOIN fighters r ON f.r_fighter_id = r.fighter_id
        JOIN fighters b ON f.b_fighter_id = b.fighter_id
        ORDER BY f.date
        """,
        conn,
    )

    # Filter to new fights only
    def _is_new(row: pd.Series) -> bool:
        key = (row["date"], row["r_name"].lower().strip(), row["b_name"].lower().strip())
        if key in existing_keys:
            return False
        # Also check with alias resolution
        r_resolved = NAME_ALIASES.get(row["r_name"].lower().strip(), row["r_name"]).lower()
        b_resolved = NAME_ALIASES.get(row["b_name"].lower().strip(), row["b_name"]).lower()
        return (row["date"], r_resolved, b_resolved) not in existing_keys

    new_fights = ufcstats[ufcstats.apply(_is_new, axis=1)].copy()

    # Default: only append fights after the last date already in the CSV.
    # Pass --from-date to override (e.g. backfill older prelims).
    if from_date:
        new_fights = new_fights[new_fights["date"] >= from_date]
    else:
        last_csv_date = df["date"].max()
        new_fights = new_fights[new_fights["date"] > last_csv_date]
        print(f"  (default: only fights after CSV cutoff {last_csv_date})")

    print(f"New fights to append: {len(new_fights)}")
    if new_fights.empty:
        print("Nothing to do.")
        conn.close()
        return

    print(f"  Date range: {new_fights['date'].min()} to {new_fights['date'].max()}")

    rows = build_rows(new_fights, conn, kaggle_lower, kaggle_name_map)
    conn.close()

    print(f"  Rows built: {len(rows)} (skipped {len(new_fights) - len(rows)} missing fight_stats)")

    if not rows:
        return

    new_df = pd.DataFrame(rows, columns=df.columns.tolist())

    if dry_run:
        print("\nDry run -- sample of rows that would be appended:")
        preview_cols = ["date", "R_fighter", "B_fighter", "Winner", "weight_class",
                        "R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed",
                        "R_avg_TD_landed", "B_avg_TD_landed", "finish"]
        print(new_df[preview_cols].tail(10).to_string(index=False))
        print(f"\nDry run -- {len(new_df)} rows not written.")
        return

    combined = pd.concat([df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(KAGGLE_CSV, index=False)
    print(f"\nAppended {len(new_df)} rows. CSV now has {len(combined)} fights.")
    print("\nNext steps:")
    print("  python scripts/fix_splm_in_csv.py")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")
    print("  python ml/train_v1_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append missing UFC fights to ufc-master.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--from-date", default=None,
                        help="Only append fights on or after this date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, from_date=args.from_date)
