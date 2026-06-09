"""
Fix unreasonable R_avg_SIG_STR_landed / B_avg_SIG_STR_landed values in ufc-master.csv.

The Kaggle CSV was compiled in two passes: fights up to ~mid-2019 used average
strikes *per fight* (values typically 20-80), while fights after that date were
added with a different script that computes strikes *per minute* (values 2-9).

Two-pass fix:
  Pass 1 (THRESHOLD check): cells where kaggle_val > THRESHOLD are clearly in
    per-fight scale (UFCStats career-average splm maxes out at ~12.86 for any
    established fighter).  Replace unconditionally where UFCStats match exists.
  Pass 2 (RATIO check): cells in the AMBIGUOUS_LOW..THRESHOLD range are checked
    against UFCStats.  If kaggle_val / ufcstats_val > RATIO_THRESHOLD the cell
    is in per-fight scale (e.g. kaggle=8, ufcstats=0.53 -> ratio=15).  Cells
    with ratio near 1.0 are already correct per-minute values (exact matches
    confirmed for aggressive finishers like Tom Aspinall, Justin Gaethje, etc.).

Run once, then re-ingest and retrain:
    python scripts/fix_splm_in_csv.py
    python scripts/fix_splm_in_csv.py --dry-run
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"
DB_PATH    = ROOT / "db" / "ufc_ufcstats.db"

# Kaggle name -> UFCStats canonical name.
# Sources: name changes after marriage, nicknames vs real names, Chinese name reversal,
# abbreviated first names, transliteration differences.
NAME_ALIASES: dict[str, str] = {
    # Name changes (marriage / official UFC name update)
    "joanne calderwood":            "Joanne Wood",
    "tecia torres":                 "Tecia Pennington",
    "michelle waterson":            "Michelle Waterson-Gomez",
    "katlyn chookagian":            "Katlyn Cerminara",
    "yana kunitskaya":              "Yana Santos",
    "cheyanne buys":                "Cheyanne Vlismas",
    # Nicknames used as full name in Kaggle
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
    # Chinese / Korean name-order reversal
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
    # Abbreviated / alternate first names
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
    # Punctuation / encoding differences
    "roldan sangcha-an":            "Roldan Sangcha'an",
    "kai kara france":              "Kai Kara-France",
    "waldo cortes-acosta":          "Waldo Cortes Acosta",
    # Middle name / suffix differences
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
    # Transliteration differences
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

# Pass 1: cells above this are definitively per-fight scale — replace unconditionally.
# UFCStats absolute career-average max for established fighters = 12.86.
DEFAULT_THRESHOLD = 10.0

# Pass 2: for AMBIGUOUS_LOW..DEFAULT_THRESHOLD, replace only if
# kaggle / ufcstats > RATIO_THRESHOLD (per-fight confirmed by comparison).
# Ratio near 1.0 = already correct per-minute (Tom Aspinall 13.3, Gaethje 10.6, etc.)
AMBIGUOUS_LOW   = 6.0
RATIO_THRESHOLD = 1.5


def build_splm_lookup(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return a DataFrame with (name, date, splm) for every fight_stats row
    in the UFCStats DB.  splm here is the pre-fight rolling career average
    (per minute, shift-1 applied by rolling.py).
    """
    df = pd.read_sql_query(
        """
        SELECT fi.name,
               f.date,
               CAST(fs.splm AS REAL) AS splm
        FROM fight_stats fs
        JOIN fights     f  ON fs.fight_id  = f.fight_id
        JOIN fighters   fi ON fs.fighter_id = fi.fighter_id
        WHERE fs.splm IS NOT NULL AND CAST(fs.splm AS REAL) > 0
        ORDER BY fi.name, f.date
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["name_lower"] = df["name"].str.lower().str.strip()
    return df


def lookup_splm(
    name: str,
    fight_date: pd.Timestamp,
    lookup_by_name: dict[str, pd.DataFrame],
) -> float | None:
    """
    Return the most recent pre-fight splm for *name* at *fight_date*.
    Tries the alias map first, then falls back to the raw Kaggle name.
    Returns None if no match found in UFCStats.
    """
    key = name.lower().strip()
    # Apply alias: Kaggle name -> UFCStats canonical name
    canonical = NAME_ALIASES.get(key, name)
    resolved_key = canonical.lower().strip()

    grp = lookup_by_name.get(resolved_key)
    if grp is None or grp.empty:
        return None
    # Most recent row strictly before the fight date (pre-fight average)
    before = grp[grp["date"] < fight_date]
    if before.empty:
        # Fighter's only UFCStats row is on the same date — use it
        before = grp[grp["date"] <= fight_date]
    if before.empty:
        return None
    return float(before.iloc[-1]["splm"])


def _apply_pass(
    df: pd.DataFrame,
    new_r: pd.Series,
    new_b: pd.Series,
    mask_r: "pd.Series[bool]",
    mask_b: "pd.Series[bool]",
    lookup_by_name: dict,
    ratio_threshold: float | None = None,
) -> tuple[int, int, int, int]:
    """Replace cells selected by mask_r/mask_b.  If ratio_threshold is set,
    only replace when kaggle_val / ufcstats_val > ratio_threshold."""
    r_replaced = r_skipped = b_replaced = b_skipped = 0

    for idx in df[mask_r].index:
        name = df.at[idx, "R_fighter"]
        date = df.at[idx, "date"]
        val  = lookup_splm(name, date, lookup_by_name)
        if val is None:
            r_skipped += 1
            continue
        if ratio_threshold is not None:
            if val <= 0 or (df.at[idx, "R_avg_SIG_STR_landed"] / val) <= ratio_threshold:
                r_skipped += 1
                continue
        new_r.at[idx] = val
        r_replaced += 1

    for idx in df[mask_b].index:
        name = df.at[idx, "B_fighter"]
        date = df.at[idx, "date"]
        val  = lookup_splm(name, date, lookup_by_name)
        if val is None:
            b_skipped += 1
            continue
        if ratio_threshold is not None:
            if val <= 0 or (df.at[idx, "B_avg_SIG_STR_landed"] / val) <= ratio_threshold:
                b_skipped += 1
                continue
        new_b.at[idx] = val
        b_replaced += 1

    return r_replaced, r_skipped, b_replaced, b_skipped


def main(dry_run: bool = False, threshold: float = DEFAULT_THRESHOLD) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Connecting to UFCStats DB ...")
    conn = sqlite3.connect(DB_PATH)
    lookup_df = build_splm_lookup(conn)
    conn.close()

    lookup_by_name: dict[str, pd.DataFrame] = {
        name: grp.sort_values("date").reset_index(drop=True)
        for name, grp in lookup_df.groupby("name_lower")
    }

    r_col = "R_avg_SIG_STR_landed"
    b_col = "B_avg_SIG_STR_landed"
    df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
    df[b_col] = pd.to_numeric(df[b_col], errors="coerce")

    new_r = df[r_col].copy()
    new_b = df[b_col].copy()

    # ── Pass 1: clearly per-fight scale (value > threshold) ───────────────────
    p1_r = df[r_col] > threshold
    p1_b = df[b_col] > threshold
    print(f"\nPass 1 — cells above {threshold} (unconditional replacement):")
    print(f"  Red {p1_r.sum()} cells  |  Blue {p1_b.sum()} cells")
    rr, rs, br, bs = _apply_pass(df, new_r, new_b, p1_r, p1_b, lookup_by_name)
    print(f"  Red  replaced={rr}  kept (no match)={rs}")
    print(f"  Blue replaced={br}  kept (no match)={bs}")

    # ── Pass 2: ambiguous range (ratio check) ─────────────────────────────────
    # Use the CURRENT new_r/new_b values so we don't re-check already-fixed cells.
    p2_r = (new_r >= AMBIGUOUS_LOW) & (new_r <= threshold)
    p2_b = (new_b >= AMBIGUOUS_LOW) & (new_b <= threshold)
    print(f"\nPass 2 — cells in [{AMBIGUOUS_LOW}, {threshold}] with ratio > {RATIO_THRESHOLD}:")
    print(f"  Candidates: Red {p2_r.sum()}  |  Blue {p2_b.sum()}")
    rr2, rs2, br2, bs2 = _apply_pass(df, new_r, new_b, p2_r, p2_b, lookup_by_name,
                                      ratio_threshold=RATIO_THRESHOLD)
    print(f"  Red  replaced={rr2}  skipped (correct scale or no match)={rs2}")
    print(f"  Blue replaced={br2}  skipped (correct scale or no match)={bs2}")

    all_after = pd.concat([new_r, new_b]).dropna()
    print(f"\nFull column after both passes:")
    print(f"  mean={all_after.mean():.2f}  std={all_after.std():.2f}  "
          f"p95={np.percentile(all_after, 95):.2f}  max={all_after.max():.2f}")
    print(f"  Total replacements: Red={rr+rr2}  Blue={br+br2}")

    if dry_run:
        print("\nDry run -- no changes written.")
        return

    df[r_col] = new_r
    df[b_col] = new_b
    df.to_csv(KAGGLE_CSV, index=False)
    print(f"\nSaved patched CSV to {KAGGLE_CSV}")
    print("Next steps:")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")
    print("  python ml/train_v1_models.py")
    print("  python scripts/backtest_v1.py --from-year 2025 --model ensemble")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix per-fight splm values in ufc-master.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
