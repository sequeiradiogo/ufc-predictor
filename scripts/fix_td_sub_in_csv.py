"""
Fix R/B_avg_TD_landed and R/B_avg_SUB_ATT values in ufc-master.csv.

The Kaggle CSV has the same two-era methodology split as splm:
  Pre-~2019:  values are per-FIGHT  (e.g. 2.3 TDs per fight)
  Post-~2019: values are per-15-MIN (e.g. 2.5 TDs per 15 minutes, matching UFCStats)

Measured ratio (Kaggle / UFCStats):
  pre-2019 median: 0.67  (Kaggle is ~30% lower -- wrong scale)
  post-2019 median: 0.95 (Kaggle and UFCStats agree)

Fix: for any cell where kaggle_val / ufcstats_val < RATIO_THRESHOLD (meaning
the Kaggle value is significantly below the correct per-15-min value), replace
with the UFCStats pre-fight rolling value.

Run once, then re-ingest and retrain:
    python scripts/fix_td_sub_in_csv.py
    python scripts/fix_td_sub_in_csv.py --dry-run
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"
DB_PATH    = ROOT / "db" / "ufc_ufcstats.db"

# Replace when kaggle / ufcstats < this — below 0.85 means clearly per-fight scale.
# Post-2019 correct cells have median ratio 0.95; pre-2019 wrong cells have median 0.67.
RATIO_THRESHOLD = 0.85

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
}


def build_lookup(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT fi.name,
               f.date,
               CAST(fs.td_avg  AS REAL) AS td_avg,
               CAST(fs.sub_avg AS REAL) AS sub_avg
        FROM fight_stats fs
        JOIN fights     f  ON fs.fight_id  = f.fight_id
        JOIN fighters   fi ON fs.fighter_id = fi.fighter_id
        WHERE fs.td_avg IS NOT NULL OR fs.sub_avg IS NOT NULL
        ORDER BY fi.name, f.date
        """,
        conn,
    )


def _resolve(name: str) -> str:
    return NAME_ALIASES.get(name.lower().strip(), name)


def _get(grp: pd.DataFrame | None, fight_date: pd.Timestamp, col: str) -> float | None:
    if grp is None or grp.empty:
        return None
    before = grp[grp["date"] < fight_date]
    if before.empty:
        before = grp[grp["date"] <= fight_date]
    if before.empty:
        return None
    v = before.iloc[-1][col]
    return float(v) if pd.notna(v) and float(v) > 0 else None


def _fix_column(
    df: pd.DataFrame,
    kaggle_col: str,
    name_col: str,
    v2_col: str,
    lookup_by_name: dict[str, pd.DataFrame],
    ratio_threshold: float,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Replace cells where kaggle / v2 < ratio_threshold. Returns (replaced, skipped_ratio, no_match)."""
    vals = pd.to_numeric(df[kaggle_col], errors="coerce")
    candidates = vals[vals > 0].index
    replaced = skipped_ratio = no_match = 0

    new_vals = vals.copy()
    for idx in candidates:
        name = df.at[idx, name_col]
        date = df.at[idx, "date"]
        kaggle_val = vals.at[idx]

        canonical = _resolve(str(name))
        grp = lookup_by_name.get(canonical.lower().strip())
        v2_val = _get(grp, date, v2_col)

        if v2_val is None:
            no_match += 1
            continue
        if kaggle_val / v2_val >= ratio_threshold:
            skipped_ratio += 1
            continue
        new_vals.at[idx] = v2_val
        replaced += 1

    if not dry_run:
        df[kaggle_col] = new_vals
    return replaced, skipped_ratio, no_match


def main(dry_run: bool = False) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    print("Connecting to UFCStats DB ...")
    conn = sqlite3.connect(DB_PATH)
    lookup_df = build_lookup(conn)
    conn.close()

    lookup_df["date"] = pd.to_datetime(lookup_df["date"])
    lookup_by_name = {
        n: g.sort_values("date").reset_index(drop=True)
        for n, g in lookup_df.groupby(lookup_df["name"].str.lower().str.strip())
    }

    print(f"\nReplacing cells where kaggle / UFCStats < {RATIO_THRESHOLD}:\n")

    stats: dict[str, tuple[int, int, int]] = {}
    for kaggle_col, name_col, v2_col in [
        ("R_avg_TD_landed", "R_fighter", "td_avg"),
        ("B_avg_TD_landed", "B_fighter", "td_avg"),
        ("R_avg_SUB_ATT",   "R_fighter", "sub_avg"),
        ("B_avg_SUB_ATT",   "B_fighter", "sub_avg"),
    ]:
        r, s, n = _fix_column(df, kaggle_col, name_col, v2_col,
                              lookup_by_name, RATIO_THRESHOLD, dry_run)
        stats[kaggle_col] = (r, s, n)
        print(f"  {kaggle_col:<25}  replaced={r:4d}  in-range(kept)={s:5d}  no-match={n:4d}")

    print()
    for col in ("R_avg_TD_landed", "B_avg_TD_landed", "R_avg_SUB_ATT", "B_avg_SUB_ATT"):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        print(f"  {col:<25}  mean={v.mean():.3f}  std={v.std():.3f}  "
              f"p95={np.percentile(v, 95):.3f}  max={v.max():.3f}")

    if dry_run:
        print("\nDry run -- no changes written.")
        return

    df.to_csv(KAGGLE_CSV, index=False)
    print(f"\nSaved patched CSV to {KAGGLE_CSV}")
    print("Next steps:")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")
    print("  python ml/train_v1_models.py")
    print("  python scripts/backtest_v1.py --from-year 2025 --model ensemble")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix td_avg and sub_avg in ufc-master.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
