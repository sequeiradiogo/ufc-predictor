"""
scripts/add_defensive_stats_to_csv.py

Add R_sapm, B_sapm, R_str_def, B_str_def, R_td_def, B_td_def columns to
ufc-master.csv by looking up pre-fight rolling values from the UFCStats DB.

These stats were previously cross-referenced at feature-build time via
enrich_from_v2(). Adding them to the CSV makes it self-contained and
removes the UFCStats DB dependency from ML_data_preparation_v1.py.

Also rounds all numeric columns to 2 decimal places.

Run once, then update ingest_mdabbert.py and ML_data_preparation_v1.py:
    python scripts/add_defensive_stats_to_csv.py
    python scripts/add_defensive_stats_to_csv.py --dry-run
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"
DB_PATH    = ROOT / "db" / "ufc_ufcstats.db"

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

DEFENSIVE_COLS = (
    "sapm", "str_def", "td_def",
    "head_acc", "body_acc", "leg_acc",
    "head_def", "body_def", "dist_def", "ground_def",
)


def build_lookup(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Return {name_lower: DataFrame with all defensive/zone cols} from UFCStats."""
    df = pd.read_sql_query(
        """
        SELECT fi.name,
               f.date,
               CAST(fs.sapm       AS REAL) AS sapm,
               CAST(fs.str_def    AS REAL) AS str_def,
               CAST(fs.td_def     AS REAL) AS td_def,
               CAST(fs.head_acc   AS REAL) AS head_acc,
               CAST(fs.body_acc   AS REAL) AS body_acc,
               CAST(fs.leg_acc    AS REAL) AS leg_acc,
               CAST(fs.head_def   AS REAL) AS head_def,
               CAST(fs.body_def   AS REAL) AS body_def,
               CAST(fs.dist_def   AS REAL) AS dist_def,
               CAST(fs.ground_def AS REAL) AS ground_def
        FROM fight_stats fs
        JOIN fights     f  ON fs.fight_id  = f.fight_id
        JOIN fighters   fi ON fs.fighter_id = fi.fighter_id
        ORDER BY fi.name, f.date
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])
    return {
        name: grp.sort_values("date").reset_index(drop=True)
        for name, grp in df.groupby(df["name"].str.lower().str.strip())
    }


def _lookup(name: str, fight_date: pd.Timestamp,
            lookup: dict[str, pd.DataFrame]) -> tuple:
    """Return (sapm, str_def, td_def, head_acc, body_acc, leg_acc, head_def, body_def, dist_def, ground_def)."""
    key = NAME_ALIASES.get(name.lower().strip(), name).lower().strip()
    grp = lookup.get(key)
    _nan10 = (np.nan,) * 10
    if grp is None or grp.empty:
        return _nan10
    before = grp[grp["date"] <= fight_date]
    if before.empty:
        return _nan10
    row = before.iloc[-1]
    def _f(c):
        return float(row[c]) if c in row.index and pd.notna(row[c]) else np.nan
    return (
        _f("sapm"), _f("str_def"), _f("td_def"),
        _f("head_acc"), _f("body_acc"), _f("leg_acc"),
        _f("head_def"), _f("body_def"), _f("dist_def"), _f("ground_def"),
    )


def main(dry_run: bool = False) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    print("Building UFCStats defensive stat lookup ...")
    conn = sqlite3.connect(DB_PATH)
    lookup = build_lookup(conn)
    conn.close()

    # Populate defensive stat columns
    n = len(df)
    r_sapm       = np.full(n, np.nan)
    r_str_def    = np.full(n, np.nan)
    r_td_def     = np.full(n, np.nan)
    r_head_acc   = np.full(n, np.nan)
    r_body_acc   = np.full(n, np.nan)
    r_leg_acc    = np.full(n, np.nan)
    r_head_def   = np.full(n, np.nan)
    r_body_def   = np.full(n, np.nan)
    r_dist_def   = np.full(n, np.nan)
    r_ground_def = np.full(n, np.nan)
    b_sapm       = np.full(n, np.nan)
    b_str_def    = np.full(n, np.nan)
    b_td_def     = np.full(n, np.nan)
    b_head_acc   = np.full(n, np.nan)
    b_body_acc   = np.full(n, np.nan)
    b_leg_acc    = np.full(n, np.nan)
    b_head_def   = np.full(n, np.nan)
    b_body_def   = np.full(n, np.nan)
    b_dist_def   = np.full(n, np.nan)
    b_ground_def = np.full(n, np.nan)

    matched = no_match = 0
    for i, row in enumerate(df.itertuples(index=False)):
        rs, rsd, rtd, rha, rba, rla, rhd, rbd, rdd, rgd = _lookup(row.R_fighter, row.date, lookup)
        bs, bsd, btd, bha, bba, bla, bhd, bbd, bdd, bgd = _lookup(row.B_fighter, row.date, lookup)
        r_sapm[i], r_str_def[i], r_td_def[i] = rs, rsd, rtd
        r_head_acc[i], r_body_acc[i], r_leg_acc[i] = rha, rba, rla
        r_head_def[i], r_body_def[i], r_dist_def[i], r_ground_def[i] = rhd, rbd, rdd, rgd
        b_sapm[i], b_str_def[i], b_td_def[i] = bs, bsd, btd
        b_head_acc[i], b_body_acc[i], b_leg_acc[i] = bha, bba, bla
        b_head_def[i], b_body_def[i], b_dist_def[i], b_ground_def[i] = bhd, bbd, bdd, bgd
        if pd.notna(rs):
            matched += 1
        else:
            no_match += 1

    print(f"  R_fighter: matched={matched}  no_match={no_match} out of {len(df)}")

    if dry_run:
        sample = df[["date","R_fighter","B_fighter"]].copy()
        sample["R_sapm"]    = r_sapm
        sample["R_str_def"] = r_str_def
        sample["R_td_def"]  = r_td_def
        sample["R_head_acc"] = r_head_acc
        sample["R_head_def"] = r_head_def
        print("\nSample (last 5 rows):")
        print(sample.tail(5).to_string(index=False))
        print("\nDry run -- no changes written.")
        return

    df["R_sapm"]       = np.round(r_sapm,       2)
    df["B_sapm"]       = np.round(b_sapm,       2)
    df["R_str_def"]    = np.round(r_str_def,    2)
    df["B_str_def"]    = np.round(b_str_def,    2)
    df["R_td_def"]     = np.round(r_td_def,     2)
    df["B_td_def"]     = np.round(b_td_def,     2)
    df["R_head_acc"]   = np.round(r_head_acc,   2)
    df["B_head_acc"]   = np.round(b_head_acc,   2)
    df["R_body_acc"]   = np.round(r_body_acc,   2)
    df["B_body_acc"]   = np.round(b_body_acc,   2)
    df["R_leg_acc"]    = np.round(r_leg_acc,    2)
    df["B_leg_acc"]    = np.round(b_leg_acc,    2)
    df["R_head_def"]   = np.round(r_head_def,   2)
    df["B_head_def"]   = np.round(b_head_def,   2)
    df["R_body_def"]   = np.round(r_body_def,   2)
    df["B_body_def"]   = np.round(b_body_def,   2)
    df["R_dist_def"]   = np.round(r_dist_def,   2)
    df["B_dist_def"]   = np.round(b_dist_def,   2)
    df["R_ground_def"] = np.round(r_ground_def, 2)
    df["B_ground_def"] = np.round(b_ground_def, 2)

    # Round all numeric columns to 2 decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(KAGGLE_CSV, index=False)
    print(f"\nSaved {KAGGLE_CSV.name}: {len(df)} rows, {len(df.columns)} columns")
    print("  Added: R/B_sapm, R/B_str_def, R/B_td_def, R/B_head_acc, R/B_body_acc, R/B_leg_acc")
    print("  Added: R/B_head_def, R/B_body_def, R/B_dist_def, R/B_ground_def")
    print("  All numeric columns rounded to 2 decimal places")
    print("\nNext steps:")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")
    print("  python ml/train_v1_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
