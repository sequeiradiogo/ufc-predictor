"""
scripts/add_rankings_to_csv.py

Populate R_match_weightclass_rank and B_match_weightclass_rank in ufc-master.csv
by looking up each fighter's most recent ranking <= the fight date from
raw_data/rankings_history.csv.

Overwrites the existing sparse columns (Kaggle original had ~27% fill rate).

Usage:
    python scripts/add_rankings_to_csv.py
    python scripts/add_rankings_to_csv.py --dry-run
"""
import argparse
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
KAGGLE_CSV    = ROOT / "raw_data" / "ufc-master.csv"
RANKINGS_CSV  = ROOT / "raw_data" / "rankings_history.csv"

# Kaggle name (lowercase) -> rankings_history canonical name (lowercase)
NAME_ALIASES: dict[str, str] = {
    "joanne calderwood":            "joanne wood",
    "tecia torres":                 "tecia pennington",
    "michelle waterson":            "michelle waterson-gomez",
    "katlyn chookagian":            "katlyn cerminara",
    "yana kunitskaya":              "yana santos",
    "cheyanne buys":                "cheyanne vlismas",
    "cris cyborg":                  "cristiane justino",
    "mirko cro cop":                "mirko filipovic",
    "rampage jackson":              "quinton jackson",
    "minotauro nogueira":           "antonio rodrigo nogueira",
    "polo reyes":                   "marco polo reyes",
    "rafael feijao":                "rafael cavalcante",
    "tiago trator":                 "tiago dos santos e silva",
    "bubba mcdaniel":               "robert mcdaniel",
    "bobby green":                  "king green",
    "patricio freire":              "patricio pitbull",
    "weili zhang":                  "zhang weili",
    "ning guangyou":                "guangyou ning",
    "liu pingyuan":                 "pingyuan liu",
    "an ying wang":                 "anying wang",
    "aori qileng":                  "aoriqileng",
    "wuliji buren":                 "wulijiburen",
    "chanmi jeon":                  "chan-mi jeon",
    "seohee ham":                   "seo hee ham",
    "da un jung":                   "da woon jung",
    "da-un jung":                   "da woon jung",
    "danaa batgerel":               "batgerel danaa",
    "na liang":                     "liang na",
    "tiequan zhang":                "zhang tiequan",
    "rong zhu":                     "rongzhu",
    "su mudaerji":                  "sumudaerji",
    "heili alateng":                "alatengheili",
    "rick glenn":                   "ricky glenn",
    "bradley scott":                "brad scott",
    "jimmy wallhead":               "jim wallhead",
    "nico musoke":                  "nicholas musoke",
    "costas philippou":             "constantinos philippou",
    "rob whiteford":                "robert whiteford",
    "benny alloway":                "ben alloway",
    "joe gigliotti":                "joseph gigliotti",
    "philip rowe":                  "phil rowe",
    "juan puig":                    "juan manuel puig",
    "kai kamaka":                   "kai kamaka iii",
    "tim johnson":                  "timothy johnson",
    "jim crute":                    "jimmy crute",
    "joshua culibao":               "josh culibao",
    "phillip hawes":                "phil hawes",
    "zachary reese":                "zach reese",
    "luci pudilova":                "lucie pudilova",
    "montserrat rendon":            "montse rendon",
    "zu anyanwu":                   "azunna anyanwu",
    "roldan sangcha-an":            "roldan sangcha'an",
    "kai kara france":              "kai kara-france",
    "waldo cortes-acosta":          "waldo cortes acosta",
    "heather jo clark":             "heather clark",
    "emily peters kagan":           "emily kagan",
    "carlo pedersoli":              "carlo pedersoli jr.",
    "glaico franca":                "glaico franca moreira",
    "joshua sampo":                 "josh sampo",
    "alvaro herrera":               "alvaro herrera mendoza",
    "wendell oliveira":             "wendell oliveira marques",
    "elizeu dos santos":            "elizeu zaleski dos santos",
    "humberto brown":               "humberto brown morrison",
    "raphael pessoa nunes":         "raphael pessoa",
    "rodolfo rubio":                "rodolfo rubio perez",
    "montserrat conejo":            "montserrat conejo ruiz",
    "rocco martin":                 "anthony rocco martin",
    "vernon ramos":                 "vernon ramos ho",
    "omar antonio morales ferrer":  "omar morales",
    "aleksandra albu":              "aleksandra albu",
    "alexandra albu":               "aleksandra albu",
    "william patolino":             "william macario",
    "alekander volkov":             "alexander volkov",
    "alex munoz":                   "alexander munoz",
    "ali qaisi":                    "ali alqaisi",
    "caludia gadelha":              "claudia gadelha",
    "caludio puelles":              "claudio puelles",
    "grigorii popov":               "grigory popov",
    "ian garry":                    "ian machado garry",
    "isabela de pauda":             "isabela de padua",
    "jun yong park":                "junyong park",
    "kalinn williams":              "khaos williams",
    "krzystof jotko":               "krzysztof jotko",
    "mizuki inoue":                 "mizuki",
    "ode obsourne":                 "ode osbourne",
    "peter yan":                    "petr yan",
    "vincente luque":               "vicente luque",
    "youssef zalel":                "youssef zalal",
    "zhalgas zhamagulov":           "zhalgas zhumagulov",
    "nina ansaroff":                "nina nunes",
    "ariane lipski":                "ariane da silva",
    "brianna van buren":            "brianna fortino",
    "ulka sasaki":                  "yuta sasaki",
    "roberto sanchez":              "robert sanchez",
}


def _build_lookup(rh: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """Return {(name_lower, division_lower): [(date_str, rank), ...]} sorted by date."""
    rh = rh[~rh["weightclass"].str.contains("Pound-for-Pound", case=False, na=False)]
    rh = rh.dropna(subset=["fighter", "rank"])
    rh["rank"] = rh["rank"].astype(int)
    rh = rh.sort_values("date").reset_index(drop=True)

    lookup: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for row in rh.itertuples(index=False):
        name = str(row.fighter).lower().strip()
        div  = str(row.weightclass).lower().strip()
        lookup.setdefault((name, div), []).append((str(row.date)[:10], int(row.rank)))
    return lookup


def _resolve(name: str) -> str:
    """Apply NAME_ALIASES to map a Kaggle fighter name to its canonical lowercase form."""
    key = name.lower().strip()
    return NAME_ALIASES.get(key, key)


def _lookup_rank(
    lookup: dict[tuple[str, str], list[tuple[str, int]]],
    name: str,
    division: str,
    fight_date: str,
) -> float:
    """Return most recent rank <= fight_date for fighter in division, or NaN."""
    key = (_resolve(name), division.lower().strip())
    series = lookup.get(key)
    if not series:
        return np.nan
    dates = [s[0] for s in series]
    pos = bisect.bisect_right(dates, fight_date) - 1
    if pos < 0:
        return np.nan
    return float(series[pos][1])


def main(dry_run: bool = False) -> None:
    print(f"Reading {KAGGLE_CSV.name} ...")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Reading {RANKINGS_CSV.name} ...")
    rh = pd.read_csv(RANKINGS_CSV)
    lookup = _build_lookup(rh)
    print(f"  {len(lookup)} (fighter, division) ranking series loaded.")

    r_ranks = np.full(len(df), np.nan)
    b_ranks = np.full(len(df), np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        date_str = str(row.date)[:10]
        division = str(row.weight_class) if pd.notna(row.weight_class) else ""
        r_ranks[i] = _lookup_rank(lookup, row.R_fighter, division, date_str)
        b_ranks[i] = _lookup_rank(lookup, row.B_fighter, division, date_str)

    r_filled = int(np.sum(~np.isnan(r_ranks)))
    b_filled = int(np.sum(~np.isnan(b_ranks)))
    print(f"  R_match_weightclass_rank: {r_filled}/{len(df)} filled ({r_filled/len(df):.1%})")
    print(f"  B_match_weightclass_rank: {b_filled}/{len(df)} filled ({b_filled/len(df):.1%})")

    if dry_run:
        sample = df[["date", "R_fighter", "B_fighter", "weight_class"]].copy()
        sample["R_rank_new"] = r_ranks
        sample["B_rank_new"] = b_ranks
        print("\nSample (5 rows with rankings):")
        print(sample[~np.isnan(r_ranks)].head(5).to_string(index=False))
        print("\nDry run -- no changes written.")
        return

    df["R_match_weightclass_rank"] = np.where(np.isnan(r_ranks), np.nan, r_ranks)
    df["B_match_weightclass_rank"] = np.where(np.isnan(b_ranks), np.nan, b_ranks)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(KAGGLE_CSV, index=False)
    print(f"\nSaved {KAGGLE_CSV.name}: {len(df)} rows")
    print("  Updated: R_match_weightclass_rank, B_match_weightclass_rank")
    print("\nNext steps:")
    print("  python db/ingest_mdabbert.py --csv raw_data/ufc-master.csv")
    print("  python ml/ML_data_preparation_v1.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
