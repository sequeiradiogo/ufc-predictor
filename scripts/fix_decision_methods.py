"""
One-time migration: fix S-DEC and M-DEC method labels in ufc_ufcstats.db.

The UFCStats scraper stored all split and majority decisions as
"Decision - Unanimous" because _normalize_method checked 'split in t'
which doesn't match the abbreviated "S-DEC" text from event listing tables.

This script matches fights by date + fighter names between the Kaggle CSV
(which has the correct per-fight finish codes) and the UFCStats DB, then
updates the method column for mismatched rows.

Only S-DEC -> Decision - Split and M-DEC -> Decision - Majority are fixed.
Other discrepancies (Overturned vs. KO/TKO etc.) are skipped - they involve
genuinely ambiguous outcomes and are too few to affect accuracy.

Run once after the ufcstats DB is built:
    python scripts/fix_decision_methods.py
    python scripts/fix_decision_methods.py --dry-run   # preview only
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CSV = ROOT / "raw_data" / "ufc-master.csv"
DB_PATH = ROOT / "db" / "ufc_ufcstats.db"

FINISH_TO_METHOD = {
    "S-DEC": "Decision - Split",
    "M-DEC": "Decision - Majority",
}


def main(dry_run: bool = False) -> None:
    df = pd.read_csv(KAGGLE_CSV, usecols=["date", "R_fighter", "B_fighter", "finish"], low_memory=False)
    df = df[df["finish"].isin(FINISH_TO_METHOD)].copy()
    df["method_correct"] = df["finish"].map(FINISH_TO_METHOD)
    df["date"] = df["date"].astype(str)
    df["r_lower"] = df["R_fighter"].str.lower().str.strip()
    df["b_lower"] = df["B_fighter"].str.lower().str.strip()

    conn = sqlite3.connect(DB_PATH)
    try:
        fights_db = pd.read_sql(
            """
            SELECT f.fight_id, f.date, f.method,
                   r.name AS r_name, b.name AS b_name
            FROM fights f
            JOIN fighters r ON f.r_fighter_id = r.fighter_id
            JOIN fighters b ON f.b_fighter_id = b.fighter_id
            """,
            conn,
        )
        fights_db["date"] = fights_db["date"].astype(str)
        fights_db["r_lower"] = fights_db["r_name"].str.lower().str.strip()
        fights_db["b_lower"] = fights_db["b_name"].str.lower().str.strip()

        merged = df.merge(fights_db, on=["date", "r_lower", "b_lower"], how="inner")
        to_fix = merged[merged["method"] != merged["method_correct"]][
            ["fight_id", "date", "r_name", "b_name", "method", "method_correct"]
        ]

        print(f"Fights to update: {len(to_fix)}")
        print(to_fix.groupby(["method", "method_correct"]).size().to_string())

        if dry_run:
            print("\nDry run -- no changes written.")
            return

        cur = conn.cursor()
        updated = 0
        for row in to_fix.itertuples(index=False):
            cur.execute(
                "UPDATE fights SET method = ? WHERE fight_id = ?",
                (row.method_correct, row.fight_id),
            )
            updated += cur.rowcount
        conn.commit()
        print(f"\nUpdated {updated} rows.")

        print("\nNew method distribution:")
        result = pd.read_sql("SELECT method, COUNT(*) as cnt FROM fights GROUP BY method ORDER BY cnt DESC", conn)
        print(result.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix S-DEC/M-DEC methods in UFCStats DB")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
