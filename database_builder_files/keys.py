#!/usr/bin/env python3
"""
add_fks_to_ufc_db.py

Rebuilds fighters, fights, and fight_stats tables in an existing SQLite DB
so that foreign keys are declared in the table definitions.

Safety:
- Creates a timestamped backup of your DB before making changes.
- Runs DDL/data copy with PRAGMA foreign_keys = OFF, then re-enables and checks.

Usage:
    python add_fks_to_ufc_db.py
"""

import sqlite3
from pathlib import Path
import shutil
import datetime
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))
from config import DB_PATH

def backup_db(db_path: Path) -> Path:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = db_path.with_name(db_path.stem + f"_backup_{stamp}" + db_path.suffix)
    shutil.copy2(db_path, bak)
    print(f"Backup created: {bak}")
    return bak

def pragma_table_info(cur, table: str):
    return cur.execute(f"PRAGMA table_info('{table}');").fetchall()

def table_exists(cur, table: str) -> bool:
    r = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone()
    return r is not None

def rebuild_table(cur, table_name: str, create_sql: str, copy_cols: list):
    """
    Rebuild `table_name` using provided create_sql (string). copy_cols is the column list
    to copy from the old table to the new table (must match order for INSERT .. SELECT).
    This will:
     - rename table_name -> table_name_old
     - execute create_sql to create the new table
     - copy data: INSERT INTO table_name(cols) SELECT cols FROM table_name_old
     - drop table_name_old
    """
    old = table_name + "_old"

    if not table_exists(cur, table_name):
        print(f"Table '{table_name}' does not exist — skipping rebuild.")
        return

    print(f"\nRebuilding table '{table_name}' ...")
    # rename
    cur.execute(f"ALTER TABLE {table_name} RENAME TO {old};")
    print(f" - renamed existing '{table_name}' -> '{old}'")

    # create new
    cur.executescript(create_sql)
    print(" - created new table with desired schema")

    # determine columns intersection (copy only columns that exist in old table)
    old_cols_info = pragma_table_info(cur, old)
    old_cols = [c[1] for c in old_cols_info]
    to_copy = [c for c in copy_cols if c in old_cols]
    if not to_copy:
        print(" - no matching columns found to copy (nothing to insert).")
    else:
        cols_sql = ", ".join(f'"{c}"' for c in to_copy)
        cur.execute(f'INSERT INTO "{table_name}" ({cols_sql}) SELECT {cols_sql} FROM "{old}";')
        print(f" - copied {cur.rowcount if hasattr(cur,'rowcount') else 'rows?'} rows into '{table_name}' (columns: {to_copy[:6]}{'...' if len(to_copy)>6 else ''})")

    # drop old
    cur.execute(f"DROP TABLE IF EXISTS {old};")
    print(f" - dropped old table '{old}'")

def main():
    db_file = DB_PATH
    if not db_file.exists():
        print("DB file not found:", db_file)
        sys.exit(1)

    # backup
    bak = backup_db(db_file)

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # turn off foreign keys while we rename/create/copy
    cur.execute("PRAGMA foreign_keys = OFF;")
    conn.commit()

    # --- REBUILD fighters ---
    # We'll ensure fighter_id is PRIMARY KEY in the new table.
    fighters_create = """
    CREATE TABLE IF NOT EXISTS fighters (
        fighter_id TEXT PRIMARY KEY,
        name TEXT
    );
    """
    # copy whatever fighter_id and name columns exist in old table (common names)
    fighters_copy_cols = ['fighter_id', 'name']
    # But if old table used other names (r_id etc.), we allow broad intersection by checking old columns:
    # We'll attempt to copy columns named exactly fighter_id or name only; if old table had different names,
    # the data will still exist in the old table backup; user can transform manually if needed.
    if table_exists(cur, "fighters"):
        # If table exists, rebuild it using the schema above
        rebuild_table(cur, "fighters", fighters_create, fighters_copy_cols)
    else:
        # create fresh
        cur.executescript(fighters_create)
        print("Created 'fighters' (fresh)")

    conn.commit()

    # --- REBUILD fights ---
    # Create fights with FK references to fighters(fighter_id)
    # We'll try to preserve common fight-level columns. Types set to TEXT/INTEGER/REAL for compatibility.
    fights_create = """
    CREATE TABLE IF NOT EXISTS fights (
        fight_id TEXT PRIMARY KEY,
        event_id TEXT,
        event_name TEXT,
        date TEXT,
        location TEXT,
        division TEXT,
        title_fight INTEGER,
        method TEXT,
        finish_round INTEGER,
        match_time_sec REAL,
        total_rounds INTEGER,
        referee TEXT,
        r_fighter_id TEXT,
        b_fighter_id TEXT,
        winner_id TEXT,
        FOREIGN KEY (r_fighter_id) REFERENCES fighters(fighter_id),
        FOREIGN KEY (b_fighter_id) REFERENCES fighters(fighter_id),
        FOREIGN KEY (winner_id) REFERENCES fighters(fighter_id)
    );
    """
    # Preferential copy columns in this order if present in old table
    fights_copy_cols = [
        'fight_id','event_id','event_name','date','location','division','title_fight',
        'method','finish_round','match_time_sec','total_rounds','referee',
        'r_fighter_id','b_fighter_id','winner_id'
    ]
    if table_exists(cur, "fights"):
        rebuild_table(cur, "fights", fights_create, fights_copy_cols)
    else:
        cur.executescript(fights_create)
        print("Created 'fights' (fresh)")

    conn.commit()

    # --- REBUILD fight_stats ---
    # For fight_stats, we will fetch the old column names (if table exists) and recreate preserving them
    if table_exists(cur, "fight_stats"):
        # get old column list to build create SQL dynamically, but ensure FK at end
        old_cols_info = cur.execute("PRAGMA table_info(fight_stats);").fetchall()
        old_cols = [c[1] for c in old_cols_info]  # (cid, name, type, notnull, dflt, pk)
        # ensure fundamental columns exist; otherwise user will have to inspect
        required = ['fight_id', 'fighter_id', 'corner']
        # We'll construct columns definitions as TEXT for simplicity except numeric-ish names we leave TEXT (safe).
        col_defs = []
        for col in old_cols:
            # ensure corner/fk columns are present
            if col == 'fighter_id':
                col_defs.append("fighter_id TEXT")
            elif col == 'fight_id':
                col_defs.append("fight_id TEXT")
            elif col == 'corner':
                col_defs.append("corner TEXT")
            else:
                # keep name, default TEXT (user-friendly and tolerant)
                safe = col.replace('"','""')
                col_defs.append(f'"{safe}" TEXT')
        cols_sql = ",\n  ".join(col_defs)
        create_stats_sql = f"""
        CREATE TABLE IF NOT EXISTS fight_stats (
          {cols_sql},
          FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
          FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
        );"""
        # use the rebuild helper but pass copy_cols = old_cols (preserve order)
        rebuild_table(cur, "fight_stats", create_stats_sql, old_cols)
    else:
        # If no old table, create a minimal schema
        create_stats_sql = """
        CREATE TABLE IF NOT EXISTS fight_stats (
            fight_id TEXT,
            fighter_id TEXT,
            corner TEXT,
            FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
            FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
        );"""
        cur.executescript(create_stats_sql)
        print("Created 'fight_stats' (fresh minimal)")

    conn.commit()

    # Re-enable FK checks and run integrity check
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()

    fk_violations = cur.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_violations:
        print("\nWARNING: foreign_key_check reported violations (rows that reference missing parents).")
        print("Sample violations (table, rowid, parent):", fk_violations[:10])
        print("You may need to inspect these rows and fix or remove them.")
    else:
        print("\nNo foreign key violations detected. All tables rebuilt successfully.")

    # Print summary counts
    def safe_count(t):
        if table_exists(cur, t):
            return cur.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        return 0

    print("\nSummary:")
    print(" fighters:", safe_count("fighters"))
    print(" fights:", safe_count("fights"))
    print(" fight_stats:", safe_count("fight_stats"))

    conn.close()
    print("\nDone. Backup is at:", bak)

if __name__ == "__main__":
    main()
