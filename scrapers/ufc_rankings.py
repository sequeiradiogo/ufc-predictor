"""
ufc_rankings.py -- Fetch UFC rankings snapshots from the Kaggle dataset
                   martj42/ufc-rankings and store them in a DB rankings table.

Rankings are stored in a separate `rankings` table keyed by
(fighter_id, division, date) so feature engineering can join the most recent
ranking prior to each fight.

Requires the Kaggle API credentials to be configured:
  ~/.kaggle/kaggle.json   or   KAGGLE_USERNAME / KAGGLE_KEY env vars

Non-blocking: if Kaggle is not available a warning is logged and the caller
continues without rankings data.

Schema created if not present:
    rankings(fighter_id TEXT, division TEXT, rank INTEGER, date TEXT,
             PRIMARY KEY (fighter_id, division, date))
"""

import difflib
import io
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DB_PATH
from logger import get_logger

log = get_logger(__name__)

_DATASET   = "martj42/ufc-rankings"
_TABLE     = "rankings"
_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    fighter_id TEXT NOT NULL,
    division   TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    date       TEXT NOT NULL,
    PRIMARY KEY (fighter_id, division, date),
    FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
)
"""


# ── Kaggle download ───────────────────────────────────────────────────────────

def _download_rankings_csv() -> Path | None:
    """
    Download the martj42/ufc-rankings dataset via the Kaggle API.
    Returns the path to the extracted CSV, or None on failure.
    """
    try:
        import kaggle  # noqa: F401 -- triggers auth on import
        kaggle.api.authenticate()
    except Exception as exc:
        log.warning("Kaggle API not available (%s) -- skipping rankings", exc)
        return None

    tmp = Path(tempfile.mkdtemp(prefix="ufc_rankings_"))
    try:
        import kaggle
        kaggle.api.dataset_download_files(_DATASET, path=str(tmp), unzip=False, quiet=True)
        # Find the zip
        zips = list(tmp.glob("*.zip"))
        if not zips:
            log.warning("No zip found after Kaggle download -- skipping rankings")
            return None
        with zipfile.ZipFile(zips[0]) as zf:
            csvs = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csvs:
                log.warning("No CSV inside Kaggle zip -- skipping rankings")
                return None
            zf.extract(csvs[0], tmp)
            return tmp / csvs[0]
    except Exception as exc:
        log.warning("Kaggle download failed: %s -- skipping rankings", exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return None


# ── Name matching ─────────────────────────────────────────────────────────────

def _name_key(name: str) -> str:
    return re.sub(r"[^a-z ]", "", (name or "").lower()).strip()


def _load_fighter_name_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {name_key: fighter_id} for all fighters in the DB."""
    rows = conn.execute("SELECT fighter_id, name FROM fighters").fetchall()
    return {_name_key(r[1]): r[0] for r in rows if r[1]}


def _resolve_fighter_id(name: str, name_map: dict[str, str]) -> str | None:
    key = _name_key(name)
    if key in name_map:
        return name_map[key]
    matches = difflib.get_close_matches(key, name_map.keys(), n=1, cutoff=0.82)
    return name_map[matches[0]] if matches else None


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _parse_rankings_csv(csv_path: Path) -> list[dict]:
    """
    Parse the rankings CSV into a list of
    {fighter_id_hint, division, rank, date} dicts.

    The martj42/ufc-rankings dataset is expected to have columns like:
        fighter_name, weight_class, rank, date
    Column names are normalised case-insensitively.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available for rankings parsing")
        return []

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read rankings CSV: %s", exc)
        return []

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    name_col = next(
        (c for c in df.columns if "name" in c or "fighter" in c), None
    )
    div_col  = next(
        (c for c in df.columns if "weight" in c or "class" in c or "division" in c), None
    )
    rank_col = next(
        (c for c in df.columns if "rank" in c), None
    )
    date_col = next(
        (c for c in df.columns if "date" in c), None
    )

    if not all([name_col, rank_col, date_col]):
        log.warning("Rankings CSV missing expected columns: %s", list(df.columns))
        return []

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name":     str(r[name_col]) if name_col else "",
            "division": str(r[div_col]).lower().strip() if div_col else "unknown",
            "rank":     int(r[rank_col]) if pd.notna(r[rank_col]) else None,
            "date":     str(r[date_col])[:10] if date_col else None,
        })
    return rows


# ── DB upsert ─────────────────────────────────────────────────────────────────

def _upsert_rankings(rows: list[dict], conn: sqlite3.Connection) -> int:
    """Upsert resolved ranking rows into the rankings table. Returns inserted count."""
    name_map = _load_fighter_name_map(conn)
    resolved = []
    unmatched = 0
    for row in rows:
        if row["rank"] is None or row["date"] is None:
            continue
        fid = _resolve_fighter_id(row["name"], name_map)
        if not fid:
            unmatched += 1
            continue
        resolved.append((fid, row["division"], row["rank"], row["date"]))

    if unmatched:
        log.debug("Rankings: %d fighter names could not be matched to DB", unmatched)

    conn.executemany(
        f"INSERT OR REPLACE INTO {_TABLE} (fighter_id, division, rank, date) VALUES (?,?,?,?)",
        resolved,
    )
    return len(resolved)


# ── Public API ────────────────────────────────────────────────────────────────

def refresh_rankings(db_path: Path = DB_PATH) -> bool:
    """
    Download the latest UFC rankings from Kaggle and upsert into the DB.

    Returns True if rankings were updated, False if skipped.
    Called from refresh_data.py after inserting new fights.
    """
    csv_path = _download_rankings_csv()
    if csv_path is None:
        return False

    try:
        rows = _parse_rankings_csv(csv_path)
        if not rows:
            return False

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(_CREATE_SQL)
        n = _upsert_rankings(rows, conn)
        conn.commit()
        conn.close()
        log.info("Rankings: upserted %d rows into %s table", n, _TABLE)
        return True
    except Exception as exc:
        log.warning("Rankings update failed: %s", exc)
        return False
    finally:
        # Clean up temp download
        try:
            if csv_path:
                shutil.rmtree(csv_path.parent, ignore_errors=True)
        except Exception:
            pass


def ensure_rankings_table(conn: sqlite3.Connection) -> None:
    """Create the rankings table if it doesn't exist. Safe to call repeatedly."""
    conn.execute(_CREATE_SQL)
