"""
db/backup.py -- timestamped DB backups with rotation.

Every full rolling.py rebuild copies the DB first; without rotation those
copies piled up locally (90 files / 2.4 GB by 2026-11). History now lives in
the dated `data-YYYY-MM-DD` GitHub releases created by monthly-refresh.yml
(issue #70), so only the last DB_BACKUPS_KEEP local copies are kept.

Only automatic backups (`<stem>_backup_YYYYMMDD_HHMMSS<suffix>`) are rotated;
hand-named copies such as `ufc_v2_backup_local_20261009.db` are never touched.
"""

import datetime
import re
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_BACKUPS_KEEP

_STAMP_RE = re.compile(r"_backup_\d{8}_\d{6}$")


def _auto_backups(db_path: Path) -> list[Path]:
    """Automatic backups of db_path, oldest first (the timestamp sorts lexically)."""
    candidates = db_path.parent.glob(f"{db_path.stem}_backup_*{db_path.suffix}")
    return sorted(p for p in candidates if _STAMP_RE.search(p.stem[len(db_path.stem):]))


def rotate_backups(db_path: Path, keep: int = DB_BACKUPS_KEEP) -> list[Path]:
    """Delete all but the newest `keep` automatic backups of db_path. Returns the deleted paths."""
    backups = _auto_backups(db_path)
    stale = backups[:-keep] if keep > 0 else backups
    for p in stale:
        p.unlink()
    return stale


def backup_db(db_path: Path, keep: int = DB_BACKUPS_KEEP) -> Path:
    """Copy db_path to `<stem>_backup_<timestamp><suffix>`, then rotate old backups."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(db_path.stem + f"_backup_{stamp}" + db_path.suffix)
    shutil.copy2(db_path, backup)
    rotate_backups(db_path, keep=max(keep, 1))  # never delete the copy just made
    return backup
