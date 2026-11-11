"""
tests/test_backup.py -- db/backup.py rotation of automatic DB backups (issue #70).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from db.backup import backup_db, rotate_backups


def _touch(path: Path) -> Path:
    path.write_bytes(b"x")
    return path


class TestRotateBackups:
    def test_keeps_newest_n(self, tmp_path):
        db = _touch(tmp_path / "ufc.db")
        stamps = [f"202611{d:02d}_120000" for d in range(1, 8)]
        for s in stamps:
            _touch(tmp_path / f"ufc_backup_{s}.db")

        deleted = rotate_backups(db, keep=3)

        assert sorted(p.name for p in deleted) == [f"ufc_backup_{s}.db" for s in stamps[:4]]
        left = sorted(p.name for p in tmp_path.glob("ufc_backup_*.db"))
        assert left == [f"ufc_backup_{s}.db" for s in stamps[4:]]
        assert db.exists()

    def test_ignores_hand_named_and_other_dbs(self, tmp_path):
        db = _touch(tmp_path / "ufc.db")
        manual = _touch(tmp_path / "ufc_backup_local_20261009.db")
        other = _touch(tmp_path / "ufc_v2_backup_20260101_000000.db")
        other_stem = _touch(tmp_path / "ufcx_backup_20260101_000000.db")
        _touch(tmp_path / "ufc_backup_20260101_000000.db")

        rotate_backups(db, keep=0)

        assert manual.exists() and other.exists() and other_stem.exists()
        assert not (tmp_path / "ufc_backup_20260101_000000.db").exists()


class TestBackupDb:
    def test_copies_and_rotates(self, tmp_path):
        db = tmp_path / "ufc.db"
        db.write_bytes(b"payload")
        for d in range(1, 6):
            _touch(tmp_path / f"ufc_backup_202601{d:02d}_000000.db")

        backup = backup_db(db, keep=2)

        assert backup.read_bytes() == b"payload"
        left = sorted(p.name for p in tmp_path.glob("ufc_backup_*.db"))
        assert left == ["ufc_backup_20260105_000000.db", backup.name]

    def test_never_deletes_the_new_backup(self, tmp_path):
        db = tmp_path / "ufc.db"
        db.write_bytes(b"payload")
        backup = backup_db(db, keep=0)
        assert backup.exists()
