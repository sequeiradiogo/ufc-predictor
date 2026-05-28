"""
logger.py — Shared logging setup for the UFC Predictor project.

Usage in any script:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Loading data...")
    log.warning("Missing value in column X")
    log.error("Database not found")

All logs are written to both the console and logs/ufc_predictor.log.
"""

import logging
from pathlib import Path

# Create logs directory next to this file
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "ufc_predictor.log"

_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
_DATE   = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger named *name* configured with a console handler and a
    rotating file handler.  Safe to call multiple times — handlers are only
    attached once.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured for this name

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE)

    # ── Console ──────────────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ── File ─────────────────────────────────────────────────────────────────
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # always capture DEBUG to file even if console is INFO
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
