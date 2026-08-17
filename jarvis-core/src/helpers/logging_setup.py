"""Configuration du logging Jarvis et fabrique de loggers nommés.

Racine du paquet helpers : tous les autres sous-modules tirent `get_logger` d'ici.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s  %(name)-24s  %(levelname)s  %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def setup_logging(log_file: str = "/opt/jarvis/logs/jarvis-api.log") -> None:
    """
    Configure the root Jarvis logger once: two rotating file handlers.
    - jarvis-api.log  : INFO+  (5 MB × 3, operational)
    - jarvis-debug.log: DEBUG+ (10 MB × 2, verbose — for review)
    Safe to call multiple times (no-op after first call).
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # INFO+ rotating file: 5 MB × 3 backups
    fh = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # DEBUG+ rotating file: 10 MB × 2 backups
    debug_file = os.path.join(log_dir, "jarvis-debug.log")
    dfh = RotatingFileHandler(
        debug_file, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(fmt)
    root.addHandler(dfh)

    # Quiet noisy third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "primp",
        "sentence_transformers",
        "apscheduler",
        "urllib3",
        "asyncio",
        "rustls",
        "hyper_util",
        "h2",
        "reqwest",
        "hyper",
        "ddgs",
        "ddgs.ddgs",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress known-spurious warnings that are harmless but noisy
    for suppress in (
        "ddgs.engines.yahoo_news",  # IndexError in post-processing — library catches it, results still returned
    ):
        logging.getLogger(suppress).setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """Return a named Jarvis logger. Thin wrapper around logging.getLogger."""
    return logging.getLogger(name)
