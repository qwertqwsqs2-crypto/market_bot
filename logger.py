# logger.py - FIXED VERSION with idempotent setup
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_INITIALIZED = False


def setup_logging(log_dir: Path = Path("logs"), level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    Safe to call multiple times - will only initialize once.
    """
    global _LOGGER_INITIALIZED

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "market_bot.log"

    logger = logging.getLogger("market_bot")

    # If already initialized, just return existing logger
    if _LOGGER_INITIALIZED and logger.handlers:
        logger.info("Logger already initialized, reusing existing configuration")
        return logger

    # First time setup - clear any old handlers and configure
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        fh = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.info("Logging initialized: %s", log_file)

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        _LOGGER_INITIALIZED = True

    except Exception as e:
        logger.error("Failed to create file handler: %s", e)

    return logger


def add_gui_handler(text_widget, logger_name: str = "market_bot") -> logging.Handler:
    """
    Add a GUI text widget handler to an existing logger.
    Returns the handler so it can be removed later.
    """
    # Import here to avoid circular dependency
    try:
        from gui.runtime_window import TkTextHandler
    except ImportError:
        # Fallback if runtime_window not available
        logging.getLogger(__name__).warning("TkTextHandler not available")
        return None

    logger = logging.getLogger(logger_name)

    # Remove any old GUI handlers first (important for re-runs)
    for handler in logger.handlers[:]:  # Use slice to avoid modification during iteration
        if isinstance(handler, TkTextHandler):
            logger.removeHandler(handler)

    # Add new GUI handler
    gui_handler = TkTextHandler(text_widget)
    gui_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(gui_handler)

    return gui_handler