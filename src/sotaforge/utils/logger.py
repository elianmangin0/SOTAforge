"""Configures and provides a logger for the application."""

import logging
import os
from datetime import datetime


def get_logger(name: str | None = None, log_dir: str = "logs") -> logging.Logger:
    """Initialize and return a configured logger.

    Args:
        name (str): Optional module name for the logger.
        log_dir (str): Directory where log files are saved.

    Returns:
        logging.Logger: Configured logger instance.

    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d}.log")

    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # File handler (capture everything)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler (info and above only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
