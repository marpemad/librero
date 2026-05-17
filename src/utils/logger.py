"""Logger central — formato consistente y nivel configurable."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger
