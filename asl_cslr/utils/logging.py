"""
Structured logging and TensorBoard helpers.
"""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None,
):
    """Configure structured logging for the project.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file.
        format_string: Custom format string.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
        )

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy libraries
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
