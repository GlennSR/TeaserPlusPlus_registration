"""Logging utilities with colored output."""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log level names only."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Save the original levelname
        levelname = record.levelname

        # Add color to levelname only
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname

        return result


def setup_logging(level=logging.INFO):
    """Set up logging configuration with colored output.

    Args:
        level: The logging level (default: logging.INFO).

    Returns:
        The root logger instance.
    """
    # Create formatter with timestamp and level
    formatter = ColoredFormatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)

    return root_logger


# Initialize logging when module is imported
setup_logging()
