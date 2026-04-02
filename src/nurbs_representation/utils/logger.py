import logging
import os
import sys
from colorama import Fore, Style, init

# Initialize colorama (to enable color on Windows, etc.)
init(autoreset=True)

# Define a custom log level for "SUCCESS"
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, msg, *args, obj=None, **kwargs):
    """
    Custom log method for success messages at level 25.
    Optionally append an object's string representation.
    """
    if obj is not None:
        msg = f"{msg} - {obj}"
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, msg, args, **kwargs)

# Extend the built-in logging.Logger class to add our 'success' method
logging.Logger.success = success

class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color to console logs based on the log level.
    """
    LOG_COLORS = {
        logging.DEBUG: Fore.WHITE,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
        SUCCESS_LEVEL_NUM: Fore.GREEN,
    }

    def format(self, record):
        original_levelname = record.levelname
        color = self.LOG_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        formatted = super().format(record)
        # Restore so file logs are not color-coded
        record.levelname = original_levelname
        return formatted

class CustomLogger(logging.Logger):
    """
    A custom logger that:
    1. Inherits from logging.Logger
    2. Provides an optional 'obj' parameter for each log method
    """
    
    def debug(self, msg, *args, obj=None, **kwargs):
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, obj=None, **kwargs):
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, obj=None, **kwargs):
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, obj=None, **kwargs):
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, obj=None, **kwargs):
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().critical(msg, *args, **kwargs)

    def exception(self, msg, *args, obj=None, exc_info=True, **kwargs):
        """
        Log an ERROR message with exception information.
        """
        if obj is not None:
            msg = f"{msg} - {obj}"
        super().exception(msg, *args, exc_info=exc_info, **kwargs)

    def success(self, msg, *args, obj=None, **kwargs):
        """
        Custom success log method (level 25).
        """
        if obj is not None:
            msg = f"{msg} - {obj}"
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, msg, args, **kwargs)

_logger_instance = None

def setup_logger(name="MyLogger", log_file="app.log", level=logging.DEBUG):
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_folder = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_file)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _logger_instance = logger
    return logger