"""
utils/logger.py
===============
Structured logging architecture with rotating files.
Outputs separate logs for debug trace, runtime events, navigation, and safety.
"""
import logging
import logging.handlers
import os
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def add_file_handler(filename: str, handler_level: int, max_mb: int, backup: int):
        fpath = os.path.join(LOG_DIR, filename)
        handler = logging.handlers.RotatingFileHandler(
            fpath, maxBytes=max_mb * 1024 * 1024, backupCount=backup
        )
        handler.setLevel(handler_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Runtime & Debug logs (all modules)
    add_file_handler("runtime.log", logging.INFO, 5, 5)
    add_file_handler("debug.log", logging.DEBUG, 10, 3)
    
    # Specific specialized logs
    if "navigation" in name or "controller" in name:
        add_file_handler("navigation.log", logging.INFO, 5, 5)
        
    if "safety" in name or "camera" in name:
        add_file_handler("safety.log", logging.INFO, 5, 5)

    return logger

def get_logger(name: str) -> logging.Logger:
    return setup_logger(name, level=logging.INFO)
