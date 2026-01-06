#!/usr/bin/env python3
"""
Logging configuration and utilities for the dictionary viewer.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(output_dir: Path = None, debug: bool = False, quiet: bool = False):
    """
    Set up logging to file and console.
    
    Args:
        output_dir: Directory for log files (default: current directory)
        debug: Enable debug mode
        quiet: Quiet mode (errors only)
    
    Returns:
        Path to log file
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if debug else (logging.ERROR if quiet else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation (keep last 7 days)
    file_handler = RotatingFileHandler(
        log_file,
        encoding='utf-8',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=7
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

