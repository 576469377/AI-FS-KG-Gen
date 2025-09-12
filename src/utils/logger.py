"""
Logging configuration for AI-FS-KG-Gen pipeline
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import LOGS_DIR

def setup_logger(log_level: str = "INFO", log_file: str = "ai_fs_kg_gen.log") -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
    """
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler
    log_path = LOGS_DIR / log_file
    logger.add(
        log_path,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="1 month",
        compression="zip"
    )
    
    logger.info("Logger initialized successfully")

def get_logger(name: str):
    """Get a logger instance for a specific module"""
    return logger.bind(name=name)