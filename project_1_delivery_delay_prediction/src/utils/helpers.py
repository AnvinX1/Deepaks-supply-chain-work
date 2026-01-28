"""
Helper Utilities

Common utility functions used across the project.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        format_string: Log format string
        
    Returns:
        Configured logger
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_json(data: Dict[str, Any], output_path: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    
    # Navigate up until we find the project root (contains config/)
    for parent in current.parents:
        if (parent / 'config').exists():
            return parent
    
    return current.parent.parent.parent


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{name}: {value:.{decimal_places}f}")
        else:
            lines.append(f"{name}: {value}")
    
    return "\n".join(lines)


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Timestamp string (YYYY-MM-DD_HH-MM-SS)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
