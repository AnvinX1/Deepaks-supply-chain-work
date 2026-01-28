"""Helper utilities."""
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

def setup_logging(level=logging.INFO, log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_json(data: Dict, path: str, indent: int = 2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
