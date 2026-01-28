"""
Data Loading Utilities

Helper functions to load models and data across different projects.
"""

import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path
import sys

# Add project roots to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

@st.cache_data
def load_main_dataset():
    """Load the main logistics dataset."""
    data_path = PROJECT_ROOT / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_model(project_dir, model_name):
    """Load a trained model artifact."""
    model_path = PROJECT_ROOT / project_dir / "models" / model_name
    if model_path.exists():
        return joblib.load(model_path)
    return None

def load_config(project_dir):
    """Load project config."""
    config_path = PROJECT_ROOT / project_dir / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}
