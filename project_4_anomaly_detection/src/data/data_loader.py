"""
Data Loader for Anomaly Detection

Load and prepare data for anomaly detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load data for anomaly detection."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """Load data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records")
        
        return self.df
    
    def get_feature_subset(self, feature_groups: dict) -> pd.DataFrame:
        """Get subset of features for anomaly detection."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        all_features = []
        for group, features in feature_groups.items():
            for f in features:
                if f in self.df.columns:
                    all_features.append(f)
        
        return self.df[all_features].copy()
    
    def get_stats(self, columns: List[str]) -> pd.DataFrame:
        """Get statistics for specified columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        existing = [c for c in columns if c in self.df.columns]
        return self.df[existing].describe()
