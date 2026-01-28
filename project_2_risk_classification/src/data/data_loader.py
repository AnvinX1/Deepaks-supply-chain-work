"""
Data Loader for Risk Classification

Loads and validates supply chain data with risk labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for risk classification."""
    
    CLASS_MAPPING = {
        'Low Risk': 0,
        'Moderate Risk': 1,
        'High Risk': 2
    }
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """Load data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records")
        
        return self.df
    
    def encode_target(self, target_col: str = 'risk_classification') -> pd.Series:
        """Encode risk labels to numeric values."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        encoded = self.df[target_col].map(self.CLASS_MAPPING)
        
        # Log class distribution
        dist = encoded.value_counts(normalize=True)
        logger.info(f"Class distribution: {dist.to_dict()}")
        
        return encoded
    
    def get_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        counts = self.df['risk_classification'].value_counts()
        total = len(self.df)
        n_classes = len(counts)
        
        weights = {}
        for label, count in counts.items():
            class_idx = self.CLASS_MAPPING[label]
            weights[class_idx] = total / (n_classes * count)
        
        return weights


if __name__ == "__main__":
    loader = DataLoader("../../data/dynamic_supply_chain_logistics_dataset.csv")
    df = loader.load()
    y = loader.encode_target()
    weights = loader.get_class_weights()
    print(f"Class weights: {weights}")
