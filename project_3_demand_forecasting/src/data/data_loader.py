"""
Time Series Data Loader

Load and prepare data for time series forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """Load time series data for forecasting."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load(self, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """Load and parse datetime index."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, parse_dates=[datetime_col])
        self.df = self.df.sort_values(datetime_col).reset_index(drop=True)
        self.df.set_index(datetime_col, inplace=True)
        
        logger.info(f"Loaded {len(self.df)} records from {self.df.index.min()} to {self.df.index.max()}")
        
        return self.df
    
    def resample(self, rule: str = 'H', target_col: str = 'warehouse_inventory_level') -> pd.DataFrame:
        """Resample to regular frequency."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        # Aggregate to specified frequency
        resampled = self.df[[target_col]].resample(rule).mean()
        
        # Forward fill missing values
        resampled = resampled.fillna(method='ffill')
        
        logger.info(f"Resampled to {rule} frequency: {len(resampled)} records")
        
        return resampled
    
    def get_train_test_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split time series preserving temporal order."""
        n = len(df)
        split_idx = int(n * (1 - test_size))
        
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train)}, Test: {len(test)}")
        
        return train, test
    
    def prepare_prophet_data(self, target_col: str) -> pd.DataFrame:
        """Prepare data for Prophet format (ds, y)."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        prophet_df = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[target_col].values
        })
        
        return prophet_df


if __name__ == "__main__":
    loader = TimeSeriesDataLoader("../../data/dynamic_supply_chain_logistics_dataset.csv")
    df = loader.load()
    resampled = loader.resample('H', 'warehouse_inventory_level')
    train, test = loader.get_train_test_split(resampled)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
