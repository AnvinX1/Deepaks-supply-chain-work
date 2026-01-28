"""
Temporal Feature Engineering

Create advanced temporal features for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """Engineer temporal features for forecasting."""
    
    def __init__(self):
        self.feature_names = []
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for temporal features."""
        df = df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour of day (0-23)
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            
            # Day of week (0-6)
            df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # Month (1-12)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        logger.info("Added cyclical temporal features")
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add trend-based features."""
        df = df.copy()
        
        # Difference from previous period
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24'] = df[target_col].diff(24)  # 24h ago
        
        # Percent change
        df[f'{target_col}_pct_change'] = df[target_col].pct_change()
        
        # Expanding mean (historical average)
        df[f'{target_col}_expanding_mean'] = df[target_col].expanding().mean()
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday/special day features."""
        df = df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            # End of month
            df['is_month_end'] = df.index.is_month_end.astype(int)
            
            # Start of month
            df['is_month_start'] = df.index.is_month_start.astype(int)
            
            # Quarter
            df['quarter'] = df.index.quarter
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply all temporal feature engineering."""
        df = self.add_cyclical_features(df)
        df = self.add_trend_features(df, target_col)
        df = self.add_holiday_features(df)
        
        self.feature_names = list(df.columns)
        
        return df
