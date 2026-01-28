"""
Time Series Preprocessing

Create lag features, rolling statistics, and sequences for LSTM.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocess time series data."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lag features."""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features")
        
        return df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        windows: List[int],
        stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """Create rolling statistics."""
        df = df.copy()
        
        for window in windows:
            rolling = df[target_col].rolling(window=window, min_periods=1)
            
            if 'mean' in stats:
                df[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
            if 'std' in stats:
                df[f'{target_col}_rolling_std_{window}'] = rolling.std()
            if 'min' in stats:
                df[f'{target_col}_rolling_min_{window}'] = rolling.min()
            if 'max' in stats:
                df[f'{target_col}_rolling_max_{window}'] = rolling.max()
        
        logger.info(f"Created rolling features for windows: {windows}")
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from datetime index."""
        df = df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return df
    
    def scale_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale data to 0-1 range."""
        if fit:
            scaled = self.scaler.fit_transform(data.reshape(-1, 1))
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(data.reshape(-1, 1))
        
        return scaled.flatten()
    
    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        lookback: int, 
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(data[i:i + forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: List[int] = [1, 6, 12, 24],
        windows: List[int] = [6, 12, 24]
    ) -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        df = self.create_lag_features(df, target_col, lags)
        df = self.create_rolling_features(df, target_col, windows)
        df = self.add_temporal_features(df)
        
        # Drop rows with NaN from lag features
        df = df.dropna()
        
        return df
