"""
Preprocessing for Anomaly Detection

Prepare data for anomaly detection models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyPreprocessor:
    """Preprocess data for anomaly detection."""
    
    def __init__(self, scaler_type: str = 'standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
    
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        
        if not self.is_fitted:
            df_imputed = pd.DataFrame(
                self.imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            df_imputed = pd.DataFrame(
                self.imputer.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        return df_imputed
    
    def scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features."""
        df = df.copy()
        
        if fit:
            scaled = self.scaler.fit_transform(df)
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(df)
        
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)
    
    def inverse_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data."""
        return pd.DataFrame(
            self.scaler.inverse_transform(df),
            columns=df.columns,
            index=df.index
        )
    
    def remove_outliers_for_training(
        self, 
        df: pd.DataFrame, 
        z_threshold: float = 5.0
    ) -> pd.DataFrame:
        """Remove extreme outliers for cleaner training."""
        df = df.copy()
        z_scores = np.abs((df - df.mean()) / df.std())
        mask = (z_scores < z_threshold).all(axis=1)
        
        logger.info(f"Removed {(~mask).sum()} extreme outliers for training")
        
        return df[mask]
    
    def full_pipeline(
        self, 
        df: pd.DataFrame, 
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        df = self.handle_missing(df)
        
        if remove_outliers:
            df = self.remove_outliers_for_training(df)
        
        df = self.scale(df)
        
        return df
