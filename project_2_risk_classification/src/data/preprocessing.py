"""
Data Preprocessing for Risk Classification

Handles feature scaling, encoding, and class balancing.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for risk classification."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from timestamp."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Impute missing values."""
        df = df.copy()
        existing = [c for c in columns if c in df.columns]
        
        if not self.is_fitted:
            df[existing] = self.imputer.fit_transform(df[existing])
        else:
            df[existing] = self.imputer.transform(df[existing])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        existing = [c for c in columns if c in df.columns]
        
        if fit:
            df[existing] = self.scaler.fit_transform(df[existing])
            self.is_fitted = True
        else:
            df[existing] = self.scaler.transform(df[existing])
        
        return df
    
    def balance_classes(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using SMOTE or other methods."""
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        else:
            return X, y
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        logger.info(f"Balanced dataset: {len(X)} -> {len(X_balanced)} samples")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        balance: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Run full preprocessing pipeline."""
        # Extract temporal features
        df = self.extract_temporal_features(df)
        
        # Add temporal to feature list
        temporal = ['hour', 'day_of_week', 'month', 'is_weekend']
        all_features = feature_cols + [f for f in temporal if f in df.columns]
        
        # Handle missing values
        df = self.handle_missing_values(df, all_features)
        
        # Prepare X and y
        existing_features = [c for c in all_features if c in df.columns]
        X = df[existing_features].copy()
        y = df[target_col].copy()
        
        # Scale features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Balance classes if needed
        if balance:
            X, y = self.balance_classes(X, y, method='smote')
        
        return X, y
