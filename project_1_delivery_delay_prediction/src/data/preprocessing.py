"""
Data Preprocessing Module

Handles data cleaning, transformation, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess supply chain data for ML modeling.
    
    Handles:
    - Missing value imputation
    - Feature scaling
    - Temporal feature extraction
    - Train/test data preparation
    """
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            
            logger.info("Extracted temporal features: hour, day_of_week, day_of_month, month, is_weekend, is_night")
        
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        numerical_cols: List[str],
        strategy: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values in numerical columns.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            strategy: Imputation strategy ('median', 'mean', 'most_frequent')
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Filter to only existing columns
        existing_cols = [col for col in numerical_cols if col in df.columns]
        
        missing_before = df[existing_cols].isnull().sum().sum()
        
        if not self.is_fitted:
            self.imputer = SimpleImputer(strategy=strategy)
            df[existing_cols] = self.imputer.fit_transform(df[existing_cols])
        else:
            df[existing_cols] = self.imputer.transform(df[existing_cols])
        
        missing_after = df[existing_cols].isnull().sum().sum()
        logger.info(f"Imputed missing values: {missing_before} -> {missing_after}")
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Filter to only existing columns
        existing_cols = [col for col in columns if col in df.columns]
        
        if fit:
            df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
            self.is_fitted = True
        else:
            df[existing_cols] = self.scaler.transform(df[existing_cols])
        
        logger.info(f"Scaled {len(existing_cols)} features")
        
        return df
    
    def create_delay_label(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Create binary delay label from delay probability.
        
        Args:
            df: Input DataFrame
            threshold: Probability threshold for delay classification
            
        Returns:
            DataFrame with 'is_delayed' column
        """
        df = df.copy()
        
        if 'delay_probability' in df.columns:
            df['is_delayed'] = (df['delay_probability'] >= threshold).astype(int)
            
            delay_rate = df['is_delayed'].mean()
            logger.info(f"Created delay label. Delay rate: {delay_rate:.2%}")
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y)
        """
        # Filter to existing columns
        existing_features = [col for col in feature_cols if col in df.columns]
        
        X = df[existing_features].copy()
        y = df[target_col].copy()
        
        logger.info(f"Prepared {len(existing_features)} features, target: {target_col}")
        
        return X, y
    
    def full_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        target_classification: str = 'is_delayed',
        target_regression: str = 'delivery_time_deviation',
        delay_threshold: float = 0.5,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns
            target_classification: Classification target column
            target_regression: Regression target column
            delay_threshold: Threshold for delay classification
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (X, y_classification, y_regression)
        """
        # Extract temporal features
        df = self.extract_temporal_features(df)
        
        # Create delay label
        df = self.create_delay_label(df, threshold=delay_threshold)
        
        # Add temporal features to numerical columns
        temporal_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_night']
        all_numerical = numerical_cols + temporal_features
        
        # Handle missing values
        df = self.handle_missing_values(df, all_numerical)
        
        # Scale features
        df = self.scale_features(df, all_numerical, fit=fit)
        
        # Prepare features
        X = df[[col for col in all_numerical if col in df.columns]]
        y_class = df[target_classification] if target_classification in df.columns else None
        y_reg = df[target_regression] if target_regression in df.columns else None
        
        return X, y_class, y_reg


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    loader = DataLoader("../../data/dynamic_supply_chain_logistics_dataset.csv")
    df = loader.load()
    
    preprocessor = DataPreprocessor()
    
    numerical_cols = [
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level',
        'warehouse_inventory_level', 'loading_unloading_time', 'weather_condition_severity'
    ]
    
    X, y_class, y_reg = preprocessor.full_preprocessing_pipeline(df, numerical_cols)
    print(f"Features shape: {X.shape}")
    print(f"Classification target shape: {y_class.shape}")
    print(f"Regression target shape: {y_reg.shape}")
