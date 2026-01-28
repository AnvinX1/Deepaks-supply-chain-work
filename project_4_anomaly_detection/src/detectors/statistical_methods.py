"""
Statistical Anomaly Detection Methods

Simple statistical approaches for anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.stats = {}
    
    def fit(self, X: pd.DataFrame) -> 'StatisticalDetector':
        """Calculate statistics from training data."""
        self.stats = {
            'mean': X.mean(),
            'std': X.std(),
            'median': X.median(),
            'q1': X.quantile(0.25),
            'q3': X.quantile(0.75),
            'iqr': X.quantile(0.75) - X.quantile(0.25)
        }
        
        logger.info(f"Fitted statistical detector on {len(X)} samples")
        
        return self
    
    def detect_zscore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Z-score."""
        z_scores = np.abs((X - self.stats['mean']) / self.stats['std'])
        
        # Anomaly if any feature exceeds threshold
        is_anomaly = (z_scores > self.z_score_threshold).any(axis=1)
        max_z = z_scores.max(axis=1)
        
        return pd.DataFrame({
            'is_anomaly': is_anomaly.astype(int),
            'max_z_score': max_z,
            'anomaly_score': max_z / self.z_score_threshold
        }, index=X.index)
    
    def detect_iqr(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using IQR method."""
        lower_bound = self.stats['q1'] - self.iqr_multiplier * self.stats['iqr']
        upper_bound = self.stats['q3'] + self.iqr_multiplier * self.stats['iqr']
        
        is_below = (X < lower_bound).any(axis=1)
        is_above = (X > upper_bound).any(axis=1)
        is_anomaly = is_below | is_above
        
        # Calculate how far outside bounds
        below_diff = (lower_bound - X).clip(lower=0).max(axis=1)
        above_diff = (X - upper_bound).clip(lower=0).max(axis=1)
        max_diff = np.maximum(below_diff, above_diff)
        
        return pd.DataFrame({
            'is_anomaly': is_anomaly.astype(int),
            'max_deviation': max_diff,
            'anomaly_score': max_diff / (self.stats['iqr'].mean() + 1e-10)
        }, index=X.index)
    
    def detect_combined(self, X: pd.DataFrame) -> pd.DataFrame:
        """Combine Z-score and IQR methods."""
        zscore_results = self.detect_zscore(X)
        iqr_results = self.detect_iqr(X)
        
        # Anomaly if detected by either method
        is_anomaly = (zscore_results['is_anomaly'] | iqr_results['is_anomaly']).astype(int)
        
        # Combined score
        combined_score = (zscore_results['anomaly_score'] + iqr_results['anomaly_score']) / 2
        
        return pd.DataFrame({
            'is_anomaly': is_anomaly,
            'zscore_anomaly': zscore_results['is_anomaly'],
            'iqr_anomaly': iqr_results['is_anomaly'],
            'combined_score': combined_score
        }, index=X.index)


def main():
    """Test statistical detector."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    features = ['iot_temperature', 'fuel_consumption_rate']
    X = df[[f for f in features if f in df.columns]].fillna(0)
    
    detector = StatisticalDetector()
    detector.fit(X)
    
    results = detector.detect_combined(X)
    print(f"Z-score anomalies: {results['zscore_anomaly'].sum()}")
    print(f"IQR anomalies: {results['iqr_anomaly'].sum()}")
    print(f"Combined anomalies: {results['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()
