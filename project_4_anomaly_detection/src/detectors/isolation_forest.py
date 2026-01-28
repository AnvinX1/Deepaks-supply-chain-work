"""
Isolation Forest Anomaly Detector

Tree-based anomaly detection algorithm.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest for anomaly detection."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = None
        self.feature_names = []
    
    def fit(self, X: pd.DataFrame) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.feature_names = list(X.columns)
        self.model.fit(X)
        
        logger.info(f"Fitted Isolation Forest on {len(X)} samples, {len(self.feature_names)} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores.
        
        Lower scores = more anomalous.
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.score_samples(X)
    
    def detect_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies and return results."""
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        results = pd.DataFrame({
            'is_anomaly': (predictions == -1).astype(int),
            'anomaly_score': scores,
            'anomaly_probability': self._score_to_probability(scores)
        }, index=X.index)
        
        n_anomalies = results['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        return results
    
    def _score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to probability-like values."""
        # Normalize scores to 0-1 range (higher = more anomalous)
        min_score, max_score = scores.min(), scores.max()
        if max_score - min_score > 0:
            normalized = (max_score - scores) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores)
        return normalized
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'contamination': self.contamination
            }
        }, path)
        
        logger.info(f"Saved Isolation Forest to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data.get('feature_names', [])
        
        logger.info(f"Loaded Isolation Forest from {path}")


def main():
    """Train Isolation Forest detector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import AnomalyPreprocessor
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    # Select features
    features = ['iot_temperature', 'fuel_consumption_rate', 'traffic_congestion_level',
                'driver_behavior_score', 'fatigue_monitoring_score']
    X = df[[f for f in features if f in df.columns]].copy()
    
    # Preprocess
    preprocessor = AnomalyPreprocessor()
    X_processed = preprocessor.full_pipeline(X, remove_outliers=False)
    
    # Train
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(X_processed)
    
    # Detect
    results = detector.detect_anomalies(X_processed)
    print(f"Anomaly rate: {results['is_anomaly'].mean()*100:.2f}%")
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / "models"
    detector.save(str(output_dir / "isolation_forest.pkl"))


if __name__ == "__main__":
    main()
