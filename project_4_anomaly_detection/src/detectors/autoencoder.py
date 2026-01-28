"""
Autoencoder Anomaly Detector

Deep learning based anomaly detection using reconstruction error.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoencoderDetector:
    """Autoencoder-based anomaly detection."""
    
    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: List[int] = [32, 16],
        epochs: int = 50,
        batch_size: int = 32,
        threshold_percentile: float = 95
    ):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.encoder = None
        self.threshold = None
        self.input_dim = None
    
    def build_model(self, input_dim: int) -> Any:
        """Build autoencoder architecture."""
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Dropout
        except ImportError:
            logger.error("TensorFlow not installed")
            raise
        
        self.input_dim = input_dim
        
        # Encoder
        inputs = Input(shape=(input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.1)(x)
        
        encoded = Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = encoded
        for units in reversed(self.hidden_layers):
            x = Dense(units, activation='relu')(x)
        
        decoded = Dense(input_dim, activation='linear')(x)
        
        # Models
        self.model = Model(inputs, decoded, name='autoencoder')
        self.encoder = Model(inputs, encoded, name='encoder')
        
        self.model.compile(optimizer='adam', loss='mse')
        
        logger.info(f"Built Autoencoder: {input_dim} -> {self.encoding_dim} -> {input_dim}")
        
        return self.model
    
    def fit(self, X: pd.DataFrame, validation_split: float = 0.15) -> Dict[str, Any]:
        """Train the autoencoder."""
        if self.model is None:
            self.build_model(X.shape[1])
        
        history = self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Calculate reconstruction threshold
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)
        
        logger.info(f"Training complete. Threshold: {self.threshold:.4f}")
        
        return history.history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        return (mse > self.threshold).astype(int)
    
    def get_reconstruction_error(self, X: pd.DataFrame) -> np.ndarray:
        """Get reconstruction error for each sample."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - reconstructions, 2), axis=1)
    
    def detect_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies with scores."""
        mse = self.get_reconstruction_error(X)
        predictions = (mse > self.threshold).astype(int)
        
        # Normalize MSE to probability-like score
        normalized = mse / (mse.max() + 1e-10)
        
        results = pd.DataFrame({
            'is_anomaly': predictions,
            'reconstruction_error': mse,
            'anomaly_score': normalized
        }, index=X.index)
        
        n_anomalies = results['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        return results
    
    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        
        # Save threshold separately
        import joblib
        joblib.dump({'threshold': self.threshold, 'input_dim': self.input_dim},
                    str(Path(path).with_suffix('.meta')))
        
        logger.info(f"Saved Autoencoder to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        from tensorflow.keras.models import load_model
        import joblib
        
        self.model = load_model(path)
        
        meta_path = str(Path(path).with_suffix('.meta'))
        if Path(meta_path).exists():
            meta = joblib.load(meta_path)
            self.threshold = meta.get('threshold')
            self.input_dim = meta.get('input_dim')
        
        logger.info(f"Loaded Autoencoder from {path}")


def main():
    """Train Autoencoder detector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import AnomalyPreprocessor
    
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    features = ['iot_temperature', 'fuel_consumption_rate', 'traffic_congestion_level',
                'driver_behavior_score', 'fatigue_monitoring_score']
    X = df[[f for f in features if f in df.columns]].copy()
    
    preprocessor = AnomalyPreprocessor()
    X_processed = preprocessor.full_pipeline(X)
    
    detector = AutoencoderDetector(encoding_dim=4, epochs=30)
    detector.fit(X_processed)
    
    results = detector.detect_anomalies(X_processed)
    print(f"Anomaly rate: {results['is_anomaly'].mean()*100:.2f}%")
    
    output_dir = Path(__file__).parent.parent.parent / "models"
    detector.save(str(output_dir / "autoencoder.h5"))


if __name__ == "__main__":
    main()
