"""
LSTM Forecaster

Deep learning model for time series forecasting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMForecaster:
    """LSTM-based time series forecaster."""
    
    def __init__(
        self,
        lookback: int = 24,
        forecast_horizon: int = 1,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            raise
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built LSTM model: {model.count_params()} parameters")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10
    ) -> Dict[str, Any]:
        """Train the LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training complete. Final loss: {self.history.history['loss'][-1]:.4f}")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, verbose=0)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        from tensorflow.keras.models import load_model
        
        self.model = load_model(path)
        logger.info(f"Loaded model from {path}")
