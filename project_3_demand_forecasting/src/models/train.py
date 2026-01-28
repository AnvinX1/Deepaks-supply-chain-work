"""
Training Script for Demand Forecasting

Train LSTM, Prophet, or LightGBM models.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_lstm(config: Dict[str, Any]) -> None:
    """Train LSTM model."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import TimeSeriesDataLoader
    from data.time_series_preprocessing import TimeSeriesPreprocessor
    from models.lstm_model import LSTMForecaster
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = TimeSeriesDataLoader(str(data_path))
    df = loader.load()
    
    target = 'warehouse_inventory_level'
    resampled = loader.resample('H', target)
    
    # Preprocess
    preprocessor = TimeSeriesPreprocessor()
    scaled_data = preprocessor.scale_data(resampled[target].values)
    
    # Create sequences
    lstm_config = config.get('models', {}).get('lstm', {})
    lookback = lstm_config.get('lookback', 24)
    horizon = lstm_config.get('forecast_horizon', 1)
    
    X, y = preprocessor.create_sequences(scaled_data, lookback, horizon)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train
    model = LSTMForecaster(
        lookback=lookback,
        forecast_horizon=horizon,
        units=lstm_config.get('units', 64),
        dropout=lstm_config.get('dropout', 0.2)
    )
    
    model.train(
        X_train, y_train, X_val, y_val,
        epochs=lstm_config.get('epochs', 50),
        batch_size=lstm_config.get('batch_size', 32)
    )
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / "models"
    model.save(str(output_dir / "lstm_forecaster.h5"))
    
    logger.info("LSTM training complete!")


def train_prophet(config: Dict[str, Any]) -> None:
    """Train Prophet model."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import TimeSeriesDataLoader
    from models.prophet_model import ProphetForecaster
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = TimeSeriesDataLoader(str(data_path))
    df = loader.load()
    
    # Prepare Prophet format
    target = 'warehouse_inventory_level'
    resampled = loader.resample('H', target)
    
    prophet_df = resampled.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Use 80% for training
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:train_size]
    
    # Train
    prophet_config = config.get('models', {}).get('prophet', {})
    model = ProphetForecaster(
        yearly_seasonality=prophet_config.get('yearly_seasonality', False),
        weekly_seasonality=prophet_config.get('weekly_seasonality', True),
        daily_seasonality=prophet_config.get('daily_seasonality', True)
    )
    
    model.train(train_df)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / "models"
    model.save(str(output_dir / "prophet_model.pkl"))
    
    logger.info("Prophet training complete!")


def train_model(model_type: str = 'lstm') -> None:
    """Train specified model."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    config = load_config(str(config_path)) if config_path.exists() else {}
    
    if model_type == 'lstm':
        train_lstm(config)
    elif model_type == 'prophet':
        train_prophet(config)
    else:
        logger.error(f"Unknown model: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'prophet', 'lgbm'])
    args = parser.parse_args()
    
    train_model(args.model)


if __name__ == "__main__":
    main()
