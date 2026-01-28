"""
Evaluation Script for Demand Forecasting

Evaluate forecast accuracy and generate metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate forecast metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mape': float(mape)
    }


def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    save_path: Optional[str] = None
):
    """Plot forecast against actual values."""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Scatter')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.close()


def evaluate_forecast(
    model_type: str = 'lstm',
    horizon: int = 24
) -> Dict[str, Any]:
    """Evaluate forecast model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import TimeSeriesDataLoader
    from data.time_series_preprocessing import TimeSeriesPreprocessor
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = TimeSeriesDataLoader(str(data_path))
    df = loader.load()
    
    target = 'warehouse_inventory_level'
    resampled = loader.resample('H', target)
    
    preprocessor = TimeSeriesPreprocessor()
    
    if model_type == 'lstm':
        from models.lstm_model import LSTMForecaster
        
        model_path = Path(__file__).parent.parent.parent / "models" / "lstm_forecaster.h5"
        if not model_path.exists():
            logger.error("LSTM model not found. Train first.")
            return {}
        
        model = LSTMForecaster()
        model.load(str(model_path))
        
        # Prepare test data
        scaled = preprocessor.scale_data(resampled[target].values)
        X, y = preprocessor.create_sequences(scaled, 24, 1)
        
        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]
        
        y_pred = model.predict(X_test)
        
        # Inverse scale
        y_true = preprocessor.inverse_scale(y_test.flatten())
        y_pred = preprocessor.inverse_scale(y_pred.flatten())
    
    elif model_type == 'prophet':
        from models.prophet_model import ProphetForecaster
        
        model_path = Path(__file__).parent.parent.parent / "models" / "prophet_model.pkl"
        if not model_path.exists():
            logger.error("Prophet model not found. Train first.")
            return {}
        
        model = ProphetForecaster()
        model.load(str(model_path))
        
        forecast = model.predict(horizon)
        y_pred = forecast['yhat'].tail(horizon).values
        y_true = resampled[target].tail(horizon).values
    
    else:
        logger.error(f"Unknown model: {model_type}")
        return {}
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    logger.info(f"Evaluation Metrics ({model_type}):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Plot
    plot_path = Path(__file__).parent.parent.parent / "models" / "plots" / f"{model_type}_forecast.png"
    plot_forecast(y_true, y_pred, f"{model_type.upper()} Forecast", str(plot_path))
    
    # Save metrics
    metrics_path = Path(__file__).parent.parent.parent / "models" / f"{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--horizon', type=int, default=24)
    args = parser.parse_args()
    
    evaluate_forecast(args.model, args.horizon)


if __name__ == "__main__":
    main()
