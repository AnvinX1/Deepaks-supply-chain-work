"""
Prophet Forecaster

Facebook Prophet for time series forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Prophet-based time series forecaster."""
    
    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
    
    def build_model(self) -> Any:
        """Initialize Prophet model."""
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Run: pip install prophet")
            raise
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        logger.info("Built Prophet model")
        
        return self.model
    
    def train(self, df: pd.DataFrame) -> Any:
        """
        Train Prophet model.
        
        Args:
            df: DataFrame with columns 'ds' (datetime) and 'y' (target)
        """
        if self.model is None:
            self.build_model()
        
        # Suppress Prophet output
        import logging as log
        log.getLogger('prophet').setLevel(log.WARNING)
        log.getLogger('cmdstanpy').setLevel(log.WARNING)
        
        self.model.fit(df)
        logger.info(f"Trained Prophet on {len(df)} records")
        
        return self.model
    
    def predict(self, periods: int, freq: str = 'H') -> pd.DataFrame:
        """Make future predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_components(self) -> Dict[str, Any]:
        """Get trend and seasonality components."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return {
            'trend': self.model.params.get('trend'),
            'seasonality': self.model.seasonalities
        }
    
    def plot_components(self, forecast: pd.DataFrame, save_path: Optional[str] = None):
        """Plot forecast components."""
        import matplotlib.pyplot as plt
        
        fig = self.model.plot_components(forecast)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved components plot to {save_path}")
        
        plt.close()
    
    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved Prophet model to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        self.model = joblib.load(path)
        logger.info(f"Loaded Prophet model from {path}")
