"""Tests for demand forecasting."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.time_series_preprocessing import TimeSeriesPreprocessor


class TestTimeSeriesPreprocessor:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        return pd.DataFrame({'value': np.random.uniform(100, 500, 100)}, index=dates)
    
    @pytest.fixture
    def preprocessor(self):
        return TimeSeriesPreprocessor()
    
    def test_create_lag_features(self, preprocessor, sample_data):
        result = preprocessor.create_lag_features(sample_data, 'value', [1, 6])
        assert 'value_lag_1' in result.columns
        assert 'value_lag_6' in result.columns
    
    def test_create_rolling_features(self, preprocessor, sample_data):
        result = preprocessor.create_rolling_features(sample_data, 'value', [6])
        assert 'value_rolling_mean_6' in result.columns
    
    def test_create_sequences(self, preprocessor):
        data = np.arange(100)
        X, y = preprocessor.create_sequences(data, lookback=10, forecast_horizon=1)
        assert X.shape[0] == 90
        assert X.shape[1] == 10
        assert y.shape[0] == 90


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
