"""Tests for risk classification models."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'feature1': np.random.uniform(0, 10, 100),
            'feature2': np.random.uniform(0, 1, 100)
        })
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    def test_extract_temporal_features(self, preprocessor, sample_data):
        result = preprocessor.extract_temporal_features(sample_data)
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
    
    def test_scale_features(self, preprocessor, sample_data):
        result = preprocessor.scale_features(sample_data, ['feature1', 'feature2'])
        assert abs(result['feature1'].mean()) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
