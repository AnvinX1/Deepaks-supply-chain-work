"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'fuel_consumption_rate': np.random.uniform(5, 20, 100),
            'traffic_congestion_level': np.random.uniform(0, 10, 100),
            'weather_condition_severity': np.random.uniform(0, 1, 100),
            'delay_probability': np.random.uniform(0, 1, 100),
            'delivery_time_deviation': np.random.uniform(-5, 10, 100)
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor()
    
    def test_extract_temporal_features(self, preprocessor, sample_data):
        """Test temporal feature extraction."""
        result = preprocessor.extract_temporal_features(sample_data)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_night' in result.columns
        
        # Check value ranges
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
    
    def test_create_delay_label(self, preprocessor, sample_data):
        """Test delay label creation."""
        result = preprocessor.create_delay_label(sample_data, threshold=0.5)
        
        assert 'is_delayed' in result.columns
        assert result['is_delayed'].isin([0, 1]).all()
        
        # Check threshold is applied correctly
        high_prob = sample_data[sample_data['delay_probability'] >= 0.5].index
        assert (result.loc[high_prob, 'is_delayed'] == 1).all()
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        # Add some missing values
        sample_data.loc[0:10, 'fuel_consumption_rate'] = np.nan
        
        numerical_cols = ['fuel_consumption_rate', 'traffic_congestion_level']
        result = preprocessor.handle_missing_values(sample_data, numerical_cols)
        
        assert result['fuel_consumption_rate'].isnull().sum() == 0
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling."""
        columns = ['fuel_consumption_rate', 'traffic_congestion_level']
        result = preprocessor.scale_features(sample_data, columns)
        
        # After scaling, mean should be ~0 and std ~1
        for col in columns:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
