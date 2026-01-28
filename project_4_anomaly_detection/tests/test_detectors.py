"""Tests for anomaly detection."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detectors.isolation_forest import IsolationForestDetector
from detectors.statistical_methods import StatisticalDetector


class TestIsolationForest:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        normal = np.random.normal(0, 1, (100, 3))
        outliers = np.random.normal(5, 1, (5, 3))
        data = np.vstack([normal, outliers])
        return pd.DataFrame(data, columns=['f1', 'f2', 'f3'])
    
    def test_fit_predict(self, sample_data):
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(sample_data)
        
        results = detector.detect_anomalies(sample_data)
        
        assert 'is_anomaly' in results.columns
        assert 'anomaly_score' in results.columns
        assert results['is_anomaly'].isin([0, 1]).all()


class TestStatisticalDetector:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'f1': np.random.normal(0, 1, 100),
            'f2': np.random.normal(5, 2, 100)
        })
    
    def test_zscore_detection(self, sample_data):
        detector = StatisticalDetector(z_score_threshold=2.0)
        detector.fit(sample_data)
        
        results = detector.detect_zscore(sample_data)
        assert 'is_anomaly' in results.columns
        assert 'max_z_score' in results.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
