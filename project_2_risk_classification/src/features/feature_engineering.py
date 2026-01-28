"""
Feature Engineering for Risk Classification

Creates risk-specific features and composite scores.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for risk classification."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_risk_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk indicators."""
        df = df.copy()
        
        # Combined disruption risk
        risk_cols = ['disruption_likelihood_score', 'delay_probability', 'route_risk_level']
        existing = [c for c in risk_cols if c in df.columns]
        if existing:
            df['combined_risk_score'] = df[existing].mean(axis=1)
        
        # Driver safety composite
        driver_cols = ['driver_behavior_score', 'fatigue_monitoring_score']
        existing = [c for c in driver_cols if c in df.columns]
        if existing:
            df['driver_safety_score'] = df[existing].mean(axis=1)
        
        # External factors composite
        external = ['weather_condition_severity', 'traffic_congestion_level', 'port_congestion_level']
        existing = [c for c in external if c in df.columns]
        if existing:
            df['external_risk_score'] = df[existing].mean(axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()
        
        if 'supplier_reliability_score' in df.columns and 'lead_time_days' in df.columns:
            df['reliability_time_ratio'] = df['supplier_reliability_score'] / (df['lead_time_days'] + 1)
        
        if 'shipping_costs' in df.columns and 'warehouse_inventory_level' in df.columns:
            df['cost_inventory_ratio'] = df['shipping_costs'] / (df['warehouse_inventory_level'] + 1)
        
        return df
    
    def create_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary threshold features."""
        df = df.copy()
        
        if 'iot_temperature' in df.columns:
            # Temperature anomaly (outside normal range)
            df['temp_anomaly'] = ((df['iot_temperature'] < -5) | (df['iot_temperature'] > 35)).astype(int)
        
        if 'delay_probability' in df.columns:
            df['high_delay_risk'] = (df['delay_probability'] > 0.7).astype(int)
            df['medium_delay_risk'] = ((df['delay_probability'] > 0.3) & (df['delay_probability'] <= 0.7)).astype(int)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering."""
        df = self.create_risk_composite_features(df)
        df = self.create_interaction_features(df)
        df = self.create_threshold_features(df)
        
        self.feature_names = list(df.columns)
        logger.info(f"Created features. Total: {len(self.feature_names)}")
        
        return df
