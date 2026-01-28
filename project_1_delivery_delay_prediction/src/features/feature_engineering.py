"""
Feature Engineering Module

Creates advanced features for delivery delay prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for supply chain delay prediction.
    
    Creates:
    - Interaction features
    - Aggregated statistics
    - Risk composite scores
    - Geospatial features
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names: List[str] = []
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between related variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Traffic x Weather interaction
        if 'traffic_congestion_level' in df.columns and 'weather_condition_severity' in df.columns:
            df['traffic_weather_interaction'] = (
                df['traffic_congestion_level'] * df['weather_condition_severity']
            )
        
        # Port congestion x Customs time
        if 'port_congestion_level' in df.columns and 'customs_clearance_time' in df.columns:
            df['port_customs_interaction'] = (
                df['port_congestion_level'] * df['customs_clearance_time']
            )
        
        # Driver fatigue x Route risk
        if 'fatigue_monitoring_score' in df.columns and 'route_risk_level' in df.columns:
            df['fatigue_risk_interaction'] = (
                df['fatigue_monitoring_score'] * df['route_risk_level']
            )
        
        # Supplier reliability x Lead time
        if 'supplier_reliability_score' in df.columns and 'lead_time_days' in df.columns:
            df['reliability_leadtime_ratio'] = (
                df['supplier_reliability_score'] / (df['lead_time_days'] + 1)
            )
        
        logger.info("Created interaction features")
        
        return df
    
    def create_risk_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk scores from multiple risk indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with composite risk scores
        """
        df = df.copy()
        
        risk_cols = [
            'route_risk_level',
            'disruption_likelihood_score',
            'weather_condition_severity',
            'traffic_congestion_level'
        ]
        
        existing_risk_cols = [col for col in risk_cols if col in df.columns]
        
        if existing_risk_cols:
            # Normalize each risk factor to 0-1 scale and average
            for col in existing_risk_cols:
                max_val = df[col].max()
                if max_val > 0:
                    df[f'{col}_normalized'] = df[col] / max_val
            
            normalized_cols = [f'{col}_normalized' for col in existing_risk_cols]
            df['composite_risk_score'] = df[normalized_cols].mean(axis=1)
            
            # Drop normalized columns
            df = df.drop(columns=normalized_cols)
            
            logger.info(f"Created composite risk score from {len(existing_risk_cols)} factors")
        
        return df
    
    def create_operational_efficiency_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create operational efficiency metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with efficiency metrics
        """
        df = df.copy()
        
        # Loading efficiency
        if 'loading_unloading_time' in df.columns and 'handling_equipment_availability' in df.columns:
            df['loading_efficiency'] = (
                df['handling_equipment_availability'] / (df['loading_unloading_time'] + 0.1)
            )
        
        # Cost efficiency
        if 'shipping_costs' in df.columns and 'lead_time_days' in df.columns:
            df['cost_per_day'] = df['shipping_costs'] / (df['lead_time_days'] + 1)
        
        # Fuel efficiency anomaly
        if 'fuel_consumption_rate' in df.columns:
            mean_fuel = df['fuel_consumption_rate'].mean()
            std_fuel = df['fuel_consumption_rate'].std()
            df['fuel_anomaly_score'] = np.abs(df['fuel_consumption_rate'] - mean_fuel) / std_fuel
        
        logger.info("Created operational efficiency features")
        
        return df
    
    def create_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from GPS coordinates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with geospatial features
        """
        df = df.copy()
        
        if 'vehicle_gps_latitude' in df.columns and 'vehicle_gps_longitude' in df.columns:
            # Distance from center point (approximate US center)
            center_lat, center_lon = 39.8283, -98.5795
            
            df['lat_deviation'] = np.abs(df['vehicle_gps_latitude'] - center_lat)
            df['lon_deviation'] = np.abs(df['vehicle_gps_longitude'] - center_lon)
            
            # Approximate distance using Euclidean (simplified)
            df['distance_from_center'] = np.sqrt(
                df['lat_deviation']**2 + df['lon_deviation']**2
            )
            
            # Region classification (simplified US regions)
            df['region'] = pd.cut(
                df['vehicle_gps_longitude'],
                bins=[-130, -100, -85, -70],
                labels=['west', 'central', 'east']
            ).astype(str)
            
            # One-hot encode region
            region_dummies = pd.get_dummies(df['region'], prefix='region')
            df = pd.concat([df, region_dummies], axis=1)
            df = df.drop(columns=['region'])
            
            logger.info("Created geospatial features")
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        df = self.create_interaction_features(df)
        df = self.create_risk_composite_score(df)
        df = self.create_operational_efficiency_score(df)
        df = self.create_geospatial_features(df)
        
        # Store feature names
        self.feature_names = list(df.columns)
        
        logger.info(f"Total features after engineering: {len(self.feature_names)}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.data_loader import DataLoader
    
    loader = DataLoader("../../data/dynamic_supply_chain_logistics_dataset.csv")
    df = loader.load()
    
    engineer = FeatureEngineer()
    df_engineered = engineer.create_all_features(df)
    
    print(f"Original columns: {len(loader.df.columns)}")
    print(f"After engineering: {len(df_engineered.columns)}")
    print(f"New features: {set(df_engineered.columns) - set(loader.df.columns)}")
