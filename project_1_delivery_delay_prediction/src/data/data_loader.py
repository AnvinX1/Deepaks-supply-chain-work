"""
Data Loader Module

Handles loading and initial validation of the supply chain dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate supply chain logistics data.
    
    Attributes:
        data_path: Path to the CSV dataset
        df: Loaded DataFrame
    """
    
    REQUIRED_COLUMNS = [
        'timestamp', 'vehicle_gps_latitude', 'vehicle_gps_longitude',
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level',
        'warehouse_inventory_level', 'loading_unloading_time',
        'handling_equipment_availability', 'weather_condition_severity',
        'port_congestion_level', 'shipping_costs', 'supplier_reliability_score',
        'lead_time_days', 'iot_temperature', 'route_risk_level',
        'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
        'disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation'
    ]
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        self._validate_columns()
        
        return self.df
    
    def _validate_columns(self) -> None:
        """Validate that all required columns are present."""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
    
    def get_info(self) -> dict:
        """
        Get dataset information.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        return train_df, test_df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("../data/dynamic_supply_chain_logistics_dataset.csv")
    df = loader.load()
    print(loader.get_info())
