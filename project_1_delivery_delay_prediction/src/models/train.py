"""
Model Training Module

Handles training of classification and regression models for delay prediction.
"""

import os
import sys
import yaml
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train ML models for delivery delay prediction.
    
    Supports:
    - XGBoost (default)
    - LightGBM
    - Random Forest
    """
    
    SUPPORTED_MODELS = {
        'xgboost': (XGBClassifier, XGBRegressor),
        'lightgbm': (LGBMClassifier, LGBMRegressor),
        'random_forest': (RandomForestClassifier, RandomForestRegressor)
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.classifier = None
        self.regressor = None
        self.feature_names = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._default_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'data': {
                'path': '../data/dynamic_supply_chain_logistics_dataset.csv',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'classification': {
                    'name': 'xgboost',
                    'params': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
                },
                'regression': {
                    'name': 'xgboost',
                    'params': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
                }
            },
            'output': {
                'model_dir': 'models/',
                'classification_model': 'delay_classifier.pkl',
                'regression_model': 'delay_regressor.pkl'
            }
        }
    
    def _get_model(self, model_name: str, task: str, params: Dict[str, Any]):
        """
        Get model instance based on name and task.
        
        Args:
            model_name: Name of the model ('xgboost', 'lightgbm', 'random_forest')
            task: 'classification' or 'regression'
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        classifier_cls, regressor_cls = self.SUPPORTED_MODELS[model_name]
        
        if task == 'classification':
            return classifier_cls(**params)
        else:
            return regressor_cls(**params)
    
    def train_classifier(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Any:
        """
        Train classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained classifier
        """
        model_config = self.config['model']['classification']
        model_name = model_config['name']
        params = model_config.get('params', {})
        
        logger.info(f"Training {model_name} classifier...")
        
        self.classifier = self._get_model(model_name, 'classification', params)
        
        if X_val is not None and y_val is not None and model_name in ['xgboost', 'lightgbm']:
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.classifier.fit(X_train, y_train)
        
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Classifier training complete. Features: {len(self.feature_names)}")
        
        return self.classifier
    
    def train_regressor(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Any:
        """
        Train regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained regressor
        """
        model_config = self.config['model']['regression']
        model_name = model_config['name']
        params = model_config.get('params', {})
        
        logger.info(f"Training {model_name} regressor...")
        
        self.regressor = self._get_model(model_name, 'regression', params)
        
        if X_val is not None and y_val is not None and model_name in ['xgboost', 'lightgbm']:
            self.regressor.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.regressor.fit(X_train, y_train)
        
        logger.info("Regressor training complete.")
        
        return self.regressor
    
    def cross_validate(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation with {scoring}...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def save_models(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Output directory (uses config if not specified)
            
        Returns:
            Tuple of (classifier_path, regressor_path)
        """
        output_dir = output_dir or self.config['output']['model_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        classifier_path = None
        regressor_path = None
        
        if self.classifier is not None:
            classifier_path = os.path.join(output_dir, self.config['output']['classification_model'])
            joblib.dump({
                'model': self.classifier,
                'feature_names': self.feature_names,
                'config': self.config['model']['classification'],
                'trained_at': datetime.now().isoformat()
            }, classifier_path)
            logger.info(f"Saved classifier to {classifier_path}")
        
        if self.regressor is not None:
            regressor_path = os.path.join(output_dir, self.config['output']['regression_model'])
            joblib.dump({
                'model': self.regressor,
                'feature_names': self.feature_names,
                'config': self.config['model']['regression'],
                'trained_at': datetime.now().isoformat()
            }, regressor_path)
            logger.info(f"Saved regressor to {regressor_path}")
        
        return classifier_path, regressor_path
    
    def get_feature_importance(self, model_type: str = 'classifier') -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_type: 'classifier' or 'regressor'
            
        Returns:
            DataFrame with feature importances
        """
        model = self.classifier if model_type == 'classifier' else self.regressor
        
        if model is None:
            raise ValueError(f"No {model_type} trained yet.")
        
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """Main training script."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import DataPreprocessor
    from features.feature_engineering import FeatureEngineer
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    # Initialize components
    trainer = ModelTrainer(str(config_path))
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.extract_temporal_features(df)
    df = preprocessor.create_delay_label(df, threshold=0.5)
    
    # Feature engineering
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    # Prepare features
    feature_cols = [
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level',
        'warehouse_inventory_level', 'loading_unloading_time', 'handling_equipment_availability',
        'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
        'supplier_reliability_score', 'lead_time_days', 'iot_temperature',
        'route_risk_level', 'customs_clearance_time', 'driver_behavior_score',
        'fatigue_monitoring_score', 'disruption_likelihood_score',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
        'composite_risk_score', 'loading_efficiency', 'fuel_anomaly_score'
    ]
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y_class = df['is_delayed']
    y_reg = df['delivery_time_deviation']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train/test split
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    # Train models
    trainer.train_classifier(X_train, y_class_train, X_test, y_class_test)
    trainer.train_regressor(X_train, y_reg_train, X_test, y_reg_test)
    
    # Cross-validation
    cv_results = trainer.cross_validate(trainer.classifier, X_train, y_class_train, cv=5, scoring='roc_auc')
    
    # Save models
    trainer.save_models()
    
    # Feature importance
    importance = trainer.get_feature_importance()
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
