"""
Model Training for Risk Classification

Multi-class classification with XGBoost/LightGBM.
"""

import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train multi-class risk classification models."""
    
    CLASS_LABELS = ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = []
        
    def _load_config(self, path: str) -> Dict:
        config_file = Path(path)
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            'model': {
                'name': 'xgboost',
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'objective': 'multi:softprob',
                    'num_class': 3
                }
            },
            'output': {
                'model_dir': 'models/',
                'model_file': 'risk_classifier.pkl'
            }
        }
    
    def get_model(self, name: str, params: Dict) -> Any:
        """Get model instance."""
        models = {
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'random_forest': RandomForestClassifier
        }
        return models[name](**params)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Any:
        """Train the classification model."""
        config = self.config['model']
        
        logger.info(f"Training {config['name']} classifier...")
        
        self.model = self.get_model(config['name'], config.get('params', {}))
        self.feature_names = list(X_train.columns)
        
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        if X_val is not None and y_val is not None:
            if config['name'] in ['xgboost', 'lightgbm']:
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False
        
        self.model.fit(X_train, y_train, **fit_params)
        
        logger.info(f"Training complete. Features: {len(self.feature_names)}")
        
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Run stratified cross-validation."""
        config = self.config['model']
        model = self.get_model(config['name'], config.get('params', {}))
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1_macro')
        
        results = {
            'mean_f1_macro': float(np.mean(scores)),
            'std_f1_macro': float(np.std(scores)),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV Macro F1: {results['mean_f1_macro']:.4f} (+/- {results['std_f1_macro']:.4f})")
        
        return results
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """Save trained model."""
        output_dir = output_dir or self.config['output']['model_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = Path(output_dir) / self.config['output']['model_file']
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'class_labels': self.CLASS_LABELS,
            'config': self.config['model'],
            'trained_at': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Saved model to {model_path}")
        
        return str(model_path)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


def main():
    """Main training script."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import DataPreprocessor
    from features.feature_engineering import FeatureEngineer
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    # Encode target
    y = loader.encode_target()
    
    # Feature engineering
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.extract_temporal_features(df)
    
    # Features
    feature_cols = [
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level',
        'warehouse_inventory_level', 'loading_unloading_time', 'handling_equipment_availability',
        'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
        'supplier_reliability_score', 'lead_time_days', 'iot_temperature',
        'route_risk_level', 'customs_clearance_time', 'driver_behavior_score',
        'fatigue_monitoring_score', 'disruption_likelihood_score', 'delay_probability',
        'hour', 'day_of_week', 'month', 'combined_risk_score', 'driver_safety_score'
    ]
    
    existing = [c for c in feature_cols if c in df.columns]
    X = df[existing].fillna(df[existing].median())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    trainer = ModelTrainer(str(config_path))
    
    trainer.train(X_train, y_train, X_test, y_test)
    cv_results = trainer.cross_validate(X_train, y_train)
    trainer.save_model()
    
    print("\nTop 10 Features:")
    print(trainer.get_feature_importance().head(10))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
