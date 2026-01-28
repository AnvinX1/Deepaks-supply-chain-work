"""
Model Evaluation Module

Evaluates trained models and generates performance metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate ML models for delay prediction.
    
    Provides metrics for:
    - Binary classification (delay prediction)
    - Regression (time deviation prediction)
    """
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.regressor = None
        
    def load_models(
        self, 
        classifier_file: str = "delay_classifier.pkl",
        regressor_file: str = "delay_regressor.pkl"
    ) -> None:
        """
        Load saved models from disk.
        
        Args:
            classifier_file: Classifier filename
            regressor_file: Regressor filename
        """
        classifier_path = self.model_dir / classifier_file
        regressor_path = self.model_dir / regressor_file
        
        if classifier_path.exists():
            data = joblib.load(classifier_path)
            self.classifier = data['model']
            logger.info(f"Loaded classifier from {classifier_path}")
        
        if regressor_path.exists():
            data = joblib.load(regressor_path)
            self.regressor = data['model']
            logger.info(f"Loaded regressor from {regressor_path}")
    
    def evaluate_classifier(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification model.
        
        Args:
            X_test: Test features
            y_test: True labels
            model: Model to evaluate (uses loaded model if None)
            
        Returns:
            Dictionary of metrics
        """
        model = model or self.classifier
        
        if model is None:
            raise ValueError("No classifier available. Load or provide a model.")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Classification Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_regressor(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate regression model.
        
        Args:
            X_test: Test features
            y_test: True values
            model: Model to evaluate (uses loaded model if None)
            
        Returns:
            Dictionary of metrics
        """
        model = model or self.regressor
        
        if model is None:
            raise ValueError("No regressor available. Load or provide a model.")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (handling zeros)
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        logger.info(f"Regression Metrics:")
        logger.info(f"  MAE:      {metrics['mae']:.4f}")
        logger.info(f"  RMSE:     {metrics['rmse']:.4f}")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  MAPE:     {metrics['mape']:.2f}%")
        
        return metrics
    
    def generate_report(
        self,
        classification_metrics: Dict[str, Any],
        regression_metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            classification_metrics: Classification metrics dict
            regression_metrics: Regression metrics dict
            output_path: Path to save JSON report
            
        Returns:
            Combined report dictionary
        """
        report = {
            'classification': classification_metrics,
            'regression': regression_metrics,
            'summary': {
                'best_classification_metric': 'roc_auc',
                'classification_score': classification_metrics.get('roc_auc', 0),
                'best_regression_metric': 'rmse',
                'regression_score': regression_metrics.get('rmse', 0)
            }
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved evaluation report to {output_path}")
        
        return report


def main():
    """Main evaluation script."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import DataPreprocessor
    from features.feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
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
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night'
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_class = df['is_delayed']
    y_reg = df['delivery_time_deviation']
    
    # Split
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # Load and evaluate
    evaluator = ModelEvaluator(str(Path(__file__).parent.parent.parent / "models"))
    evaluator.load_models()
    
    if evaluator.classifier:
        class_metrics = evaluator.evaluate_classifier(X_test, y_class_test)
    
    if evaluator.regressor:
        reg_metrics = evaluator.evaluate_regressor(X_test, y_reg_test)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
