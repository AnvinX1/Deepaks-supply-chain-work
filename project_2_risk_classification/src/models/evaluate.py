"""
Model Evaluation for Risk Classification

Multi-class metrics and SHAP interpretability.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate multi-class risk classification models."""
    
    CLASS_LABELS = ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = []
    
    def load_model(self, model_file: str = "risk_classifier.pkl") -> None:
        """Load saved model."""
        model_path = self.model_dir / model_file
        if model_path.exists():
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_names = data.get('feature_names', [])
            logger.info(f"Loaded model from {model_path}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=self.CLASS_LABELS, output_dict=True)
        }
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:     {metrics['f1_macro']:.4f}")
        logger.info(f"  Weighted F1:  {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.CLASS_LABELS,
                    yticklabels=self.CLASS_LABELS)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Risk Classification Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def get_shap_values(self, X: pd.DataFrame) -> Any:
        """Calculate SHAP values for interpretability."""
        try:
            import shap
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            return shap_values
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            return None
    
    def plot_shap_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SHAP summary."""
        try:
            import shap
            
            shap_values = self.get_shap_values(X)
            if shap_values is None:
                return
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved SHAP summary to {save_path}")
            
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {e}")
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Save metrics to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Saved metrics to {output_path}")


def main():
    """Main evaluation script."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import DataPreprocessor
    from features.feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Load and prepare data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    y = loader.encode_target()
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.extract_temporal_features(df)
    
    feature_cols = [
        'fuel_consumption_rate', 'traffic_congestion_level', 'weather_condition_severity',
        'port_congestion_level', 'supplier_reliability_score', 'route_risk_level',
        'driver_behavior_score', 'disruption_likelihood_score', 'delay_probability',
        'hour', 'day_of_week'
    ]
    
    existing = [c for c in feature_cols if c in df.columns]
    X = df[existing].fillna(df[existing].median())
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Evaluate
    evaluator = ModelEvaluator(str(Path(__file__).parent.parent.parent / "models"))
    evaluator.load_model()
    
    if evaluator.model:
        metrics = evaluator.evaluate(X_test, y_test)
        evaluator.save_metrics(metrics, str(Path(__file__).parent.parent.parent / "models" / "metrics.json"))


if __name__ == "__main__":
    main()
