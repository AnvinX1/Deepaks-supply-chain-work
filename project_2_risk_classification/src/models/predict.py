"""
Prediction Module for Risk Classification
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiskPredictor:
    """Make risk predictions."""
    
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
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict risk levels."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Align features
        if self.feature_names:
            available = [f for f in self.feature_names if f in X.columns]
            X = X[available].copy()
        
        X = X.fillna(X.median())
        
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        results = pd.DataFrame({
            'risk_level_code': y_pred,
            'risk_level': [self.CLASS_LABELS[i] for i in y_pred],
            'prob_low_risk': y_prob[:, 0],
            'prob_moderate_risk': y_prob[:, 1],
            'prob_high_risk': y_prob[:, 2],
            'confidence': np.max(y_prob, axis=1)
        }, index=X.index)
        
        return results
    
    def predict_from_csv(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Predict from CSV file."""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        predictions = self.predict(df)
        results = pd.concat([df, predictions], axis=1)
        
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Risk classification predictions')
    parser.add_argument('--input', type=str, help='Input CSV')
    parser.add_argument('--output', type=str, help='Output CSV')
    parser.add_argument('--model-dir', type=str, default='models/')
    
    args = parser.parse_args()
    
    predictor = RiskPredictor(args.model_dir)
    predictor.load_model()
    
    if args.input:
        results = predictor.predict_from_csv(args.input, args.output)
        print(f"\nRisk Distribution:")
        print(results['risk_level'].value_counts())


if __name__ == "__main__":
    main()
