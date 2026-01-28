"""
Prediction Module

Makes predictions using trained models.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    Make delay predictions using trained models.
    
    Provides:
    - Binary delay classification
    - Time deviation regression
    - Prediction confidence scores
    """
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.regressor = None
        self.feature_names = []
        
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
            self.feature_names = data.get('feature_names', [])
            logger.info(f"Loaded classifier from {classifier_path}")
        else:
            logger.warning(f"Classifier not found at {classifier_path}")
        
        if regressor_path.exists():
            data = joblib.load(regressor_path)
            self.regressor = data['model']
            logger.info(f"Loaded regressor from {regressor_path}")
        else:
            logger.warning(f"Regressor not found at {regressor_path}")
    
    def predict_delay(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict delay probability and classification.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with predictions
        """
        results = pd.DataFrame(index=X.index)
        
        if self.classifier is not None:
            # Ensure correct feature order
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in X.columns]
                X_aligned = X[available_features].copy()
            else:
                X_aligned = X.copy()
            
            X_aligned = X_aligned.fillna(X_aligned.median())
            
            results['is_delayed'] = self.classifier.predict(X_aligned)
            results['delay_probability'] = self.classifier.predict_proba(X_aligned)[:, 1]
            results['confidence'] = np.abs(results['delay_probability'] - 0.5) * 2
            
            logger.info(f"Made {len(results)} delay predictions")
        
        return results
    
    def predict_deviation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict delivery time deviation.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with deviation predictions
        """
        results = pd.DataFrame(index=X.index)
        
        if self.regressor is not None:
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in X.columns]
                X_aligned = X[available_features].copy()
            else:
                X_aligned = X.copy()
            
            X_aligned = X_aligned.fillna(X_aligned.median())
            
            results['predicted_deviation_hours'] = self.regressor.predict(X_aligned)
            
            logger.info(f"Made {len(results)} deviation predictions")
        
        return results
    
    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make all predictions (delay + deviation).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with all predictions
        """
        delay_results = self.predict_delay(X)
        deviation_results = self.predict_deviation(X)
        
        results = pd.concat([delay_results, deviation_results], axis=1)
        
        # Add risk level interpretation
        def get_risk_level(prob, deviation):
            if prob > 0.7 and deviation > 5:
                return 'High Risk'
            elif prob > 0.5 or deviation > 3:
                return 'Moderate Risk'
            else:
                return 'Low Risk'
        
        if 'delay_probability' in results.columns and 'predicted_deviation_hours' in results.columns:
            results['risk_level'] = results.apply(
                lambda row: get_risk_level(
                    row.get('delay_probability', 0),
                    row.get('predicted_deviation_hours', 0)
                ),
                axis=1
            )
        
        return results
    
    def predict_from_csv(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions from CSV file.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with input data and predictions
        """
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        predictions = self.predict_all(df)
        
        results = pd.concat([df, predictions], axis=1)
        
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        
        return results


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description='Make delay predictions')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--model-dir', type=str, default='models/', help='Model directory')
    
    args = parser.parse_args()
    
    predictor = DelayPredictor(args.model_dir)
    predictor.load_models()
    
    if args.input:
        results = predictor.predict_from_csv(args.input, args.output)
        print(f"\nPrediction Summary:")
        print(f"  Total records: {len(results)}")
        if 'is_delayed' in results.columns:
            print(f"  Predicted delays: {results['is_delayed'].sum()}")
        if 'risk_level' in results.columns:
            print(f"  Risk distribution: {results['risk_level'].value_counts().to_dict()}")
    else:
        print("No input file specified. Use --input <path> to make predictions.")


if __name__ == "__main__":
    main()
