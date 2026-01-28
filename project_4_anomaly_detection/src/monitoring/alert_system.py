"""
Alert System

Generate alerts based on detected anomalies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertSystem:
    """Generate and manage anomaly alerts."""
    
    def __init__(
        self,
        critical_threshold: float = 0.9,
        warning_threshold: float = 0.7,
        info_threshold: float = 0.5,
        log_file: Optional[str] = None
    ):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.info_threshold = info_threshold
        self.log_file = log_file
        self.alerts: List[Dict[str, Any]] = []
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def get_severity(self, score: float) -> AlertSeverity:
        """Determine alert severity based on anomaly score."""
        if score >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif score >= self.warning_threshold:
            return AlertSeverity.WARNING
        elif score >= self.info_threshold:
            return AlertSeverity.INFO
        return None
    
    def create_alert(
        self,
        record_id: Any,
        anomaly_score: float,
        features: Dict[str, float],
        detector_type: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """Create an alert from an anomaly detection."""
        severity = self.get_severity(anomaly_score)
        
        if severity is None:
            return None
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'record_id': str(record_id),
            'severity': severity.value,
            'anomaly_score': float(anomaly_score),
            'detector_type': detector_type,
            'affected_features': features,
            'message': self._generate_message(severity, anomaly_score, features)
        }
        
        self.alerts.append(alert)
        self._log_alert(alert)
        
        return alert
    
    def _generate_message(
        self, 
        severity: AlertSeverity, 
        score: float, 
        features: Dict[str, float]
    ) -> str:
        """Generate human-readable alert message."""
        top_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        feature_str = ", ".join([f"{k}={v:.2f}" for k, v in top_features])
        
        if severity == AlertSeverity.CRITICAL:
            return f"CRITICAL anomaly detected (score: {score:.2f}). Key factors: {feature_str}"
        elif severity == AlertSeverity.WARNING:
            return f"Warning: Unusual pattern detected (score: {score:.2f}). Factors: {feature_str}"
        else:
            return f"Info: Minor anomaly (score: {score:.2f}). Factors: {feature_str}"
    
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """Log alert to file."""
        log_msg = f"[{alert['severity']}] {alert['message']}"
        
        if alert['severity'] == "CRITICAL":
            logger.critical(log_msg)
        elif alert['severity'] == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
    
    def process_detection_results(
        self,
        results: pd.DataFrame,
        original_data: pd.DataFrame,
        detector_type: str = "isolation_forest"
    ) -> List[Dict[str, Any]]:
        """Process detection results and generate alerts."""
        alerts = []
        
        anomalies = results[results['is_anomaly'] == 1]
        
        for idx in anomalies.index:
            score = results.loc[idx, 'anomaly_score']
            features = original_data.loc[idx].to_dict() if idx in original_data.index else {}
            
            alert = self.create_alert(idx, score, features, detector_type)
            if alert:
                alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts from {len(anomalies)} anomalies")
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by severity."""
        summary = {
            'CRITICAL': 0,
            'WARNING': 0,
            'INFO': 0,
            'total': len(self.alerts)
        }
        
        for alert in self.alerts:
            severity = alert.get('severity', 'INFO')
            if severity in summary:
                summary[severity] += 1
        
        return summary
    
    def export_alerts(self, path: str) -> None:
        """Export alerts to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        
        logger.info(f"Exported {len(self.alerts)} alerts to {path}")


def main():
    """Test alert system."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.data_loader import DataLoader
    from data.preprocessing import AnomalyPreprocessor
    from detectors.isolation_forest import IsolationForestDetector
    
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dynamic_supply_chain_logistics_dataset.csv"
    loader = DataLoader(str(data_path))
    df = loader.load()
    
    features = ['iot_temperature', 'fuel_consumption_rate', 'traffic_congestion_level']
    X = df[[f for f in features if f in df.columns]].copy()
    
    preprocessor = AnomalyPreprocessor()
    X_processed = preprocessor.full_pipeline(X, remove_outliers=False)
    
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(X_processed)
    results = detector.detect_anomalies(X_processed)
    
    alert_system = AlertSystem(
        log_file=str(Path(__file__).parent.parent.parent / "logs" / "alerts.log")
    )
    
    alerts = alert_system.process_detection_results(results, X, "isolation_forest")
    
    print("\nAlert Summary:")
    print(alert_system.get_alert_summary())


if __name__ == "__main__":
    main()
