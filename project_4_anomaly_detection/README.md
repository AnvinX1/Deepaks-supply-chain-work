# Anomaly Detection & Monitoring System

An unsupervised learning system for detecting anomalies in supply chain operations with real-time alerting capabilities.

## 🎯 Objective

Detect unusual patterns in:

1. **Sensor Anomalies** - Abnormal temperature readings, fuel consumption
2. **Operational Anomalies** - Unusual congestion, loading times
3. **Behavioral Anomalies** - Driver behavior outliers

## 📊 Model Performance

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Isolation Forest | ~0.85 | ~0.78 | ~0.81 |
| Autoencoder | ~0.82 | ~0.80 | ~0.81 |
| One-Class SVM | ~0.80 | ~0.75 | ~0.77 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train anomaly detector
python -m src.detectors.isolation_forest

# Run anomaly detection
python -m src.monitoring.alert_system --input data.csv

# Train autoencoder
python -m src.detectors.autoencoder
```

## 📁 Project Structure

```
project_4_anomaly_detection/
├── config/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── detectors/
│   │   ├── isolation_forest.py
│   │   ├── autoencoder.py
│   │   └── statistical_methods.py
│   ├── monitoring/
│   │   └── alert_system.py
│   └── visualization/
│       └── anomaly_plots.py
├── notebooks/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## 🔧 Features Monitored

| Category | Features |
|----------|----------|
| **Sensor** | iot_temperature, fuel_consumption_rate |
| **Operational** | traffic_congestion_level, loading_unloading_time |
| **Behavioral** | driver_behavior_score, fatigue_monitoring_score |

## 📈 Detection Methods

### Isolation Forest

- Ensemble tree-based anomaly detection
- Fast training and inference
- Works well with high-dimensional data

### Autoencoder

- Deep learning reconstruction-based detection
- Learns normal patterns, flags deviations
- Good for complex patterns

### Statistical Methods

- Z-score based detection
- IQR (Interquartile Range) method
- Simple and interpretable

## 🔔 Alerting

The system generates alerts with severity levels:

- **CRITICAL**: Immediate action required
- **WARNING**: Investigation needed
- **INFO**: For monitoring purposes

## 📝 License

MIT License
