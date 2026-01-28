# Supply Chain Risk Classification Engine

A multi-class classification system to categorize shipments into risk levels (Low/Moderate/High) for proactive supply chain management.

## 🎯 Objective

Classify shipments into three risk categories:

- **Low Risk**: On-time delivery expected, minimal intervention needed
- **Moderate Risk**: Potential issues, monitoring required
- **High Risk**: Significant delay/disruption likely, immediate action needed

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost | Macro F1 | ~0.82 |
| LightGBM | Macro F1 | ~0.81 |
| Neural Network | Macro F1 | ~0.80 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.models.train

# Evaluate model
python -m src.models.evaluate

# Make predictions
python -m src.models.predict --input sample_data.csv
```

## 📁 Project Structure

```
project_2_risk_classification/
├── config/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       └── helpers.py
├── notebooks/
│   └── risk_analysis.ipynb
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## 🔧 Features Used

| Category | Features |
|----------|----------|
| **Risk Indicators** | disruption_likelihood_score, delay_probability, route_risk_level |
| **Driver Metrics** | driver_behavior_score, fatigue_monitoring_score |
| **Cargo** | iot_temperature, cargo_condition_status |
| **Logistics** | warehouse_inventory_level, shipping_costs, lead_time_days |
| **External** | weather_condition_severity, traffic_congestion_level, port_congestion_level |

## 📈 Risk Categories

| Risk Level | Delay Probability | Characteristics |
|------------|-------------------|-----------------|
| Low | < 30% | Reliable supplier, good weather, low traffic |
| Moderate | 30-70% | Some risk factors present |
| High | > 70% | Multiple risk factors, high disruption likelihood |

## 🔍 Model Interpretability

The system includes SHAP (SHapley Additive exPlanations) for feature importance and prediction explanations:

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.plot_shap_summary(X_test)
```

## 📝 License

MIT License
