# Delivery Delay Prediction System

A production-ready machine learning system for predicting delivery delays and time deviations in supply chain logistics.

## 🎯 Objective

Predict two key metrics:

1. **Delay Probability**: Binary classification (delayed vs on-time)
2. **Delivery Time Deviation**: Regression (hours of deviation from ETA)

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost Classifier | ROC-AUC | ~0.85 |
| XGBoost Regressor | RMSE | ~2.5 hours |
| LightGBM Classifier | ROC-AUC | ~0.84 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.models.train

# Evaluate model performance
python -m src.models.evaluate

# Make predictions
python -m src.models.predict --input sample_data.csv
```

## 📁 Project Structure

```
project_1_delivery_delay_prediction/
├── config/
│   └── config.yaml          # Model hyperparameters & settings
├── src/
│   ├── data/
│   │   ├── data_loader.py   # Load & validate data
│   │   └── preprocessing.py # Data cleaning & transformation
│   ├── features/
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   ├── train.py         # Training pipeline
│   │   ├── evaluate.py      # Model evaluation
│   │   └── predict.py       # Inference module
│   └── utils/
│       └── helpers.py       # Utility functions
├── notebooks/
│   └── exploratory_analysis.ipynb
├── models/                   # Saved model artifacts
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## 🔧 Features Used

| Category | Features |
|----------|----------|
| **Spatial** | GPS coordinates, route_risk_level |
| **Temporal** | Hour, day of week, month |
| **Operational** | eta_variation_hours, loading_unloading_time, customs_clearance_time |
| **External** | traffic_congestion_level, weather_condition_severity, port_congestion_level |
| **Supplier** | supplier_reliability_score, handling_equipment_availability |

## 📈 Training

```bash
# Full training with default config
python -m src.models.train

# Custom config
python -m src.models.train --config config/custom_config.yaml

# With specific model
python -m src.models.train --model xgboost
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📄 Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Feature selection
- Train/test split ratio
- Cross-validation folds

## 📝 License

MIT License
