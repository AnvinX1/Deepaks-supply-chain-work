# Demand Forecasting with Time Series Analysis

A time series forecasting system for predicting warehouse inventory levels and demand patterns in supply chain logistics.

## 🎯 Objective

Forecast future values of:

1. **Warehouse Inventory Levels** - Predict stock levels for inventory planning
2. **Historical Demand** - Forecast demand patterns for resource allocation

## 📊 Model Performance

| Model | Target | MAE | RMSE |
|-------|--------|-----|------|
| LSTM | Inventory | ~50 units | ~75 units |
| Prophet | Inventory | ~55 units | ~80 units |
| LightGBM | Demand | ~500 units | ~750 units |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train LSTM model
python -m src.models.train --model lstm

# Train Prophet model
python -m src.models.train --model prophet

# Generate forecasts
python -m src.models.evaluate --horizon 24
```

## 📁 Project Structure

```
project_3_demand_forecasting/
├── config/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── time_series_preprocessing.py
│   ├── features/
│   │   └── temporal_features.py
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── prophet_model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/
│       └── plots.py
├── notebooks/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## 🔧 Features

| Type | Features |
|------|----------|
| **Temporal** | Hour, day of week, month, is_weekend |
| **Lag Features** | t-1, t-6, t-12, t-24 |
| **Rolling Stats** | Mean, std, min, max (6h, 12h, 24h windows) |
| **External** | Weather, traffic, shipping costs |

## 📈 Models

### LSTM (Long Short-Term Memory)

- Sequence-to-sequence architecture
- 24-hour lookback window
- Handles complex temporal patterns

### Prophet (Meta/Facebook)

- Automatic trend and seasonality detection
- Handles missing data and outliers well
- Great for daily/weekly patterns

### LightGBM with Time Features

- Feature-based approach
- Fast training and inference
- Good baseline model

## 🔍 Forecast Horizons

| Horizon | Use Case |
|---------|----------|
| 24 hours | Operational planning |
| 48 hours | Tactical decisions |
| 7 days | Strategic planning |

## 📝 License

MIT License
