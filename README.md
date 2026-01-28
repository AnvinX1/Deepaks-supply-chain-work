# Supply Chain ML Projects

A comprehensive suite of 4 industry-standard machine learning projects for supply chain logistics optimization.

## 🎯 Projects Overview

| Project | Description | ML Type |
|---------|-------------|---------|
| [Delivery Delay Prediction](./project_1_delivery_delay_prediction/) | Predict delivery delays and time deviations | Regression + Classification |
| [Risk Classification](./project_2_risk_classification/) | Classify shipments into risk levels | Multi-class Classification |
| [Demand Forecasting](./project_3_demand_forecasting/) | Forecast inventory and demand patterns | Time Series |
| [Anomaly Detection](./project_4_anomaly_detection/) | Detect operational anomalies | Unsupervised Learning |

## 📊 Dataset

The dataset (`data/dynamic_supply_chain_logistics_dataset.csv`) contains **32,067 records** with 26 features covering:

- Vehicle GPS & Fuel data
- Traffic & Warehouse logistics
- Weather & Port conditions
- Supplier reliability metrics
- IoT sensor readings
- Risk classifications

## 🚀 Quick Start

```bash
# Clone and navigate to a project
cd project_1_delivery_delay_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python -m src.models.train

# Run predictions
python -m src.models.predict
```

## 📁 Project Structure

```
supply_chain_ml_projects/
├── data/                               # Shared dataset
├── project_1_delivery_delay_prediction/
├── project_2_risk_classification/
├── project_3_demand_forecasting/
└── project_4_anomaly_detection/
```

## 🛠️ Tech Stack

- **Core**: Python 3.10+, NumPy, Pandas, Scikit-learn
- **ML**: XGBoost, LightGBM, TensorFlow/Keras
- **Time Series**: Prophet, statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📄 License

MIT License
