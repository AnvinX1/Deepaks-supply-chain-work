# Supply Chain ML Projects

A comprehensive suite of 4 industry-standard machine learning projects for supply chain logistics optimization, with a **professional web dashboard** for real-time predictions.

## 🎯 Projects Overview

| Project | Description | ML Type |
|---------|-------------|---------|
| [Delivery Delay Prediction](./project_1_delivery_delay_prediction/) | Predict delivery delays and time deviations | Regression + Classification |
| [Risk Classification](./project_2_risk_classification/) | Classify shipments into risk levels | Multi-class Classification |
| [Demand Forecasting](./project_3_demand_forecasting/) | Forecast inventory and demand patterns | Time Series |
| [Anomaly Detection](./project_4_anomaly_detection/) | Detect operational anomalies | Unsupervised Learning |

## 🖥️ Dashboard Applications

### Option 1: Pure HTML/CSS/JS Frontend (Recommended)

A modern, premium dark-themed dashboard with Chart.js visualizations.

```bash
# Terminal 1: Start FastAPI Backend
cd supply_chain_ml_projects
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Frontend
cd supply_chain_ml_projects/frontend
python3 -m http.server 3000

# Open in browser
http://localhost:3000
```

### Option 2: .NET Blazor Server Application

A C# Blazor Server application for enterprise environments.

#### Prerequisites

```bash
# Install .NET 8 SDK (if not already installed)
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0

# Add to PATH
export PATH="$HOME/.dotnet:$PATH"
```

#### Running the Blazor App

```bash
# Terminal 1: Start FastAPI Backend (required)
cd supply_chain_ml_projects
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Blazor Frontend
cd supply_chain_ml_projects/webapp
dotnet run

# Open in browser (port shown in terminal, usually 5032)
http://localhost:5032
```

### Option 3: Streamlit Dashboard (Legacy)

```bash
cd supply_chain_ml_projects
pip install streamlit plotly
streamlit run dashboard/app.py
```

## 🔌 FastAPI Backend

The backend serves ML model predictions via REST API.

```bash
# Start the API
uvicorn api.main:app --reload --port 8000

# API Endpoints
GET  /                        # API info
GET  /health                  # Health check
POST /api/delay/predict       # Delay prediction
GET  /api/delay/stats         # Delay statistics
POST /api/risk/classify       # Risk classification
GET  /api/risk/distribution   # Risk distribution
POST /api/forecast/predict    # Demand forecast
POST /api/anomaly/detect      # Anomaly detection
GET  /api/anomaly/alerts      # Recent alerts

# Interactive API Docs
http://localhost:8000/docs
```

## 📊 Dataset

The dataset (`data/dynamic_supply_chain_logistics_dataset.csv`) contains **32,067 records** with 26 features covering:

- Vehicle GPS & Fuel data
- Traffic & Warehouse logistics
- Weather & Port conditions
- Supplier reliability metrics
- IoT sensor readings
- Risk classifications

## 🚀 Quick Start (ML Projects)

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
├── api/                                # FastAPI backend
│   ├── main.py                         # Main application
│   └── routers/                        # API endpoints
│       ├── delay.py                    # Delay prediction
│       ├── risk.py                     # Risk classification
│       ├── forecast.py                 # Demand forecasting
│       └── anomaly.py                  # Anomaly detection
├── frontend/                           # HTML/CSS/JS Dashboard
│   ├── index.html                      # Main page
│   ├── styles.css                      # Premium dark theme
│   └── app.js                          # JavaScript logic
├── webapp/                             # .NET Blazor Dashboard
│   ├── Components/Pages/               # Razor pages
│   └── Program.cs                      # Entry point
├── dashboard/                          # Streamlit Dashboard (legacy)
├── project_1_delivery_delay_prediction/
├── project_2_risk_classification/
├── project_3_demand_forecasting/
└── project_4_anomaly_detection/
```

## 🛠️ Tech Stack

- **Core**: Python 3.10+, NumPy, Pandas, Scikit-learn
- **ML**: XGBoost, LightGBM, TensorFlow/Keras
- **Time Series**: Prophet, statsmodels
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Enterprise**: ASP.NET Core 8, Blazor Server
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📄 License

MIT License
