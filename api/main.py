"""
FastAPI Backend for Supply Chain ML Models
Serves prediction endpoints for all 4 ML projects.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from api.routers import delay, risk, forecast, anomaly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supply Chain AI API",
    description="REST API for supply chain ML model predictions",
    version="1.0.0"
)

# CORS for Blazor frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:5001", "https://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(delay.router, prefix="/api/delay", tags=["Delay Prediction"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Classification"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Demand Forecasting"])
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["Anomaly Detection"])

# Global model cache
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODELS = {}

@app.on_event("startup")
async def load_models():
    """Load all ML models on startup."""
    logger.info("Loading ML models...")
    
    model_paths = {
        "delay_classifier": PROJECT_ROOT / "project_1_delivery_delay_prediction/models/delay_classifier.pkl",
        "risk_classifier": PROJECT_ROOT / "project_2_risk_classification/models/risk_classifier.pkl",
        "prophet": PROJECT_ROOT / "project_3_demand_forecasting/models/prophet_model.pkl",
        "isolation_forest": PROJECT_ROOT / "project_4_anomaly_detection/models/isolation_forest.pkl",
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                MODELS[name] = joblib.load(path)
                logger.info(f"Loaded {name} from {path}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.warning(f"Model not found: {path}")
    
    # Load dataset for reference
    data_path = PROJECT_ROOT / "data/dynamic_supply_chain_logistics_dataset.csv"
    if data_path.exists():
        MODELS["dataset"] = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(MODELS['dataset'])} records")

@app.get("/")
async def root():
    return {
        "message": "Supply Chain AI API",
        "endpoints": ["/api/delay", "/api/risk", "/api/forecast", "/api/anomaly"],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": list(MODELS.keys())}

def get_models():
    return MODELS
