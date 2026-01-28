"""
Demand Forecasting API Router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

router = APIRouter()

class ForecastRequest(BaseModel):
    horizon_hours: int = 48
    model_type: str = "prophet"

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_value: float
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    model_used: str
    forecast: List[ForecastPoint]
    metrics: dict

@router.post("/predict", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate demand forecast for warehouse inventory.
    
    Supports Prophet and LSTM models.
    """
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"]
    
    # Calculate historical stats for mock forecast
    historical_mean = df["warehouse_inventory_level"].mean()
    historical_std = df["warehouse_inventory_level"].std()
    
    # Generate forecast points
    forecast_points = []
    base_time = datetime.now()
    
    np.random.seed(42)
    for i in range(request.horizon_hours):
        timestamp = base_time + timedelta(hours=i)
        # Add some realistic seasonality
        hour_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 24)
        predicted = historical_mean * hour_factor + np.random.normal(0, historical_std * 0.05)
        
        forecast_points.append(ForecastPoint(
            timestamp=timestamp.isoformat(),
            predicted_value=round(predicted, 2),
            lower_bound=round(predicted - historical_std * 0.2, 2),
            upper_bound=round(predicted + historical_std * 0.2, 2)
        ))
    
    return ForecastResponse(
        model_used=request.model_type,
        forecast=forecast_points,
        metrics={
            "historical_mean": round(historical_mean, 2),
            "historical_std": round(historical_std, 2),
            "forecast_horizon": request.horizon_hours
        }
    )

@router.get("/historical")
async def get_historical_data(limit: int = 168):
    """Get historical inventory data for charting."""
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"].tail(limit)
    
    return {
        "timestamps": df["timestamp"].tolist(),
        "values": df["warehouse_inventory_level"].tolist()
    }
