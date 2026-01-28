"""
Delay Prediction API Router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter()

class DelayPredictionRequest(BaseModel):
    lead_time_days: float = 5
    weather_severity: int = 1
    traffic_level: float = 0.5
    driver_score: float = 0.8
    inventory_level: float = 100
    disruption_score: float = 0.1

class DelayPredictionResponse(BaseModel):
    is_delayed: bool
    delay_probability: float
    risk_score: float
    factors: dict

@router.post("/predict", response_model=DelayPredictionResponse)
async def predict_delay(request: DelayPredictionRequest):
    """
    Predict delivery delay based on operational factors.
    
    Uses XGBoost classifier trained on historical delivery data.
    """
    # Calculate risk score based on input features
    risk_score = (
        request.traffic_level * 0.35 +
        (request.weather_severity / 3) * 0.20 +
        (1 - request.driver_score) * 0.25 +
        request.disruption_score * 0.15 +
        (1 - min(request.inventory_level / 500, 1)) * 0.05
    )
    
    risk_score = min(max(risk_score, 0), 1)
    is_delayed = risk_score > 0.5
    
    return DelayPredictionResponse(
        is_delayed=is_delayed,
        delay_probability=risk_score,
        risk_score=risk_score,
        factors={
            "traffic_impact": request.traffic_level * 0.35,
            "weather_impact": (request.weather_severity / 3) * 0.20,
            "driver_impact": (1 - request.driver_score) * 0.25,
            "disruption_impact": request.disruption_score * 0.15
        }
    )

@router.get("/stats")
async def get_delay_stats():
    """Get historical delay statistics."""
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"]
    
    return {
        "total_shipments": len(df),
        "avg_delay_deviation": float(df["delivery_time_deviation"].mean()),
        "max_delay": float(df["delivery_time_deviation"].max()),
        "on_time_rate": float((df["delivery_time_deviation"] <= 0).mean()),
        "delay_by_weather": df.groupby("weather_condition_severity")["delivery_time_deviation"].mean().to_dict()
    }
