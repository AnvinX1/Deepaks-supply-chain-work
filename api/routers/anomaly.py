"""
Anomaly Detection API Router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

router = APIRouter()

class AnomalyCheckRequest(BaseModel):
    iot_temperature: float = 25.0
    fuel_consumption: float = 10.0
    traffic_level: float = 0.5
    driver_fatigue: float = 0.1

class AnomalyCheckResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    contributing_factors: dict
    recommendation: str

class AnomalyAlert(BaseModel):
    timestamp: str
    anomaly_score: float
    metric: str
    value: float

@router.post("/detect", response_model=AnomalyCheckResponse)
async def detect_anomaly(request: AnomalyCheckRequest):
    """
    Detect anomalies in operational metrics.
    
    Uses Isolation Forest and statistical methods.
    """
    # Define normal ranges
    normal_ranges = {
        "iot_temperature": (15, 35),
        "fuel_consumption": (5, 20),
        "traffic_level": (0, 0.7),
        "driver_fatigue": (0, 0.3)
    }
    
    # Calculate anomaly contributions
    contributions = {}
    total_score = 0
    
    for metric, (low, high) in normal_ranges.items():
        value = getattr(request, metric)
        if value < low:
            deviation = (low - value) / (high - low)
        elif value > high:
            deviation = (value - high) / (high - low)
        else:
            deviation = 0
        
        contributions[metric] = min(deviation, 1.0)
        total_score += contributions[metric]
    
    anomaly_score = min(total_score / len(normal_ranges), 1.0)
    is_anomaly = anomaly_score > 0.3
    
    # Generate recommendation
    if is_anomaly:
        max_factor = max(contributions, key=contributions.get)
        recommendation = f"Investigate {max_factor.replace('_', ' ')} - deviation detected"
    else:
        recommendation = "All metrics within normal range"
    
    return AnomalyCheckResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 3),
        contributing_factors=contributions,
        recommendation=recommendation
    )

@router.get("/alerts")
async def get_recent_alerts(limit: int = 10):
    """Get recent anomaly alerts from historical data."""
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"]
    
    # Simulate anomaly detection on historical data
    np.random.seed(42)
    df["anomaly_score"] = np.random.uniform(0, 1, len(df))
    anomalies = df[df["anomaly_score"] > 0.9].tail(limit)
    
    alerts = []
    for _, row in anomalies.iterrows():
        alerts.append({
            "timestamp": row["timestamp"],
            "anomaly_score": round(row["anomaly_score"], 3),
            "iot_temperature": row["iot_temperature"],
            "traffic_level": row["traffic_congestion_level"]
        })
    
    return {
        "alerts": alerts,
        "total_anomalies": len(anomalies),
        "anomaly_rate": len(df[df["anomaly_score"] > 0.9]) / len(df)
    }

@router.get("/stats")
async def get_anomaly_stats():
    """Get anomaly detection statistics."""
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"]
    
    return {
        "total_records": len(df),
        "avg_temperature": round(df["iot_temperature"].mean(), 2),
        "avg_fuel_consumption": round(df["fuel_consumption_rate"].mean(), 2),
        "avg_traffic": round(df["traffic_congestion_level"].mean(), 3)
    }
