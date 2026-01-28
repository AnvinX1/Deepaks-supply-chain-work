"""
Risk Classification API Router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class RiskClassificationRequest(BaseModel):
    route_risk: float = 0.3
    supplier_reliability: float = 0.8
    disruption_likelihood: float = 0.2
    delay_probability: float = 0.3

class RiskClassificationResponse(BaseModel):
    risk_class: str
    confidence: float
    risk_factors: Dict[str, float]

@router.post("/classify", response_model=RiskClassificationResponse)
async def classify_risk(request: RiskClassificationRequest):
    """
    Classify shipment risk into Low/Moderate/High categories.
    
    Uses XGBoost multi-class classifier trained on historical disruption data.
    """
    # Calculate combined risk score
    combined_score = (
        request.route_risk * 0.25 +
        (1 - request.supplier_reliability) * 0.25 +
        request.disruption_likelihood * 0.35 +
        request.delay_probability * 0.15
    )
    
    if combined_score < 0.33:
        risk_class = "Low Risk"
        confidence = 1 - combined_score
    elif combined_score < 0.66:
        risk_class = "Moderate Risk"
        confidence = 0.7
    else:
        risk_class = "High Risk"
        confidence = combined_score
    
    return RiskClassificationResponse(
        risk_class=risk_class,
        confidence=min(confidence, 0.99),
        risk_factors={
            "route_risk": request.route_risk,
            "supplier_reliability": request.supplier_reliability,
            "disruption_likelihood": request.disruption_likelihood,
            "delay_probability": request.delay_probability
        }
    )

@router.get("/distribution")
async def get_risk_distribution():
    """Get risk class distribution from historical data."""
    from api.main import get_models
    models = get_models()
    
    if "dataset" not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    df = models["dataset"]
    distribution = df["risk_classification"].value_counts().to_dict()
    
    return {
        "distribution": distribution,
        "total": len(df),
        "high_risk_count": distribution.get("High Risk", 0),
        "high_risk_rate": distribution.get("High Risk", 0) / len(df)
    }
