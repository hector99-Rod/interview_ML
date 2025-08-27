# TODO: Pydantic schemas for /predict
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class PredictionInput(BaseModel):
    plan_type: str = Field(..., description="Plan type: Basic, Standard, Pro")
    contract_type: str = Field(..., description="Contract type: Monthly, Annual")
    autopay: str = Field(..., description="Autopay: Yes, No")
    is_promo_user: str = Field(..., description="Is promo user: Yes, No")
    add_on_count: int = Field(..., description="Number of add-ons")
    tenure_months: int = Field(..., description="Tenure in months")
    monthly_usage_gb: float = Field(..., description="Monthly usage in GB")
    avg_latency_ms: float = Field(..., description="Average latency in ms")
    support_tickets_30d: int = Field(..., description="Support tickets in last 30 days")
    discount_pct: float = Field(..., description="Discount percentage")
    payment_failures_90d: int = Field(..., description="Payment failures in last 90 days")
    downtime_hours_30d: float = Field(..., description="Downtime hours in last 30 days")

class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn", ge=0, le=1)
    churned: bool = Field(..., description="Predicted churn status")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")

class PredictionRequest(BaseModel):
    data: List[PredictionInput]

class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
