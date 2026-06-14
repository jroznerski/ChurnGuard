"""Pydantic request / response schemas for customer prediction endpoints."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CustomerFeatures(BaseModel):
    """Single-customer feature payload."""

    gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    senior_citizen: Literal[0, 1] = Field(..., examples=[0])
    partner: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    dependents: Literal["Yes", "No"] = Field(..., examples=["No"])
    tenure: Annotated[int, Field(ge=0, le=72)] = Field(..., examples=[12])
    phone_service: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., examples=["No"]
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., examples=["Fiber optic"]
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["Yes"]
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["Yes"]
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., examples=["No"]
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., examples=["Month-to-month"]
    )
    paperless_billing: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., examples=["Electronic check"])
    monthly_charges: Annotated[float, Field(ge=0, le=500)] = Field(
        ..., examples=[79.85]
    )
    total_charges: Annotated[float, Field(ge=0)] = Field(..., examples=[958.20])

    @model_validator(mode="after")
    def validate_internet_services(self) -> "CustomerFeatures":
        no_internet_services = [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]
        if self.internet_service == "No":
            for svc in no_internet_services:
                val = getattr(self, svc)
                if val not in ("No", "No internet service"):
                    setattr(self, svc, "No internet service")
        return self


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)
    include_probabilities: bool = Field(default=True)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: Literal["Low", "Medium", "High"]
    threshold_used: float
    explanation: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: list[dict]
    summary: dict
