"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import date as date_type, datetime
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    date: date_type = Field(..., description="Date for prediction (supports YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY)")
    gmv: float = Field(..., gt=0, description="Gross Merchandise Value")
    users: int = Field(..., gt=0, description="Number of users")
    marketing_cost: float = Field(..., gt=0, description="Marketing cost")
    
    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v):
        """
        Parse date from multiple formats:
        - ISO: YYYY-MM-DD (2024-07-15)
        - EU: DD/MM/YYYY (15/07/2024)
        - US: MM/DD/YYYY (07/15/2024)
        - DD-MM-YYYY (15-07-2024)
        - MM-DD-YYYY (07-15-2024)
        """
        if isinstance(v, date_type):
            return v
        
        if isinstance(v, str):
            # Try different formats
            formats_to_try = [
                '%Y-%m-%d',  # ISO: 2024-07-15
                '%d/%m/%Y',  # EU: 15/07/2024
                '%m/%d/%Y',  # US: 07/15/2024
                '%d-%m-%Y',  # EU with dash: 15-07-2024
                '%m-%d-%Y',  # US with dash: 07-15-2024
                '%Y/%m/%d',  # ISO with slash: 2024/07/15
            ]
            
            for date_format in formats_to_try:
                try:
                    parsed = datetime.strptime(v, date_format)
                    return parsed.date()
                except ValueError:
                    continue
            
            # If none worked, raise error with helpful message
            raise ValueError(
                f"Date '{v}' is not in a recognized format. "
                f"Supported formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, "
                f"DD-MM-YYYY, MM-DD-YYYY"
            )
        
        raise ValueError(f"Date must be string or date object, got {type(v)}")
    
    @field_validator('gmv', 'marketing_cost')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-07-15",  # or "15/07/2024" or "07/15/2024"
                "gmv": 9500000,
                "users": 85000,
                "marketing_cost": 175000
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    date: date_type
    input: dict
    predictions: dict
    confidence_intervals: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-07-15",
                "input": {
                    "gmv": 9500000,
                    "users": 85000,
                    "marketing_cost": 175000  # FIXED
                },
                "predictions": {
                    "frontend_pods": 10,
                    "backend_pods": 4
                },
                "confidence_intervals": {
                    "frontend_pods": [9, 11],
                    "backend_pods": [4, 5]
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    predictions: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    count: int
    predictions: List[dict]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None