"""
ML model prediction logic.
"""

import joblib
import numpy as np
from datetime import datetime


class PodPredictor:
    """Pod forecasting predictor using trained Random Forest."""
    
    def __init__(self, model_path: str = "models/model.pkl", 
                 scaler_path: str = "models/scaler.pkl"):
        """Initialize predictor with trained model."""
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.feature_names = [
            'gmv', 'users', 'marketing_cost',
            'gmv_per_user', 'marketing_eff',
            'day_of_week', 'is_weekend', 'month',
            'gmv_millions', 'users_thousands'
        ]
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        print(f"Using feature: marketing_cost (not marketing_spend)")
    
    def engineer_features(self, date: datetime, gmv: float, 
                         users: int, marketing_cost: float) -> np.ndarray:
        """Engineer features from raw input."""
        
        gmv_per_user = gmv / users
        marketing_eff = gmv / marketing_cost
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = date.month
        gmv_millions = gmv / 1_000_000
        users_thousands = users / 1_000
        
        features = np.array([[
            gmv,
            users,
            marketing_cost,
            gmv_per_user,
            marketing_eff,
            day_of_week,
            is_weekend,
            month,
            gmv_millions,
            users_thousands
        ]])
        
        return features
    
    def predict(self, date: datetime, gmv: float, 
               users: int, marketing_cost: float) -> dict:  # FIXED parameter name
        """Predict pod requirements using trained Random Forest."""
        
        features = self.engineer_features(date, gmv, users, marketing_cost)
        
        # Scale features (same as training)
        features_scaled = self.scaler.transform(features)
        
        # Predict using actual Random Forest model
        prediction = self.model.predict(features_scaled)
        
        # Extract predictions (ensure minimum 1 pod)
        frontend_pods = max(1, int(np.round(prediction[0][0])))
        backend_pods = max(1, int(np.round(prediction[0][1])))

        # Calculate confidence intervals (+/-10%)
        fe_lower = max(1, int(np.round(frontend_pods * 0.9)))
        fe_upper = int(np.round(frontend_pods * 1.1))
        be_lower = max(1, int(np.round(backend_pods * 0.9)))
        be_upper = int(np.round(backend_pods * 1.1))

        return {
            "predictions": {
                "frontend_pods": frontend_pods,
                "backend_pods": backend_pods
            },
            "confidence_intervals": {
                "frontend_pods": [fe_lower, fe_upper],
                "backend_pods": [be_lower, be_upper]
            }
        }
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "RandomForestRegressor",
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "note": "Uses marketing_cost (not marketing_spend)"
        }