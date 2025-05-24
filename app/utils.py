import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    try:
        df = pd.DataFrame([data])
        
        # Handle categorical encoding (should match training preprocessing)
        categorical_columns = ['gender', 'category_preference', 'price_range']
        
        # Create dummy variables
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input data"""
    required_fields = ['user_id', 'age', 'gender', 'category_preference', 'price_range']
    
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate data types and ranges
    if not isinstance(data['user_id'], int) or data['user_id'] < 0:
        return False
    
    if not isinstance(data['age'], int) or not (18 <= data['age'] <= 100):
        return False
    
    if data['gender'] not in ['M', 'F']:
        return False
    
    if data['category_preference'] not in ['Electronics', 'Fashion', 'Home', 'Sports']:
        return False
    
    if data['price_range'] not in ['Low', 'Medium', 'High']:
        return False
    
    return True

def format_recommendations(predictions: Any, confidence: float) -> Dict[str, Any]:
    """Format model predictions into readable recommendations"""
    if not isinstance(predictions, (list, np.ndarray)):
        predictions = [predictions]
    
    recommendations = []
    for i, pred in enumerate(predictions):
        recommendations.append({
            "rank": i + 1,
            "product_category": f"category_{pred}",
            "confidence_score": round(confidence - i * 0.1, 2),
            "recommendation_reason": f"Based on user preference analysis"
        })
    
    return {
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
        "overall_confidence": confidence
    }
