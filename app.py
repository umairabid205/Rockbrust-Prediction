#!/usr/bin/env python3
"""
Rockburst Prediction API
========================
FastAPI-based REST API for real-time rockburst prediction using trained Random Forest models.
Provides endpoints for single predictions, batch predictions, model management, and monitoring.

Features:
- Real-time prediction endpoints
- Model serving with automatic loading
- Health checks and monitoring
- Batch prediction capabilities
- Model management and versioning
- MLflow integration for model loading

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import json
import traceback
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add project paths
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./exception_logging')

# Project imports
from models.model import RockburstRandomForestModel
from exception_logging.logger import get_logger

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

#InfluxDB integration
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

# Initialize logger
logger = get_logger("rockburst_api")

# API Configuration
API_VERSION = "1.0.0"
MODEL_DIR = "./artifacts/models"
MLFLOW_TRACKING_URI = "http://localhost:5001"

# InfluxDB configuration
INFLUXDB_CONFIG = {
    "url": "http://localhost:8086",
    "token": "rockbrust-token-12345", 
    "org": "rockbrust",
    "bucket": "rockbrust-data"
}

# Initialize InfluxDB client (will be set in startup)
influxdb_client_instance = None
influxdb_write_api = None

def init_influxdb():
    """Initialize InfluxDB client"""
    global influxdb_client_instance, influxdb_write_api
    
    if not INFLUXDB_AVAILABLE:
        logger.warning("InfluxDB client not available - imports failed")
        return False
        
    try:
        influxdb_client_instance = InfluxDBClient(
            url=INFLUXDB_CONFIG["url"],
            token=INFLUXDB_CONFIG["token"],
            org=INFLUXDB_CONFIG["org"]
        )
        # Test the connection
        influxdb_client_instance.health()
        influxdb_write_api = influxdb_client_instance.write_api(write_options=SYNCHRONOUS)
        logger.info("✅ InfluxDB client initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"❌ InfluxDB initialization failed: {str(e)}")
        influxdb_client_instance = None
        influxdb_write_api = None
        return False


# Pydantic Models for API
class RockburstInput(BaseModel):
    """Single prediction input model"""
    Duration_days: float = Field(..., ge=0.1, le=100, description="Duration in days")
    Energy_Unit_log: float = Field(..., ge=0, le=10, description="Energy unit (log scale)")
    Energy_density_Joule_sqr: float = Field(..., ge=0, le=10000, description="Energy density (Joule squared)")
    Volume_m3_sqr: float = Field(..., ge=0, le=1000, description="Volume (m³ squared)")
    Event_freq_unit_per_day_log: float = Field(..., ge=0, le=10, description="Event frequency per day (log scale)")
    Energy_Joule_per_day_sqr: float = Field(..., ge=0, le=10000, description="Energy per day (Joule squared)")
    Volume_m3_per_day_sqr: float = Field(..., ge=0, le=2000, description="Volume per day (m³ squared)")
    Energy_per_Volume_log: float = Field(..., ge=0, le=10, description="Energy per volume ratio (log scale)")
    
    @validator('*')
    def validate_numeric(cls, v):
        """Ensure all inputs are numeric"""
        if not isinstance(v, (int, float)):
            raise ValueError('All inputs must be numeric')
        return float(v)

    class Config:
        schema_extra = {
            "example": {
                "Duration_days": 15.5,
                "Energy_Unit_log": 5.2,
                "Energy_density_Joule_sqr": 450.0,
                "Volume_m3_sqr": 120.0,
                "Event_freq_unit_per_day_log": 2.8,
                "Energy_Joule_per_day_sqr": 890.0,
                "Volume_m3_per_day_sqr": 350.0,
                "Energy_per_Volume_log": 3.4
            }
        }


class BatchRockburstInput(BaseModel):
    """Batch prediction input model"""
    predictions: List[RockburstInput] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "Duration_days": 15.5,
                        "Energy_Unit_log": 5.2,
                        "Energy_density_Joule_sqr": 450.0,
                        "Volume_m3_sqr": 120.0,
                        "Event_freq_unit_per_day_log": 2.8,
                        "Energy_Joule_per_day_sqr": 890.0,
                        "Volume_m3_per_day_sqr": 350.0,
                        "Energy_per_Volume_log": 3.4
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response model"""
    prediction: int = Field(..., description="Prediction result: 0 (No Rockburst), 1 (Rockburst)")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    risk_level: str = Field(..., description="Risk assessment: Low, Medium, High, Critical")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    prediction_time: str = Field(..., description="ISO timestamp of prediction")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "High",
                "confidence": 0.92,
                "prediction_time": "2025-08-10T15:30:45.123456",
                "model_version": "v1.0.0"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    batch_id: str
    total_predictions: int
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    last_trained: Optional[str]
    accuracy: Optional[float]
    features_count: int
    model_size_mb: float
    is_loaded: bool
    training_samples: Optional[int]


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    mlflow_connected: bool
    uptime_seconds: float


# Global model instance
model_instance: Optional[RockburstRandomForestModel] = None
app_start_time = datetime.now()


# InfluxDB Helper Functions
def write_to_influxdb(input_data: Dict, prediction_result: Dict, measurement: str = "rockburst_predictions"):
    """Write input data and prediction results to InfluxDB"""
    if not INFLUXDB_AVAILABLE:
        logger.warning("InfluxDB client not available - skipping data save")
        return False
    
    try:
        with InfluxDBClient(
            url=INFLUXDB_CONFIG["url"],
            token=INFLUXDB_CONFIG["token"],
            org=INFLUXDB_CONFIG["org"]
        ) as client:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            
            # Create point with timestamp
            point = Point(measurement).time(datetime.utcnow())
            
            # Add input features as fields
            for key, value in input_data.items():
                point = point.field(f"input_{key}", float(value))
            
            # Add prediction results as fields
            point = point.field("prediction", int(prediction_result.get("prediction", 0)))
            point = point.field("probability", float(prediction_result.get("probability", 0.0)))
            point = point.field("confidence", float(prediction_result.get("confidence", 0.0)))
            point = point.field("risk_level", prediction_result.get("risk_level", "Unknown"))
            point = point.field("model_version", prediction_result.get("model_version", "unknown"))
            
            # Add metadata
            point = point.tag("source", "api")
            point = point.tag("prediction_type", "single")
            
            # Write to InfluxDB
            write_api.write(bucket=INFLUXDB_CONFIG["bucket"], record=point)
            logger.info(f"Data successfully written to InfluxDB: {measurement}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to write to InfluxDB: {str(e)}")
        return False


def write_batch_to_influxdb(batch_data: List[Dict], batch_results: List[Dict], measurement: str = "rockburst_batch_predictions"):
    """Write batch input data and prediction results to InfluxDB"""
    if not INFLUXDB_AVAILABLE:
        logger.warning("InfluxDB client not available - skipping batch data save")
        return False
    
    try:
        with InfluxDBClient(
            url=INFLUXDB_CONFIG["url"],
            token=INFLUXDB_CONFIG["token"],
            org=INFLUXDB_CONFIG["org"]
        ) as client:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            timestamp = datetime.utcnow()
            
            for i, (input_data, result) in enumerate(zip(batch_data, batch_results)):
                # Create point for each prediction
                point = Point(measurement).time(timestamp)
                
                # Add input features as fields
                for key, value in input_data.items():
                    point = point.field(f"input_{key}", float(value))
                
                # Add prediction results as fields
                point = point.field("prediction", int(result.get("prediction", 0)))
                point = point.field("probability", float(result.get("probability", 0.0)))
                point = point.field("confidence", float(result.get("confidence", 0.0)))
                point = point.field("risk_level", result.get("risk_level", "Unknown"))
                point = point.field("model_version", result.get("model_version", "unknown"))
                
                # Add metadata
                point = point.tag("source", "api")
                point = point.tag("prediction_type", "batch")
                point = point.tag("batch_index", str(i))
                
                points.append(point)
            
            # Write all points to InfluxDB
            write_api.write(bucket=INFLUXDB_CONFIG["bucket"], record=points)
            logger.info(f"Batch data ({len(points)} records) successfully written to InfluxDB: {measurement}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to write batch data to InfluxDB: {str(e)}")
        return False


# FastAPI App
app = FastAPI(
    title="Rockburst Prediction API",
    description="Real-time rockburst prediction using Random Forest ML models",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility Functions
def get_risk_level(probability: float) -> str:
    """Convert probability to risk level with adjusted thresholds for better variety"""
    if probability < 0.4:
        return "Low"
    elif probability < 0.55:
        return "Medium"  
    elif probability < 0.75:
        return "High"
    else:
        return "Intense"


def calculate_confidence(probability: float, model_accuracy: float = 0.95) -> float:
    """Calculate prediction confidence based on probability and model accuracy"""
    # Simple confidence calculation: closer to 0 or 1 = higher confidence
    distance_from_center = abs(probability - 0.5) * 2
    return min(distance_from_center * model_accuracy, 1.0)


def load_model_from_mlflow(model_name: str = "rockburst_random_forest_production") -> bool:
    """Load model from MLflow model registry"""
    global model_instance
    
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping MLflow model loading")
        return False
    
    try:
        # Configure MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load latest model version
        model_uri = f"models:/{model_name}/latest"
        sklearn_model = mlflow.sklearn.load_model(model_uri)
        
        # Create wrapper instance
        model_instance = RockburstRandomForestModel()
        model_instance.model = sklearn_model
        model_instance.is_trained = True
        model_instance.last_training_time = datetime.now()
        
        logger.info(f"✅ Model loaded from MLflow: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from MLflow: {str(e)}")
        return False


def load_local_model() -> bool:
    """Load model from local artifacts directory"""
    global model_instance
    
    try:
        model_instance = RockburstRandomForestModel()
        success = model_instance.load_model(MODEL_DIR)
        
        if success:
            logger.info(f"✅ Model loaded from local directory: {MODEL_DIR}")
            return True
        else:
            logger.error(f"❌ Failed to load model from local directory: {MODEL_DIR}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading local model: {str(e)}")
        return False


def ensure_model_loaded() -> RockburstRandomForestModel:
    """Ensure model is loaded, raise exception if not available"""
    global model_instance
    
    if model_instance is None or not model_instance.is_trained:
        # Try to load model
        if not (load_model_from_mlflow() or load_local_model()):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available. Please ensure model is trained and saved."
            )
    
    return model_instance


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global model_instance, app_start_time
    
    app_start_time = datetime.now()
    logger.info("🚀 Starting Rockburst Prediction API...")
    logger.info(f"📊 API Version: {API_VERSION}")
    logger.info(f"🔗 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Initialize InfluxDB client
    init_influxdb()
    
    # Try to load model on startup
    if not (load_model_from_mlflow() or load_local_model()):
        logger.warning("⚠️ No model loaded on startup - will attempt to load on first request")
    else:
        logger.info("✅ Model loaded successfully on startup")


# Health Check Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Rockburst Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive health check endpoint"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    # Check MLflow connectivity
    mlflow_connected = False
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.get_experiment_by_name("default")
            mlflow_connected = True
        except:
            pass
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=API_VERSION,
        model_loaded=model_instance is not None and model_instance.is_trained,
        mlflow_connected=mlflow_connected,
        uptime_seconds=uptime
    )


# Model Management Endpoints
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model"""
    model = ensure_model_loaded()
    
    model_info = model.get_model_info()
    
    # Calculate model file size
    model_size = 0
    model_path = os.path.join(MODEL_DIR, 'rockburst_rf_model.pkl')
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    return ModelInfo(
        model_name="Rockburst Random Forest",
        version=model_info.get('version', 'v1.0.0'),
        last_trained=model_info.get('last_training_time'),
        accuracy=model_info.get('accuracy'),
        features_count=model_info.get('total_features', 585),
        model_size_mb=round(model_size, 2),
        is_loaded=True,
        training_samples=model_info.get('training_samples')
    )


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (try MLflow first, then local)"""
    global model_instance
    model_instance = None
    
    def reload_task():
        if not (load_model_from_mlflow() or load_local_model()):
            logger.error("Failed to reload model from any source")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Model reload initiated",
        "timestamp": datetime.now().isoformat()
    }


# Prediction Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: RockburstInput):
    """
    Make a single rockburst prediction
    
    Accepts seismic/mining parameters and returns prediction with probability and risk level.
    """
    model = ensure_model_loaded()
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Apply feature engineering if feature engineer is available
        if hasattr(model, 'feature_engineer') and model.feature_engineer is not None:
            logger.info("Applying feature engineering to input data")
            
            # The feature engineering expects multiple samples for binning
            # Create a batch of varied samples to avoid binning issues
            temp_df = input_df.copy()
            for i in range(20):  # Create 20 samples with slight variations
                varied_sample = input_df.copy()
                # Add small random variations to avoid duplicate values
                for col in varied_sample.columns:
                    if varied_sample[col].dtype in ['float64', 'int64']:
                        noise = np.random.normal(0, varied_sample[col].abs() * 0.01)  # 1% noise
                        varied_sample[col] = varied_sample[col] + noise
                temp_df = pd.concat([temp_df, varied_sample], ignore_index=True)
            
            temp_df['Intensity_Level_encoded'] = 1  # Dummy target for feature engineering
            
            # Apply feature engineering
            engineered_features = model.feature_engineer.create_geological_features(
                temp_df, target_col='Intensity_Level_encoded'
            )
            
            # Take only the first sample (all are the same input)
            engineered_features = engineered_features.iloc[:1]
            
            # Remove the dummy target column
            if 'Intensity_Level_encoded' in engineered_features.columns:
                engineered_features = engineered_features.drop('Intensity_Level_encoded', axis=1)
                
            logger.info(f"Features after engineering: {engineered_features.shape[1]}")
            
            # Apply scaling if scaler is available
            if hasattr(model, 'scaler') and model.scaler is not None:
                logger.info("Applying scaling to engineered features")
                # Only scale the features that were used in training
                if hasattr(model, 'model') and model.model is not None and hasattr(model.model, 'feature_names_in_'):
                    model_features = model.model.feature_names_in_
                    
                    # Ensure we have all required features
                    missing_features = set(model_features) - set(engineered_features.columns)
                    if missing_features:
                        logger.warning(f"Missing features: {list(missing_features)[:5]}...")
                        # Add missing features with zero values
                        for feature in missing_features:
                            engineered_features[feature] = 0.0
                    
                    # Select only the features used in training in the correct order
                    engineered_features = engineered_features[model_features]
                
                # Apply scaling
                scaled_features = model.scaler.transform(engineered_features)
                scaled_df = pd.DataFrame(scaled_features, columns=engineered_features.columns)
                prediction_features = scaled_df
            else:
                prediction_features = engineered_features
        else:
            logger.warning("No feature engineering available - using raw features")
            prediction_features = input_df
        
        # Make prediction
        prediction_start = datetime.now()
        if hasattr(model, 'model') and model.model is not None:
            prediction = model.model.predict(prediction_features)[0]
            probabilities = model.model.predict_proba(prediction_features)[0]
        else:
            # Fallback to direct model methods if available
            prediction = model.predict(prediction_features)[0]
            probabilities = model.predict_proba(prediction_features)[0]
        prediction_time = datetime.now()
        
        # Get probability for the predicted class
        probability = probabilities[prediction]
        
        # Calculate confidence and risk level
        model_info = model.get_model_info()
        model_accuracy = model_info.get('accuracy', 0.95)
        confidence = calculate_confidence(probability, model_accuracy)
        risk_level = get_risk_level(probability)
        
        logger.info(f"Prediction made: {prediction} (prob: {probability:.3f}, risk: {risk_level})")
        
        # Create response object
        response_data = {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "prediction_time": prediction_time.isoformat(),
            "model_version": model_info.get('version', 'v1.0.0')
        }
        
        # Save to InfluxDB
        try:
            input_dict = input_data.dict()
            write_to_influxdb(input_dict, response_data)
        except Exception as e:
            logger.warning(f"Failed to save prediction to InfluxDB: {str(e)}")
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchRockburstInput):
    """
    Make batch predictions for multiple samples
    
    Accepts a list of geological parameters and returns predictions for all samples.
    """
    model = ensure_model_loaded()
    
    try:
        batch_start = datetime.now()
        predictions = []
        
        # Convert all inputs to DataFrame
        input_dicts = [pred.dict() for pred in batch_input.predictions]
        batch_df = pd.DataFrame(input_dicts)
        
        # Apply feature engineering if feature engineer is available
        if hasattr(model, 'feature_engineer') and model.feature_engineer is not None:
            logger.info("Applying feature engineering to batch data")
            
            # Add dummy target for feature engineering  
            batch_df['Intensity_Level_encoded'] = 1  # Dummy target for feature engineering
            
            # Apply feature engineering
            engineered_features = model.feature_engineer.create_geological_features(
                batch_df, target_col='Intensity_Level_encoded'
            )
            
            # Remove the dummy target column
            if 'Intensity_Level_encoded' in engineered_features.columns:
                engineered_features = engineered_features.drop('Intensity_Level_encoded', axis=1)
                
            logger.info(f"Features after engineering: {engineered_features.shape[1]}")
            
            # Apply scaling if scaler is available
            if hasattr(model, 'scaler') and model.scaler is not None:
                logger.info("Applying scaling to engineered features")
                # Only scale the features that were used in training
                if hasattr(model, 'model') and model.model is not None and hasattr(model.model, 'feature_names_in_'):
                    model_features = model.model.feature_names_in_
                    
                    # Ensure we have all required features
                    missing_features = set(model_features) - set(engineered_features.columns)
                    if missing_features:
                        logger.warning(f"Missing features: {list(missing_features)[:5]}...")
                        # Add missing features with zero values
                        for feature in missing_features:
                            engineered_features[feature] = 0.0
                    
                    # Select only the features used in training in the correct order
                    engineered_features = engineered_features[model_features]
                
                # Apply scaling
                scaled_features = model.scaler.transform(engineered_features)
                scaled_df = pd.DataFrame(scaled_features, columns=engineered_features.columns)
                prediction_df = scaled_df
            else:
                prediction_df = engineered_features
        else:
            logger.warning("No feature engineering available - using raw features")
            prediction_df = batch_df
        
        # Make batch predictions
        if hasattr(model, 'model') and model.model is not None:
            batch_predictions = model.model.predict(prediction_df)
            batch_probabilities = model.model.predict_proba(prediction_df)
        else:
            # Fallback to direct model methods if available
            batch_predictions = model.predict(prediction_df)
            batch_probabilities = model.predict_proba(prediction_df)
        
        model_info = model.get_model_info()
        model_accuracy = model_info.get('accuracy', 0.95)
        model_version = model_info.get('version', 'v1.0.0')
        
        # Process each prediction
        for i, (pred, probs) in enumerate(zip(batch_predictions, batch_probabilities)):
            probability = probs[pred]
            confidence = calculate_confidence(probability, model_accuracy)
            risk_level = get_risk_level(probability)
            
            predictions.append(PredictionResponse(
                prediction=int(pred),
                probability=round(float(probability), 4),
                risk_level=risk_level,
                confidence=round(confidence, 4),
                prediction_time=datetime.now().isoformat(),
                model_version=model_version
            ))
        
        processing_time = (datetime.now() - batch_start).total_seconds()
        
        # Generate summary
        summary = {
            "total_high_risk": sum(1 for p in predictions if p.prediction == 1),
            "total_low_risk": sum(1 for p in predictions if p.prediction == 0),
            "average_probability": round(sum(p.probability for p in predictions) / len(predictions), 4),
            "average_confidence": round(sum(p.confidence for p in predictions) / len(predictions), 4),
            "risk_distribution": {
                "Low": sum(1 for p in predictions if p.risk_level == "Low"),
                "Medium": sum(1 for p in predictions if p.risk_level == "Medium"),
                "High": sum(1 for p in predictions if p.risk_level == "High"),
                "Critical": sum(1 for p in predictions if p.risk_level == "Critical")
            }
        }
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(predictions)}"
        
        logger.info(f"Batch prediction completed: {len(predictions)} samples in {processing_time:.2f}s")
        
        # Save batch data to InfluxDB
        try:
            batch_results = []
            for pred in predictions:
                batch_results.append({
                    "prediction": pred.prediction,
                    "probability": pred.probability,
                    "confidence": pred.confidence,
                    "risk_level": pred.risk_level,
                    "model_version": pred.model_version
                })
            write_batch_to_influxdb(input_dicts, batch_results)
        except Exception as e:
            logger.warning(f"Failed to save batch predictions to InfluxDB: {str(e)}")
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            batch_id=batch_id,
            total_predictions=len(predictions),
            processing_time_seconds=round(processing_time, 4)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Monitoring and Analytics Endpoints
@app.get("/predict/sample", response_model=RockburstInput)
async def get_sample_input():
    """Get a sample input for testing predictions"""
    return RockburstInput(
        Duration_days=15.5,
        Energy_Unit_log=5.2,
        Energy_density_Joule_sqr=450.0,
        Volume_m3_sqr=120.0,
        Event_freq_unit_per_day_log=2.8,
        Energy_Joule_per_day_sqr=890.0,
        Volume_m3_per_day_sqr=350.0,
        Energy_per_Volume_log=3.4
    )


@app.get("/metrics")
async def get_metrics():
    """Get API and model metrics"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    metrics = {
        "api_uptime_seconds": round(uptime, 2),
        "api_uptime_hours": round(uptime / 3600, 2),
        "model_loaded": model_instance is not None and model_instance.is_trained,
        "mlflow_available": MLFLOW_AVAILABLE,
        "model_directory": MODEL_DIR,
        "api_version": API_VERSION,
        "timestamp": datetime.now().isoformat()
    }
    
    if model_instance and model_instance.is_trained:
        model_info = model_instance.get_model_info()
        metrics.update({
            "model_accuracy": model_info.get('accuracy'),
            "model_features": model_info.get('total_features'),
            "model_last_trained": model_info.get('last_training_time'),
            "model_training_samples": model_info.get('training_samples')
        })
    
    return metrics


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with detailed error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# Main function for running the API
def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rockburst Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Model directory path")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_DIR"] = args.model_dir
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    print(f"🚀 Starting Rockburst Prediction API...")
    print(f"📊 API Version: {API_VERSION}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info"
    )


if __name__ == "__main__":
    main()
