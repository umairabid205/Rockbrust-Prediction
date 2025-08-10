# 🚀 Rockburst Prediction API Documentation

A comprehensive FastAPI-based REST API for real-time rockburst prediction using trained Random Forest machine learning models.

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Model Management](#model-management)
- [Monitoring](#monitoring)
- [Deployment](#deployment)

## ✨ Features

### 🎯 **Real-Time Predictions**

- **Single Predictions**: Instant rockburst risk assessment
- **Batch Predictions**: Process multiple samples simultaneously
- **Risk Categorization**: Low, Medium, High, Critical risk levels
- **Confidence Scoring**: Model confidence assessment

### 🤖 **Model Serving**

- **Automatic Model Loading**: From local files or MLflow registry
- **Model Versioning**: Track and manage different model versions
- **Hot Reloading**: Reload models without API restart
- **Performance Metrics**: Real-time model performance tracking

### 🔍 **Monitoring & Health**

- **Health Checks**: API and model status monitoring
- **Performance Metrics**: Response times and throughput
- **Error Handling**: Comprehensive error responses
- **Logging**: Detailed request and error logging

### 🌐 **Developer Experience**

- **Interactive Documentation**: Auto-generated Swagger/OpenAPI docs
- **Type Safety**: Full Pydantic model validation
- **CORS Support**: Cross-origin resource sharing enabled
- **JSON Responses**: Consistent JSON API responses

## 🚀 Quick Start

### 1. **Start the API Server**

```bash
# Development mode (with auto-reload)
./start_api.sh --dev

# Production mode
./start_api.sh --prod

# Manual start
python app.py --port 8000
```

### 2. **Access the API**

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. **Make Your First Prediction**

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "seismic_activity": 75.5,
    "rock_strength": 150.0,
    "depth": 1200.0,
    "stress_level": 85.3,
    "water_pressure": 45.2,
    "temperature": 25.0,
    "ground_displacement": 12.5,
    "mining_activity": 60.0,
    "geological_conditions": 70.0
  }'
```

## 🛠 API Endpoints

### **Core Endpoints**

| Method | Endpoint  | Description                    |
| ------ | --------- | ------------------------------ |
| `GET`  | `/`       | API information and navigation |
| `GET`  | `/health` | Health check and system status |
| `GET`  | `/docs`   | Interactive API documentation  |

### **Prediction Endpoints**

| Method | Endpoint          | Description                            |
| ------ | ----------------- | -------------------------------------- |
| `POST` | `/predict`        | Single rockburst prediction            |
| `POST` | `/predict/batch`  | Batch predictions (up to 1000 samples) |
| `GET`  | `/predict/sample` | Get sample input data                  |

### **Model Management**

| Method | Endpoint        | Description                     |
| ------ | --------------- | ------------------------------- |
| `GET`  | `/model/info`   | Get model information and stats |
| `POST` | `/model/reload` | Reload model from storage       |

### **Monitoring**

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| `GET`  | `/metrics` | API and model performance metrics |

## 📊 Usage Examples

### **Single Prediction**

```python
import requests

# Prediction data
data = {
    "seismic_activity": 85.0,
    "rock_strength": 120.0,
    "depth": 1500.0,
    "stress_level": 95.0,
    "water_pressure": 60.0,
    "temperature": 35.0,
    "ground_displacement": 25.0,
    "mining_activity": 80.0,
    "geological_conditions": 40.0
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Prediction: {'Rockburst' if result['prediction'] == 1 else 'No Rockburst'}")
print(f"Probability: {result['probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.3f}")
```

**Example Response:**

```json
{
  "prediction": 1,
  "probability": 0.8547,
  "risk_level": "High",
  "confidence": 0.9234,
  "prediction_time": "2025-08-10T16:41:45.123456",
  "model_version": "v1.0.0"
}
```

### **Batch Predictions**

```python
import requests

# Multiple samples
batch_data = {
    "predictions": [
        {
            "seismic_activity": 25.0,
            "rock_strength": 200.0,
            "depth": 800.0,
            "stress_level": 45.0,
            "water_pressure": 20.0,
            "temperature": 15.0,
            "ground_displacement": 5.0,
            "mining_activity": 30.0,
            "geological_conditions": 85.0
        },
        {
            "seismic_activity": 90.0,
            "rock_strength": 100.0,
            "depth": 2000.0,
            "stress_level": 120.0,
            "water_pressure": 80.0,
            "temperature": 45.0,
            "ground_displacement": 35.0,
            "mining_activity": 95.0,
            "geological_conditions": 30.0
        }
    ]
}

response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
result = response.json()

print(f"Batch ID: {result['batch_id']}")
print(f"Total Predictions: {result['total_predictions']}")
print(f"High Risk Count: {result['summary']['total_high_risk']}")
print(f"Processing Time: {result['processing_time_seconds']}s")
```

### **Using Python Client**

```python
# Run the comprehensive test suite
python test_api.py

# Run single prediction test
python test_api.py --single
```

## 🤖 Model Management

### **Model Information**

```bash
curl http://localhost:8000/model/info
```

Response includes:

- Model name and version
- Training accuracy and metrics
- Feature count and model size
- Last training timestamp

### **Model Reloading**

```bash
curl -X POST http://localhost:8000/model/reload
```

The API automatically tries to load models in this order:

1. **MLflow Model Registry** (if available)
2. **Local Model Files** (./artifacts/models/)

## 📈 Monitoring

### **Health Check**

```bash
curl http://localhost:8000/health
```

Returns:

- API status and uptime
- Model loading status
- MLflow connectivity status
- Performance metrics

### **Performance Metrics**

```bash
curl http://localhost:8000/metrics
```

Includes:

- API uptime and version
- Model performance stats
- Request processing metrics

## 🎯 Input Parameters

All prediction endpoints accept these geological and mining parameters:

| Parameter               | Type  | Range   | Description                 |
| ----------------------- | ----- | ------- | --------------------------- |
| `seismic_activity`      | float | 0-100   | Seismic activity level      |
| `rock_strength`         | float | 0-1000  | Rock strength in MPa        |
| `depth`                 | float | 0-5000  | Depth in meters             |
| `stress_level`          | float | 0-200   | Stress level in MPa         |
| `water_pressure`        | float | 0-100   | Water pressure in MPa       |
| `temperature`           | float | -50-100 | Temperature in Celsius      |
| `ground_displacement`   | float | 0-1000  | Ground displacement in mm   |
| `mining_activity`       | float | 0-100   | Mining activity level       |
| `geological_conditions` | float | 0-100   | Geological conditions score |

## 🚀 Deployment Options

### **Development Server**

```bash
# Auto-reload enabled
python app.py --port 8000 --reload --debug
```

### **Production Server**

```bash
# Multiple workers for production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Docker Deployment**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### **Environment Variables**

| Variable              | Default                 | Description             |
| --------------------- | ----------------------- | ----------------------- |
| `MODEL_DIR`           | `./artifacts/models`    | Model storage directory |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow server URL       |
| `DEBUG`               | `false`                 | Enable debug mode       |

## 🔧 Configuration

### **API Configuration**

- **Host**: `0.0.0.0` (bind to all interfaces)
- **Port**: `8000` (default, configurable)
- **Workers**: `4` (production mode)
- **Timeout**: `30s` (request timeout)

### **Model Configuration**

- **Auto-loading**: Enabled on startup
- **Fallback**: Local model if MLflow unavailable
- **Validation**: Input parameter validation
- **Caching**: Model kept in memory

## 🛡️ Error Handling

The API provides comprehensive error responses:

```json
{
  "error": "Model not available",
  "status_code": 503,
  "timestamp": "2025-08-10T16:41:45.123456",
  "path": "/predict"
}
```

**Common Error Codes:**

- `400`: Invalid input parameters
- `503`: Model not available
- `500`: Internal server error

## 📝 Testing

### **Automated Tests**

```bash
# Run comprehensive API tests
python test_api.py

# Test specific endpoints
python test_api.py --single
```

### **Manual Testing**

1. **Health Check**: `curl http://localhost:8000/health`
2. **Model Info**: `curl http://localhost:8000/model/info`
3. **Sample Prediction**: Use the `/predict/sample` endpoint
4. **Interactive Docs**: Visit http://localhost:8000/docs

## 🤝 Integration Examples

### **JavaScript/Frontend**

```javascript
// Single prediction
const prediction = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    seismic_activity: 75.5,
    rock_strength: 150.0,
    // ... other parameters
  }),
});

const result = await prediction.json();
console.log("Risk Level:", result.risk_level);
```

### **Python Script**

```python
import requests

class RockburstPredictor:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def predict(self, geological_data):
        response = requests.post(f"{self.api_url}/predict", json=geological_data)
        return response.json()

    def batch_predict(self, data_list):
        batch_data = {"predictions": data_list}
        response = requests.post(f"{self.api_url}/predict/batch", json=batch_data)
        return response.json()

# Usage
predictor = RockburstPredictor()
result = predictor.predict({...geological_data...})
```

## 🎉 Summary

The Rockburst Prediction API provides:

✅ **Real-time predictions** with confidence scoring  
✅ **Batch processing** for multiple samples  
✅ **Model management** with automatic loading  
✅ **Production-ready** with health checks and monitoring  
✅ **Developer-friendly** with interactive documentation  
✅ **MLflow integration** for model versioning  
✅ **Comprehensive testing** suite included  
✅ **Easy deployment** options

**Ready for production use with 96.75% model accuracy!** 🚀
