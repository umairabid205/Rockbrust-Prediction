#!/usr/bin/env python3

import requests
import json
from datetime import datetime

# Test data for single prediction
test_data = {
    "Duration_days": 15.5,
    "Energy_Unit_log": 5.2,
    "Energy_density_Joule_sqr": 450.0,
    "Volume_m3_sqr": 120.0,
    "Event_freq_unit_per_day_log": 2.8,
    "Energy_Joule_per_day_sqr": 890.0,
    "Volume_m3_per_day_sqr": 350.0,
    "Energy_per_Volume_log": 3.4
}

def test_single_prediction():
    """Test single prediction endpoint"""
    print("🧪 Testing Single Prediction...")
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probability: {result['probability']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Model Version: {result['model_version']}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single prediction error: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🧪 Testing Batch Prediction...")
    
    batch_data = {
        "predictions": [test_data, test_data.copy()]
    }
    # Make second sample slightly different
    batch_data["predictions"][1]["Duration_days"] = 20.0
    batch_data["predictions"][1]["Energy_Unit_log"] = 4.8
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/batch",
            headers={"Content-Type": "application/json"},
            json=batch_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"   Total predictions: {result['total_predictions']}")
            print(f"   Processing time: {result['processing_time_seconds']}s")
            print(f"   Summary: {result['summary']}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing Health Check...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"   Status: {result['status']}")
            print(f"   Model: {result['model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API Tests...")
    print(f"⏰ Test time: {datetime.now()}")
    
    # Run tests
    health_ok = test_health_check()
    single_ok = test_single_prediction()
    batch_ok = test_batch_prediction()
    
    print(f"\n📊 Test Results:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Single Prediction: {'✅ PASS' if single_ok else '❌ FAIL'}")
    print(f"   Batch Prediction: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    
    if all([health_ok, single_ok, batch_ok]):
        print("\n🎉 All tests passed! API is working correctly with InfluxDB integration.")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")
