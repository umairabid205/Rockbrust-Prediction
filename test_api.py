#!/usr/bin/env python3
"""
Rockburst Prediction API Test Script
====================================
Test script to demonstrate API functionality including single predictions,
batch predictions, and model management.

Usage: python test_api.py
"""

import requests
import json
import time
from datetime import datetime


class RockburstAPITester:
    """Test client for the Rockburst Prediction API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test the health check endpoint"""
        print("🏥 Testing Health Check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ API Status: {health_data['status']}")
                print(f"   🤖 Model Loaded: {health_data['model_loaded']}")
                print(f"   🔗 MLflow Connected: {health_data['mlflow_connected']}")
                print(f"   ⏱️  Uptime: {health_data['uptime_seconds']:.1f}s")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {str(e)}")
            return False
    
    def test_model_info(self):
        """Test the model info endpoint"""
        print("\\n📊 Testing Model Info...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                print(f"   📋 Model: {model_info['model_name']}")
                print(f"   🏷️  Version: {model_info['version']}")
                print(f"   🎯 Accuracy: {model_info.get('accuracy', 'N/A')}")
                print(f"   📈 Features: {model_info['features_count']}")
                print(f"   💾 Size: {model_info['model_size_mb']} MB")
                return True
            else:
                print(f"   ❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Model info error: {str(e)}")
            return False
    
    def test_single_prediction(self):
        """Test single prediction endpoint"""
        print("\\n🎯 Testing Single Prediction...")
        
        # Sample data for high-risk scenario
        test_data = {
            "Duration_days": 18.5,
            "Energy_Unit_log": 6.2,
            "Energy_density_Joule_sqr": 750.0,
            "Volume_m3_sqr": 180.0,
            "Event_freq_unit_per_day_log": 3.4,
            "Energy_Joule_per_day_sqr": 1200.0,
            "Volume_m3_per_day_sqr": 480.0,
            "Energy_per_Volume_log": 4.1
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict", json=test_data)
            if response.status_code == 200:
                prediction = response.json()
                print(f"   🔮 Prediction: {'Rockburst' if prediction['prediction'] == 1 else 'No Rockburst'}")
                print(f"   📊 Probability: {prediction['probability']:.3f}")
                print(f"   ⚠️  Risk Level: {prediction['risk_level']}")
                print(f"   🎯 Confidence: {prediction['confidence']:.3f}")
                return True
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Prediction error: {str(e)}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        print("\\n📦 Testing Batch Prediction...")
        
        # Multiple test scenarios
        batch_data = {
            "predictions": [
                {
                    "Duration_days": 8.2,
                    "Energy_Unit_log": 3.8,
                    "Energy_density_Joule_sqr": 180.0,
                    "Volume_m3_sqr": 65.0,
                    "Event_freq_unit_per_day_log": 1.5,
                    "Energy_Joule_per_day_sqr": 320.0,
                    "Volume_m3_per_day_sqr": 140.0,
                    "Energy_per_Volume_log": 2.2
                },
                {
                    "Duration_days": 22.5,
                    "Energy_Unit_log": 7.1,
                    "Energy_density_Joule_sqr": 920.0,
                    "Volume_m3_sqr": 220.0,
                    "Event_freq_unit_per_day_log": 4.2,
                    "Energy_Joule_per_day_sqr": 1580.0,
                    "Volume_m3_per_day_sqr": 630.0,
                    "Energy_per_Volume_log": 4.8
                },
                {
                    "Duration_days": 12.0,
                    "Energy_Unit_log": 4.9,
                    "Energy_density_Joule_sqr": 350.0,
                    "Volume_m3_sqr": 95.0,
                    "Event_freq_unit_per_day_log": 2.4,
                    "Energy_Joule_per_day_sqr": 680.0,
                    "Volume_m3_per_day_sqr": 280.0,
                    "Energy_per_Volume_log": 3.1
                }
            ]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict/batch", json=batch_data)
            if response.status_code == 200:
                batch_result = response.json()
                print(f"   📊 Batch ID: {batch_result['batch_id']}")
                print(f"   🔢 Total Predictions: {batch_result['total_predictions']}")
                print(f"   ⏱️  Processing Time: {batch_result['processing_time_seconds']:.3f}s")
                
                summary = batch_result['summary']
                print(f"   ✅ Low Risk: {summary['total_low_risk']}")
                print(f"   ⚠️  High Risk: {summary['total_high_risk']}")
                print(f"   📈 Avg Probability: {summary['average_probability']:.3f}")
                
                print("   🔍 Individual Results:")
                for i, pred in enumerate(batch_result['predictions']):
                    result = "Rockburst" if pred['prediction'] == 1 else "No Rockburst"
                    print(f"      Sample {i+1}: {result} ({pred['probability']:.3f}, {pred['risk_level']})")
                
                return True
            else:
                print(f"   ❌ Batch prediction failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Batch prediction error: {str(e)}")
            return False
    
    def test_sample_input(self):
        """Test sample input endpoint"""
        print("\\n📋 Testing Sample Input...")
        try:
            response = self.session.get(f"{self.base_url}/predict/sample")
            if response.status_code == 200:
                sample = response.json()
                print("   ✅ Sample input retrieved:")
                for key, value in sample.items():
                    print(f"      {key}: {value}")
                return True
            else:
                print(f"   ❌ Sample input failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Sample input error: {str(e)}")
            return False
    
    def test_metrics(self):
        """Test metrics endpoint"""
        print("\\n📈 Testing Metrics...")
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"   ⏰ API Uptime: {metrics.get('api_uptime_hours', 0):.2f} hours")
                print(f"   🤖 Model Loaded: {metrics.get('model_loaded', False)}")
                print(f"   📊 API Version: {metrics.get('api_version', 'N/A')}")
                if metrics.get('model_accuracy'):
                    print(f"   🎯 Model Accuracy: {metrics['model_accuracy']:.4f}")
                return True
            else:
                print(f"   ❌ Metrics failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Metrics error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Rockburst Prediction API Test Suite")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Info", self.test_model_info),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Sample Input", self.test_sample_input),
            ("Metrics", self.test_metrics)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"   ❌ Test '{test_name}' failed with exception: {str(e)}")
        
        print("\\n" + "=" * 50)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the API server and model status.")
        
        return passed == total


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rockburst Prediction API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--single", action="store_true", help="Run single prediction test only")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    tester = RockburstAPITester(base_url)
    
    print(f"🔍 Testing API at: {base_url}")
    
    if args.single:
        print("\\n🎯 Running single prediction test...")
        tester.test_single_prediction()
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main()
