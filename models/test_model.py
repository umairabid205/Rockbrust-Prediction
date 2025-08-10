"""
Model Testing Script for Rockburst Prediction
=============================================
This module provides comprehensive testing functionality for the trained
Random Forest model including prediction testing and validation.

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
sys.path.append('.')
sys.path.append('./models')

from model import RockburstRandomForestModel
from exception_logging.logger import get_logger


class ModelTester:
    """
    Handles testing and validation of the trained rockburst prediction model.
    """
    
    def __init__(self, model_dir='./artifacts/models'):
        """
        Initialize the tester.
        
        Args:
            model_dir: Directory containing the trained model
        """
        self.logger = get_logger("model_tester")
        self.model_dir = model_dir
        self.model = None
        
    def load_model(self):
        """
        Load the trained model for testing.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            self.model = RockburstRandomForestModel()
            self.model.load_model(self.model_dir)
            self.logger.info("✅ Model loaded successfully for testing")
            return True
        except (FileNotFoundError, Exception) as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def generate_test_data(self, n_samples=500):
        """
        Generate test data for model validation.
        
        Args:
            n_samples: Number of test samples to generate
            
        Returns:
            pd.DataFrame: Test data with known labels
        """
        self.logger.info(f"🎲 Generating {n_samples} test data points...")
        
        np.random.seed(123)  # Different seed for test data
        
        # Generate the 9 user input features
        data = {
            'Duration_days': np.random.uniform(1, 30, n_samples),
            'Energy_Unit_log': np.random.uniform(2, 8, n_samples),
            'Energy_density_Joule_sqr': np.random.uniform(10, 1000, n_samples),
            'Volume_m3_sqr': np.random.uniform(5, 500, n_samples),
            'Event_freq_unit_per_day_log': np.random.uniform(0, 5, n_samples),
            'Energy_Joule_per_day_sqr': np.random.uniform(20, 2000, n_samples),
            'Volume_m3_per_day_sqr': np.random.uniform(10, 800, n_samples),
            'Energy_per_Volume_log': np.random.uniform(0, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target based on geological relationships
        risk_score = (
            0.25 * (df['Energy_density_Joule_sqr'] / df['Energy_density_Joule_sqr'].max()) +
            0.20 * (df['Event_freq_unit_per_day_log'] / df['Event_freq_unit_per_day_log'].max()) +
            0.25 * (df['Energy_per_Volume_log'] / df['Energy_per_Volume_log'].max()) +
            0.15 * (df['Energy_Joule_per_day_sqr'] / df['Energy_Joule_per_day_sqr'].max()) +
            0.15 * (df['Volume_m3_per_day_sqr'] / df['Volume_m3_per_day_sqr'].max())
        )
        
        # Add geological noise for realism
        risk_score += np.random.normal(0, 0.1, n_samples)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Create intensity labels
        intensity_labels = []
        for score in risk_score:
            if score <= 0.35:
                intensity_labels.append(0)  # Low
            elif score <= 0.70:
                intensity_labels.append(1)  # Medium
            else:
                intensity_labels.append(2)  # High
                
        df['Intensity_Level_encoded'] = intensity_labels
        
        self.logger.info("✅ Test data generated successfully")
        return df
    
    def test_single_prediction(self, sample_data):
        """
        Test prediction on a single sample.
        
        Args:
            sample_data: Single row of feature data
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare features (without target)
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        else:
            sample_df = sample_data.to_frame().T if isinstance(sample_data, pd.Series) else sample_data
        
        # Remove target column if present
        feature_columns = [col for col in sample_df.columns if col != 'Intensity_Level_encoded']
        X_sample = sample_df[feature_columns]
        
        # Apply feature engineering and scaling
        X_processed, _ = self.model.prepare_features(sample_df)
        if 'Intensity_Level_encoded' in X_processed.columns:
            X_processed = X_processed.drop('Intensity_Level_encoded', axis=1)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        intensity_names = ['Low', 'Medium', 'High']
        
        result = {
            'predicted_class': int(prediction),
            'predicted_intensity': intensity_names[prediction],
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'confidence': float(max(probabilities))
        }
        
        return result
    
    def test_batch_predictions(self, test_data, target_column='Intensity_Level_encoded'):
        """
        Test predictions on a batch of data.
        
        Args:
            test_data: DataFrame with test samples
            target_column: Name of target column (for evaluation)
            
        Returns:
            dict: Batch prediction results and metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.logger.info(f"🧪 Testing model on {len(test_data)} samples...")
        
        # Prepare features
        X_test, y_test = self.model.prepare_features(test_data, target_column)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_names = ['Low', 'Medium', 'High']
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names,
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Prediction confidence statistics
        max_probabilities = np.max(y_pred_proba, axis=1)
        confidence_stats = {
            'mean_confidence': float(np.mean(max_probabilities)),
            'median_confidence': float(np.median(max_probabilities)),
            'min_confidence': float(np.min(max_probabilities)),
            'max_confidence': float(np.max(max_probabilities))
        }
        
        # Class distribution
        true_dist = pd.Series(y_test).value_counts().sort_index()
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        results = {
            'test_samples': len(test_data),
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'confidence_statistics': confidence_stats,
            'class_distribution': {
                'true': {class_names[i]: int(true_dist.get(i, 0)) for i in range(3)},
                'predicted': {class_names[i]: int(pred_dist.get(i, 0)) for i in range(3)}
            },
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        self.logger.info(f"✅ Batch testing completed. Accuracy: {accuracy:.4f}")
        return results
    
    def validate_model_integrity(self):
        """
        Validate that the model is working correctly.
        
        Returns:
            dict: Validation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.logger.info("🔍 Validating model integrity...")
        
        validation_results = {
            'model_loaded': True,
            'is_trained': self.model.is_trained,
            'can_predict': False,
            'feature_count': None,
            'expected_classes': [0, 1, 2],
            'actual_classes': None,
            'validation_passed': False
        }
        
        try:
            # Generate a small test sample
            test_sample = self.generate_test_data(n_samples=10)
            
            # Test prediction capability
            prediction_result = self.test_batch_predictions(test_sample)
            
            validation_results.update({
                'can_predict': True,
                'feature_count': test_sample.shape[1] - 1,  # Exclude target
                'actual_classes': sorted(list(set(prediction_result['predictions']))),
                'test_accuracy': prediction_result['accuracy']
            })
            
            # Check if all expected classes can be predicted
            all_classes_present = all(cls in prediction_result['predictions'] 
                                    for cls in validation_results['expected_classes'])
            
            validation_results['validation_passed'] = (
                validation_results['can_predict'] and 
                validation_results['test_accuracy'] > 0.5  # Reasonable accuracy threshold
            )
            
        except Exception as e:
            validation_results['error'] = str(e)
            self.logger.error(f"❌ Model validation failed: {str(e)}")
        
        if validation_results['validation_passed']:
            self.logger.info("✅ Model validation passed")
        else:
            self.logger.warning("⚠️ Model validation failed or showed issues")
        
        return validation_results
    
    def run_comprehensive_test(self, test_data=None, n_test_samples=500):
        """
        Run comprehensive testing suite.
        
        Args:
            test_data: Optional test data, if None generates sample data
            n_test_samples: Number of samples for generated test data
            
        Returns:
            dict: Comprehensive test results
        """
        self.logger.info("🧪 Running comprehensive model test suite...")
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model for testing")
        
        # Validate model integrity
        integrity_results = self.validate_model_integrity()
        
        # Prepare test data
        if test_data is None:
            test_data = self.generate_test_data(n_test_samples)
        
        # Run batch predictions
        batch_results = self.test_batch_predictions(test_data)
        
        # Test single prediction
        sample_row = test_data.iloc[0]
        single_prediction = self.test_single_prediction(sample_row)
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'model_integrity': integrity_results,
            'batch_test_results': batch_results,
            'single_prediction_example': single_prediction,
            'test_summary': {
                'total_samples_tested': len(test_data),
                'overall_accuracy': batch_results['accuracy'],
                'validation_passed': integrity_results['validation_passed'],
                'confidence_mean': batch_results['confidence_statistics']['mean_confidence']
            }
        }
        
        self.logger.info("✅ Comprehensive testing completed")
        return comprehensive_results


def main():
    """Main testing script"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Test Random Forest model for rockburst prediction')
    parser.add_argument('--model-dir', type=str, default='./artifacts/models',
                       help='Directory containing the trained model')
    parser.add_argument('--test-data', type=str, 
                       help='Path to test data CSV file (optional)')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of test samples to generate (if no test data file)')
    parser.add_argument('--output-file', type=str,
                       help='Save test results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = ModelTester(args.model_dir)
        
        print("🧪 Rockburst Model Testing Script")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load test data if provided
        test_data = None
        if args.test_data and os.path.exists(args.test_data):
            print(f"📄 Loading test data from: {args.test_data}")
            test_data = pd.read_csv(args.test_data)
        
        # Run comprehensive test
        results = tester.run_comprehensive_test(test_data, args.n_samples)
        
        # Print summary
        print("="*60)
        print("📊 TESTING SUMMARY")
        print("="*60)
        print(f"Model validation passed: {results['test_summary']['validation_passed']}")
        print(f"Total samples tested: {results['test_summary']['total_samples_tested']}")
        print(f"Overall accuracy: {results['test_summary']['overall_accuracy']:.4f}")
        print(f"Mean confidence: {results['test_summary']['confidence_mean']:.4f}")
        
        print(f"\\n🎯 Class Distribution:")
        batch_results = results['batch_test_results']
        for class_name, count in batch_results['class_distribution']['true'].items():
            pred_count = batch_results['class_distribution']['predicted'][class_name]
            print(f"   {class_name}: {count} actual, {pred_count} predicted")
        
        print(f"\\n📊 Classification Report:")
        class_report = batch_results['classification_report']
        for class_name in ['Low', 'Medium', 'High']:
            metrics = class_report[class_name]
            print(f"   {class_name:7}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\n💾 Test results saved to: {args.output_file}")
        
        print("="*60)
        print("✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
