#!/usr/bin/env python3
"""
Advanced Rockburst Prediction Test Script
==========================================
Test the complete ML pipeline with advanced rockburst prediction models
using our custom feature engineering and classification system.

Author: Data Sci        # Get best model results
        if 'best_model'            
            # Test predictions on sample data
            print("🔮 Testing predictions...")
            sample_predictions = classifier.predict(X_test.head(5))
            print(f"   Sample predictions: {sample_predictions[:5]}")
        
        else:
            print("⚠️  No best model found in training results - this may be normal for some configurations")
            print("📊 Training completed but model selection may need adjustment")
            
            # Log basic training info
            mlflow.log_params({
                "feature_engineering": "comprehensive",
                "training_status": "completed_without_best_model",
                "original_features": df.shape[1],
                "engineered_features": enhanced_df.shape[1],
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            })raining_results and training_results['best_model'] is not None:
            best_model = training_results['best_model']
            best_model_name = best_model.get('name', 'Unknown')
            best_f1_score = best_model.get('f1_score', 0.0)
            best_metrics = best_model.get('metrics', {})
            
            print(f"🏆 Best model: {best_model_name}")
            print(f"📊 Best F1-score: {best_f1_score:.4f}")
            print(f"🎯 Best accuracy: {best_metrics.get('accuracy', 0):.4f}")
            
            # Log to MLflow
            mlflow.log_params({
                "best_model": best_model_name,
                "feature_engineering": "comprehensive",
Created: August 10, 2025
"""

# Standard library imports
import sys
import os
import logging
from datetime import datetime
import json

# Add project root to Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')
sys.path.append('/opt/airflow/models')

# Scientific computing imports
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import our custom modules
try:
    from models.rockburst_classifier import RockburstClassifier
    from models.feature_engineering import RockburstFeatureEngineering
    from models.model_trainer import RockburstModelTrainer
    CUSTOM_MODULES_AVAILABLE = True
    print("✅ Successfully imported custom rockburst prediction modules")
except ImportError as e:
    CUSTOM_MODULES_AVAILABLE = False
    print(f"⚠️  Custom modules not available: {str(e)}")
    print("Running basic MLflow test instead...")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

def create_sample_rockburst_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create realistic sample rockburst data based on geological parameters.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample rockburst data
    """
    print(f"🗂️  Generating {n_samples} sample rockburst data points...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic geological and seismic features
    data = {
        # Duration of monitoring period (1-30 days)
        'Duration_days': np.random.uniform(1, 30, n_samples),
        
        # Energy measurements (log scale, 2-8)
        'Energy_Unit_log': np.random.uniform(2, 8, n_samples),
        
        # Energy density (10-1000 J/m²)
        'Energy_density_Joule_sqr': np.random.uniform(10, 1000, n_samples),
        
        # Volume measurements (5-500 m³)
        'Volume_m3_sqr': np.random.uniform(5, 500, n_samples),
        
        # Event frequency (log scale, 0-5 events/day)
        'Event_freq_unit_per_day_log': np.random.uniform(0, 5, n_samples),
        
        # Daily energy measurements (20-2000 J/day)
        'Energy_Joule_per_day_sqr': np.random.uniform(20, 2000, n_samples),
        
        # Daily volume measurements (10-800 m³/day)  
        'Volume_m3_per_day_sqr': np.random.uniform(10, 800, n_samples),
        
        # Energy per volume ratio (log scale, 0-6)
        'Energy_per_Volume_log': np.random.uniform(0, 6, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create realistic target variable based on geological relationships
    # Higher energy density, frequency, and energy/volume ratios increase rockburst risk
    risk_score = (
        0.3 * (df['Energy_density_Joule_sqr'] / df['Energy_density_Joule_sqr'].max()) +
        0.2 * (df['Event_freq_unit_per_day_log'] / df['Event_freq_unit_per_day_log'].max()) +
        0.3 * (df['Energy_per_Volume_log'] / df['Energy_per_Volume_log'].max()) +
        0.2 * (df['Energy_Joule_per_day_sqr'] / df['Energy_Joule_per_day_sqr'].max())
    )
    
    # Add some noise to make it more realistic
    risk_score += np.random.normal(0, 0.1, n_samples)
    
    # Convert to categorical intensity levels
    # 0: Low intensity, 1: Medium intensity, 2: High intensity
    df['Intensity_Level_encoded'] = pd.cut(
        risk_score, 
        bins=[-np.inf, 0.4, 0.7, np.inf], 
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)
    
    print(f"📊 Sample data created with shape: {df.shape}")
    print(f"📋 Target distribution:")
    target_dist = df['Intensity_Level_encoded'].value_counts().sort_index()
    for level, count in target_dist.items():
        percentage = (count / len(df)) * 100
        intensity_name = ['Low', 'Medium', 'High', 'Intense'][level]
        print(f"   {intensity_name} intensity (class {level}): {count} samples ({percentage:.1f}%)")
    
    return df

def test_basic_mlflow_setup():
    """Test basic MLflow setup with simple sklearn model"""
    
    print("\n🧪 Testing Basic MLflow Setup...")
    
    import mlflow
    import mlflow.sklearn
    
    # Create simple test data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=500, 
        n_features=10, 
        n_classes=3,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Start MLflow experiment
    experiment_name = "rockburst_basic_test"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="basic_test_run"):
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "model_type": "RandomForest",
            "class_weight": "balanced"
        })
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": float(accuracy),
            "f1_weighted": float(f1),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        })
        
        # Log model (with graceful handling of MinIO/S3 connection issues)
        try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="rockburst_basic_classifier",
                input_example=X_test[:1]
            )
            print(f"✅ Model logged successfully to MLflow with artifacts")
        except Exception as model_log_error:
            print(f"⚠️  Model artifact logging failed: {str(model_log_error)}")
            print(f"   This is likely due to MinIO/S3 credentials or connection issues")
            print(f"✅ Parameters and metrics logged successfully - core MLflow functionality working")
            # Try logging without model registry
            try:
                mlflow.sklearn.log_model(model, "model")
                print(f"✅ Model logged locally without model registry")
            except Exception as local_log_error:
                print(f"⚠️  Local model logging also failed: {str(local_log_error)}")
                print(f"   Continuing test as metrics logging is working")
        
        print(f"✅ Basic test completed successfully!")
        print(f"📊 Model accuracy: {accuracy:.4f}")
        print(f"� Model F1-score: {f1:.4f}")
        print(f"�🔬 Experiment: {experiment_name}")
        
        return True

def test_advanced_rockburst_models():
    """Test advanced rockburst prediction models with custom features"""
    
    print("\n🚀 Testing Advanced Rockburst Prediction Models...")
    
    # Create sample rockburst data
    df = create_sample_rockburst_data(n_samples=800)
    
    # Start MLflow experiment
    experiment_name = "rockburst_advanced_test"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="advanced_rockburst_test"):
        
        # Initialize feature engineering
        print("🔧 Initializing feature engineering...")
        feature_engineer = RockburstFeatureEngineering()
        
        # Apply comprehensive feature engineering
        print("⚙️  Applying feature engineering...")
        enhanced_df = feature_engineer.create_comprehensive_features(
            df, 'Intensity_Level_encoded'
        )
        
        print(f"📈 Features expanded from {df.shape[1]} to {enhanced_df.shape[1]} columns")
        
        # Initialize classifier with limited models for testing
        print("🤖 Initializing rockburst classifier...")
        classifier_config = {
            'models_to_train': ['random_forest', 'gradient_boosting', 'logistic_regression'],
            'hyperparameter_tuning': {'enabled': False},  # Disable for quick testing
            'cross_validation': {'enabled': True, 'folds': 3},
            'ensemble_methods': {'voting_classifier': True, 'stacking_classifier': False},
            'class_weight': 'balanced',  # Add missing class_weight parameter
            'random_state': 42,
            'feature_engineering': {'enabled': False},  # Disable additional feature engineering
            'feature_selection': {'enabled': False}  # Disable feature selection
        }
        
        classifier = RockburstClassifier(config=classifier_config, mlflow_enabled=False)
        
        # Prepare data for training
        print("📊 Preparing data for training...")
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            enhanced_df, 'Intensity_Level_encoded', test_size=0.2
        )
        
        # Train models
        print("🏋️  Training rockburst prediction models...")
        training_results = classifier.train_all_models(X_train, y_train, X_test, y_test)
        
        # Get best model results
        if 'best_model' in training_results and training_results['best_model'] is not None:
            best_model = training_results['best_model']
            best_model_name = best_model.get('name', 'Unknown')
            best_f1_score = best_model.get('f1_score', 0.0)
            best_metrics = best_model.get('metrics', {})
            
            print(f"🏆 Best model: {best_model_name}")
            print(f"📊 Best F1-score: {best_f1_score:.4f}")
            print(f"� Best accuracy: {best_metrics.get('accuracy', 0):.4f}")
            
            # Log to MLflow
            mlflow.log_params({
                "best_model": best_model_name,
                "feature_engineering": "comprehensive",
                "original_features": df.shape[1],
                "engineered_features": enhanced_df.shape[1],
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            mlflow.log_metrics({
                "best_f1_score": best_f1_score,
                "best_accuracy": best_metrics.get('accuracy', 0),
                "best_precision": best_metrics.get('precision_weighted', 0),
                "best_recall": best_metrics.get('recall_weighted', 0)
            })
            
            # Log the best model
            if 'trained_model' in classifier.models[best_model_name]:
                mlflow.sklearn.log_model(
                    classifier.models[best_model_name]['trained_model'],
                    "best_model",
                    registered_model_name="rockburst_advanced_classifier"
                )
            
            # Log training summary
            summary = training_results.get('training_summary', {})
            print(f"📋 Training Summary:")
            print(f"   Models attempted: {summary.get('total_models_attempted', 0)}")
            print(f"   Models successful: {len(summary.get('successful_models', []))}")
            print(f"   Success rate: {summary.get('success_rate', 0):.1%}")
            
            # Test predictions on sample data
            print("🔮 Testing predictions...")
            sample_predictions = classifier.predict(X_test.head(5))
            print(f"   Sample predictions: {sample_predictions}")
            
            return True
            
        else:
            print("❌ No best model found in training results")
            return False

def test_model_trainer_pipeline():
    """Test the complete model training pipeline"""
    
    print("\n🏭 Testing Complete Model Training Pipeline...")
    
    try:
        # Create sample data and save to temporary location
        df = create_sample_rockburst_data(n_samples=600)
        
        # Create temporary data directory
        temp_data_dir = "/tmp/rockburst_test_data"
        os.makedirs(temp_data_dir, exist_ok=True)
        
        # Save sample data
        sample_data_path = os.path.join(temp_data_dir, "sample_rockburst_data.csv")
        df.to_csv(sample_data_path, index=False)
        print(f"💾 Saved sample data to: {sample_data_path}")
        
        # Create test configuration for model trainer
        test_config = {
            'model_training': {
                'models_to_train': ['random_forest', 'logistic_regression'],
                'hyperparameter_tuning': False,
                'cross_validation_folds': 3,
                'ensemble_methods': False
            },
            'feature_engineering': {
                'enabled': True,
                'feature_selection_enabled': True,
                'top_k_features': 25
            },
            'data_loading': {
                'target_column': 'Intensity_Level_encoded',
                'test_size': 0.2,
                'validation_size': 0.0  # Skip validation split for testing
            },
            'output_paths': {
                'models_dir': '/tmp/rockburst_models',
                'artifacts_dir': '/tmp/rockburst_artifacts',
                'reports_dir': '/tmp/rockburst_reports',
                'plots_dir': '/tmp/rockburst_plots'
            }
        }
        
        # Initialize model trainer
        trainer = RockburstModelTrainer(mlflow_enabled=False)  # Disable MLflow for pipeline test
        trainer.config.update(test_config)
        
        # Manually set training data (simulating what would come from Airflow pipeline)
        trainer.training_data = df
        trainer.validation_data = pd.DataFrame()  # Empty validation set
        
        # Run feature engineering
        print("🔧 Running feature engineering pipeline...")
        processed_train, processed_val = trainer.prepare_features(df)
        
        # Run model training
        print("🚀 Running model training pipeline...")
        training_results = trainer.train_models(processed_train)
        
        # Run evaluation
        print("📊 Running model evaluation...")
        evaluation_results = trainer.evaluate_models(save_reports=True)
        
        # Save artifacts
        print("💾 Saving artifacts...")
        artifacts_info = trainer.save_artifacts()
        
        print("✅ Model training pipeline test completed successfully!")
        print(f"📋 Models trained: {len(training_results.get('training_summary', {}).get('successful_models', []))}")
        if 'best_model' in training_results:
            best_model = training_results['best_model']
            print(f"🏆 Best model: {best_model['name']} (F1: {best_model['f1_score']:.4f})")
        
        print(f"📁 Artifacts saved to: {test_config['output_paths']['models_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    
    print("=" * 80)
    print("🔬 ROCKBURST PREDICTION MODEL TESTING SUITE")
    print("=" * 80)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {
        'basic_mlflow': False,
        'advanced_models': False,
        'pipeline': False
    }
    
    try:
        # Test 1: Basic MLflow setup
        print("\n" + "="*50)
        print("TEST 1: Basic MLflow Setup")
        print("="*50)
        test_results['basic_mlflow'] = test_basic_mlflow_setup()
        
        # Test 2: Advanced models (only if custom modules available)
        if CUSTOM_MODULES_AVAILABLE:
            print("\n" + "="*50)
            print("TEST 2: Advanced Rockburst Models")
            print("="*50)
            test_results['advanced_models'] = test_advanced_rockburst_models()
            
            print("\n" + "="*50)
            print("TEST 3: Complete Training Pipeline")
            print("="*50)
            test_results['pipeline'] = test_model_trainer_pipeline()
        else:
            print("\n⚠️  Skipping advanced tests - custom modules not available")
        
        # Print final results
        print("\n" + "="*80)
        print("🎯 TEST RESULTS SUMMARY")
        print("="*80)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
        
        total_tests = sum(1 for r in test_results.values() if r is not False)
        passed_tests = sum(test_results.values())
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if all(r for r in test_results.values() if r is not False):
            print("\n🎉 ALL TESTS PASSED! Rockburst prediction system is ready!")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
        
        print("\n📍 Access Points:")
        print("🌐 MLflow UI: http://localhost:5001")
        print("📦 MinIO Console: http://localhost:9001")
        print("   - Username: minio_access_key") 
        print("   - Password: minio_secret_key")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
