#!/usr/bin/env python3
"""
Simple Random Forest Training Script for Rockburst Prediction
============================================================
This script trains only a Random Forest model for rockburst prediction.

Features:
- Uses the feature engineering pipeline
- Trains only Random Forest classifier
- Integrates with MLflow for experiment tracking
- Generates evaluation metrics and saves the model

Author: Data Science Team GAMMA  
Created: August 10, 2025
"""

# Standard library imports
import sys
import os
import logging
from datetime import datetime
import json
import argparse
from pathlib import Path

# Add project paths for imports
sys.path.append('.')
sys.path.append('./models')

# Data science and ML imports
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Custom modules
try:
    from models.feature_engineering import RockburstFeatureEngineering
    from exception_logging.logger import get_logger
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODULES_AVAILABLE = False
    print(f"❌ Error importing custom modules: {str(e)}")
    print("Please ensure all models are in the /models directory")
    sys.exit(1)

def generate_sample_data(n_samples=2000):
    """Generate realistic sample rockburst data"""
    
    print(f"🎲 Generating {n_samples} sample data points...")
    
    np.random.seed(42)
    
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
    
    # Convert to intensity levels (0: Low, 1: Medium, 2: High)
    # Clip risk_score to ensure it's within bounds
    risk_score = np.clip(risk_score, 0, 1)
    
    # Create categorical labels with explicit handling
    intensity_labels = []
    for score in risk_score:
        if score <= 0.35:
            intensity_labels.append(0)  # Low
        elif score <= 0.70:
            intensity_labels.append(1)  # Medium
        else:
            intensity_labels.append(2)  # High
            
    df['Intensity_Level_encoded'] = intensity_labels
    
    print("✅ Sample data generated successfully")
    return df

def main():
    """Main training script execution"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Random Forest for rockburst prediction')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of sample data points')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees in Random Forest')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum depth of trees')
    
    args = parser.parse_args()
    
    try:
        # Check if custom modules are available
        if not CUSTOM_MODULES_AVAILABLE:
            print("❌ Custom ML modules are not available. Please check the models directory.")
            sys.exit(1)
            
        # Setup logging
        logger = get_logger("random_forest_training")
        
        # Create output directories
        os.makedirs("./artifacts/random_forest", exist_ok=True)
        os.makedirs("./artifacts/random_forest/models", exist_ok=True)
        os.makedirs("./artifacts/random_forest/reports", exist_ok=True)
        
        print("🚀 Starting Random Forest Training for Rockburst Prediction")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Configure MLflow
        if not args.no_mlflow:
            mlflow.set_tracking_uri("http://localhost:5001")
            mlflow.set_experiment("rockburst_random_forest_only")
            
        # Step 1: Generate sample data
        print("\n" + "="*60)
        print("STEP 1: Data Generation")
        print("="*60)
        df = generate_sample_data(args.n_samples)
        
        print(f"📊 Data Summary:")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {list(df.columns)}")
        
        # Target distribution
        target_dist = df['Intensity_Level_encoded'].value_counts().sort_index()
        print("   Target Distribution:")
        for level, count in target_dist.items():
            percentage = (count / len(df)) * 100
            intensity_name = ['Low', 'Medium', 'High'][level]
            print(f"     {intensity_name} intensity (class {level}): {count} samples ({percentage:.1f}%)")
            
        # Step 2: Feature Engineering
        print("\n" + "="*60)
        print("STEP 2: Feature Engineering")
        print("="*60)
        
        feature_engineer = RockburstFeatureEngineering()
        enhanced_df = feature_engineer.create_comprehensive_features(df, 'Intensity_Level_encoded')
        
        print(f"✅ Feature engineering completed")
        print(f"   Features expanded from {df.shape[1]} to {enhanced_df.shape[1]}")
        
        # Step 3: Prepare data for training
        print("\n" + "="*60)
        print("STEP 3: Data Preparation")
        print("="*60)
        
        # Separate features and target
        X = enhanced_df.drop('Intensity_Level_encoded', axis=1)
        y = enhanced_df['Intensity_Level_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        print(f"📊 Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Total features: {X_train.shape[1]}")
        
        # Step 4: Train Random Forest
        print("\n" + "="*60)
        print("STEP 4: Random Forest Training")
        print("="*60)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"random_forest_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):\
            
            # Log parameters
            if not args.no_mlflow:
                mlflow.log_params({
                    'model_type': 'random_forest',
                    'n_estimators': args.n_estimators,
                    'max_depth': args.max_depth,
                    'random_state': args.random_state,
                    'n_samples': args.n_samples,
                    'n_features': X_train.shape[1],
                    'test_size': args.test_size
                })
            
            # Initialize and train Random Forest
            rf_params = {
                'n_estimators': args.n_estimators,
                'random_state': args.random_state,
                'n_jobs': -1
            }
            
            if args.max_depth is not None:
                rf_params['max_depth'] = args.max_depth
                
            print(f"🌲 Training Random Forest with parameters: {rf_params}")
            
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            
            print("✅ Random Forest training completed")
            
            # Step 5: Model Evaluation
            print("\n" + "="*60)
            print("STEP 5: Model Evaluation")
            print("="*60)
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            print(f"📊 Model Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            
            # Log metrics to MLflow
            if not args.no_mlflow:
                mlflow.log_metrics({
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                })
            
            # Detailed classification report
            class_report = classification_report(y_test, y_pred, 
                                               target_names=['Low', 'Medium', 'High'])
            print(f"\n📋 Classification Report:")
            print(class_report)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n🎯 Confusion Matrix:")
            print(cm)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Step 6: Save Model and Artifacts
            print("\n" + "="*60)
            print("STEP 6: Saving Model and Artifacts")
            print("="*60)
            
            # Save model locally
            import joblib
            model_path = "./artifacts/random_forest/models/random_forest_model.pkl"
            joblib.dump(rf_model, model_path)
            print(f"💾 Model saved locally to: {model_path}")
            
            # Save feature importance
            importance_path = "./artifacts/random_forest/reports/feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            print(f"📊 Feature importance saved to: {importance_path}")
            
            # Save classification report
            report_path = "./artifacts/random_forest/reports/classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Random Forest Classification Report\\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                f.write(f"Model Parameters:\\n{rf_params}\\n\\n")
                f.write(f"Performance Metrics:\\n")
                f.write(f"Accuracy: {accuracy:.4f}\\n")
                f.write(f"F1-Score: {f1:.4f}\\n")
                f.write(f"Precision: {precision:.4f}\\n")
                f.write(f"Recall: {recall:.4f}\\n\\n")
                f.write(f"Classification Report:\\n{class_report}")
            print(f"📄 Classification report saved to: {report_path}")
            
            # Log model to MLflow
            if not args.no_mlflow:
                mlflow.sklearn.log_model(
                    rf_model, 
                    "random_forest_model",
                    registered_model_name="rockburst_random_forest"
                )
                print("✅ Model logged to MLflow")
        
        # Final Summary
        print("\n" + "="*80)
        print("🎯 RANDOM FOREST TRAINING SUMMARY")
        print("="*80)
        print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Data samples: {args.n_samples:,}")
        print(f"🔧 Features engineered: {X_train.shape[1]:,}")
        print(f"🌲 Random Forest Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"📁 Artifacts saved to: ./artifacts/random_forest/")
        if not args.no_mlflow:
            print(f"🌐 MLflow UI: http://localhost:5001")
        print("="*80)
        
        print("\\n✅ Random Forest training completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
