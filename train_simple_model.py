"""
Simple Rockburst Model Training
===============================
Train a simple model that works directly with the 8 base seismic features without complex feature engineering.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import sys

# Add project paths
sys.path.append('.')
sys.path.append('./models')

from exception_logging.logger import get_logger

def create_simple_rockburst_data(n_samples=2000):
    """Create simple seismic data for training"""
    np.random.seed(42)
    
    # Generate the 8 base seismic features
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
    
    # Create realistic target based on seismic relationships
    risk_score = (
        0.25 * (df['Energy_density_Joule_sqr'] / df['Energy_density_Joule_sqr'].max()) +
        0.20 * (df['Event_freq_unit_per_day_log'] / df['Event_freq_unit_per_day_log'].max()) +
        0.25 * (df['Energy_per_Volume_log'] / df['Energy_per_Volume_log'].max()) +
        0.15 * (df['Energy_Joule_per_day_sqr'] / df['Energy_Joule_per_day_sqr'].max()) +
        0.15 * (df['Volume_m3_per_day_sqr'] / df['Volume_m3_per_day_sqr'].max())
    )
    
    # Add noise for realism
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = np.clip(risk_score, 0, 1)
    
    # Convert to binary classification (0: Low risk, 1: High risk)
    target = (risk_score > 0.5).astype(int)
    df['target'] = target
    
    print(f"Generated {n_samples} samples")
    print(f"High risk samples: {target.sum()} ({target.mean()*100:.1f}%)")
    
    return df

def train_simple_model():
    """Train a simple Random Forest model without complex feature engineering"""
    
    logger = get_logger("simple_model_trainer")
    logger.info("Training simple rockburst model...")
    
    # Generate training data
    print("🔄 Generating training data...")
    data = create_simple_rockburst_data(2000)
    
    # Prepare features and target
    feature_columns = [
        'Duration_days', 'Energy_Unit_log', 'Energy_density_Joule_sqr', 'Volume_m3_sqr',
        'Event_freq_unit_per_day_log', 'Energy_Joule_per_day_sqr', 'Volume_m3_per_day_sqr', 'Energy_per_Volume_log'
    ]
    
    X = data[feature_columns]
    y = data['target']
    
    print(f"Feature shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Training completed!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    print("\n💾 Saving model...")
    model_dir = './artifacts/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model, scaler, and metadata
    model_path = os.path.join(model_dir, 'simple_rockburst_model.pkl')
    scaler_path = os.path.join(model_dir, 'simple_scaler.pkl')
    metadata_path = os.path.join(model_dir, 'simple_model_metadata.json')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'model_name': 'Simple Rockburst Random Forest',
        'version': 'v1.0-simple',
        'trained_at': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'features': feature_columns,
        'n_features': len(feature_columns),
        'training_samples': len(X_train),
        'model_type': 'RandomForestClassifier'
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Model saved to: {model_path}")
    print(f"   Scaler saved to: {scaler_path}")
    print(f"   Metadata saved to: {metadata_path}")
    
    return model, scaler, metadata

if __name__ == "__main__":
    print("🚀 Simple Rockburst Model Training")
    print("="*50)
    
    try:
        model, scaler, metadata = train_simple_model()
        print("\n✅ Simple model training completed successfully!")
        print(f"   Model accuracy: {metadata['accuracy']:.1%}")
        print(f"   Features used: {metadata['n_features']}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
