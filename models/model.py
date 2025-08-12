"""
Random Forest Model Definition for Rockburst Prediction
======================================================
This module defines the Random Forest model architecture and configuration
for rockburst intensity prediction.

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append('.')
sys.path.append('./models')

from exception_logging.logger import get_logger


class RockburstRandomForestModel:
    """
    Simple Random Forest Model for Rockburst Prediction using only real features.
    """
    def __init__(self):
        self.logger = get_logger("rockburst_random_forest")
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        self.feature_names = [
            'Duration_days',
            'Energy_Unit_log',
            'Energy_density_Joule_sqr',
            'Volume_m3_sqr',
            'Event_freq_unit_per_day_log',
            'Energy_Joule_per_day_sqr',
            'Volume_m3_per_day_sqr',
            'Energy_per_Volume_log'
        ]
        self.target_name = 'Intensity_Level_encoded'
        self.class_names = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'}
        
    def load_data(self, data_path):
        """Load and prepare data from CSV file."""
        self.logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        X = df[self.feature_names]
        y = df[self.target_name]
        return X, y


    def train(self, X, y, test_size=0.2, random_state=42):
        self.logger.info("Training Random Forest model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'
        )


        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.last_training_time = datetime.now()
        train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        self.logger.info(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
        return train_acc, test_acc
        



    def prepare_features(self, X):
        """Prepare features for prediction (scaling, order)."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return X_scaled
        


    def get_model_info(self):
        info = {
            'model_type': 'RandomForestClassifier',
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'feature_count': len(self.feature_names),
            'class_names': self.class_names
        }
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            info['feature_importances'] = dict(zip(self.feature_names, self.model.feature_importances_))
        return info
        


    def save_model(self, model_dir='./artifacts/models'):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'rockburst_rf_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        meta = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'class_names': self.class_names,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.logger.info(f"Model and scaler saved to: {model_dir}")




    def load_model(self, model_dir='./artifacts/models'):
        model_path = os.path.join(model_dir, 'rockburst_rf_model.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        meta_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.feature_names = meta['feature_names']
                self.target_name = meta['target_name']
                self.class_names = {int(k): v for k, v in meta['class_names'].items()}
                if meta.get('last_training_time'):
                    self.last_training_time = datetime.fromisoformat(meta['last_training_time'])
            self.is_trained = True
            self.logger.info(f"Model loaded from: {model_dir}")
        else:
     
     
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.prepare_features(X)
        return self.model.predict(X_scaled)



    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.prepare_features(X)
        return self.model.predict_proba(X_scaled)


if __name__ == "__main__":
    # Example usage: train and save model
    model = RockburstRandomForestModel()
    X, y = model.load_data('data/preprocess/Rockburst_in_Tunnel_V3.csv')
    model.train(X, y)
    model.save_model()
    print("Random Forest model trained and saved successfully.")
