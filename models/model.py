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

# Add project root to path
sys.path.append('.')
sys.path.append('./models')

from feature_engineering import RockburstFeatureEngineering
from exception_logging.logger import get_logger


class RockburstRandomForestModel:
    """
    Random Forest Model for Rockburst Prediction
    
    This class encapsulates the Random Forest model with optimized parameters
    for rockburst intensity classification (Low, Medium, High).
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Random Forest model.
        
        Args:
            config_path: Path to model configuration file
        """
        self.logger = get_logger("rockburst_random_forest")
        self.model = None
        self.feature_engineer = None
        self.scaler = None
        self.is_trained = False
        self.last_training_time = None
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model with configuration
        self._initialize_model()
        
    def _load_config(self, config_path=None):
        """Load model configuration"""
        default_config = {
            'model_params': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42,
                'retrain_interval_hours': 24,
                'min_samples_for_training': 100
            },
            'features': {
                'use_feature_engineering': True,
                'scale_features': True,
                'feature_selection': False,
                'n_top_features': 50
            },
            'model_persistence': {
                'model_dir': './artifacts/models',
                'model_filename': 'rockburst_rf_model.pkl',
                'scaler_filename': 'feature_scaler.pkl',
                'feature_engineer_filename': 'feature_engineer.pkl',
                'config_filename': 'model_config.json',
                'training_log_filename': 'training_log.json'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Update default config with user config
            default_config.update(user_config)
            
        return default_config
        
    def _initialize_model(self):
        """Initialize the Random Forest model with configured parameters"""
        self.model = RandomForestClassifier(**self.config['model_params'])
        self.feature_engineer = RockburstFeatureEngineering()
        
        if self.config['features']['scale_features']:
            self.scaler = StandardScaler()
            
        self.logger.info("Random Forest model initialized successfully")
        
    def needs_retraining(self):
        """
        Check if model needs retraining based on 24-hour interval.
        
        Returns:
            bool: True if model needs retraining
        """
        if not self.is_trained or self.last_training_time is None:
            return True
            
        time_since_training = datetime.now() - self.last_training_time
        retrain_interval = timedelta(hours=self.config['training']['retrain_interval_hours'])
        
        return time_since_training >= retrain_interval
        
    def prepare_features(self, df, target_column='Intensity_Level_encoded'):
        """
        Prepare features for training or prediction.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            X: Feature matrix
            y: Target vector (if target_column exists)
        """
        # Apply feature engineering if enabled
        if self.config['features']['use_feature_engineering']:
            df_processed = self.feature_engineer.create_comprehensive_features(df, target_column)
        else:
            df_processed = df.copy()
            
        # Separate features and target
        if target_column in df_processed.columns:
            X = df_processed.drop(target_column, axis=1)
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
            
        # Apply scaling if enabled
        if self.config['features']['scale_features'] and self.scaler is not None:
            if self.is_trained:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                
        return X, y
        
    def get_model_info(self):
        """
        Get detailed model information.
        
        Returns:
            dict: Model information
        """
        info = {
            'model_type': 'RandomForestClassifier',
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'needs_retraining': self.needs_retraining(),
            'config': self.config,
            'feature_count': None,
            'class_names': ['Low', 'Medium', 'High']
        }
        
        if self.is_trained and hasattr(self.model, 'n_features_in_'):
            info['feature_count'] = self.model.n_features_in_
            info['feature_importances'] = self.model.feature_importances_.tolist()
            
        return info
        
    def save_model(self, model_dir=None):
        """
        Save the trained model and associated components.
        
        Args:
            model_dir: Directory to save model (optional)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        if model_dir is None:
            model_dir = self.config['model_persistence']['model_dir']
            
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, self.config['model_persistence']['model_filename'])
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to: {model_path}")
        
        # Save feature engineer
        if self.feature_engineer is not None:
            fe_path = os.path.join(model_dir, self.config['model_persistence']['feature_engineer_filename'])
            joblib.dump(self.feature_engineer, fe_path)
            self.logger.info(f"Feature engineer saved to: {fe_path}")
            
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(model_dir, self.config['model_persistence']['scaler_filename'])
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Scaler saved to: {scaler_path}")
            
        # Save configuration
        config_path = os.path.join(model_dir, self.config['model_persistence']['config_filename'])
        with open(config_path, 'w') as f:
            config_to_save = self.config.copy()
            config_to_save['last_training_time'] = self.last_training_time.isoformat() if self.last_training_time else None
            json.dump(config_to_save, f, indent=2)
            
        # Save training log
        log_path = os.path.join(model_dir, self.config['model_persistence']['training_log_filename'])
        training_log = {
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_version': '1.0',
            'training_completed': True
        }
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
            
        return model_path
        
    def load_model(self, model_dir=None):
        """
        Load a previously trained model.
        
        Args:
            model_dir: Directory containing the saved model
        """
        if model_dir is None:
            model_dir = self.config['model_persistence']['model_dir']
            
        # Load model
        model_path = os.path.join(model_dir, self.config['model_persistence']['model_filename'])
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load feature engineer
        fe_path = os.path.join(model_dir, self.config['model_persistence']['feature_engineer_filename'])
        if os.path.exists(fe_path):
            self.feature_engineer = joblib.load(fe_path)
            self.logger.info(f"Feature engineer loaded from: {fe_path}")
            
        # Load scaler
        scaler_path = os.path.join(model_dir, self.config['model_persistence']['scaler_filename'])
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Scaler loaded from: {scaler_path}")
            
        # Load training timestamp from log
        log_path = os.path.join(model_dir, self.config['model_persistence']['training_log_filename'])
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                training_log = json.load(f)
            if training_log.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(training_log['last_training_time'])
                
        self.is_trained = True
        return model_path
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Array of predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            probabilities: Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict_proba(X)


def create_model(config_path=None):
    """
    Factory function to create a RockburstRandomForestModel instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RockburstRandomForestModel: Configured model instance
    """
    return RockburstRandomForestModel(config_path)


if __name__ == "__main__":
    # Example usage
    model = create_model()
    print("Random Forest model created successfully")
    print(f"Model info: {model.get_model_info()}")
