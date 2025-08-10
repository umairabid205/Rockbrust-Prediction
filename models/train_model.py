"""
Model Training Script for Rockburst Prediction
===============================================
This module handles training of the Random Forest model for rockburst prediction
with automatic retraining capabilities every 24 hours.

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Add project root to path
sys.path.append('.')
sys.path.append('./models')

from model import RockburstRandomForestModel
from exception_logging.logger import get_logger

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelTrainer:
    """
    Handles training and retraining of the rockburst prediction model.
    """
    
    def __init__(self, model_dir='./artifacts/models'):
        """
        Initialize the trainer.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.logger = get_logger("model_trainer")
        self.model_dir = model_dir
        self.model = None
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def generate_sample_data(self, n_samples=2000):
        """
        Generate sample rockburst data for training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        self.logger.info(f"Generating {n_samples} sample data points...")
        
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
        risk_score = np.clip(risk_score, 0, 1)
        
        intensity_labels = []
        for score in risk_score:
            if score <= 0.35:
                intensity_labels.append(0)  # Low
            elif score <= 0.70:
                intensity_labels.append(1)  # Medium
            else:
                intensity_labels.append(2)  # High
                
        df['Intensity_Level_encoded'] = intensity_labels
        
        self.logger.info("✅ Sample data generated successfully")
        return df
    
    def load_training_data(self, data_path=None):
        """
        Load training data from file or generate sample data.
        
        Args:
            data_path: Path to training data CSV file
            
        Returns:
            pd.DataFrame: Training data
        """
        if data_path and os.path.exists(data_path):
            self.logger.info(f"Loading training data from: {data_path}")
            df = pd.read_csv(data_path)
        else:
            self.logger.info("No data file provided, generating sample data")
            df = self.generate_sample_data()
            
        return df
    
    def train_model(self, training_data, target_column='Intensity_Level_encoded'):
        """
        Train the Random Forest model.
        
        Args:
            training_data: DataFrame with training data
            target_column: Name of the target column
            
        Returns:
            dict: Training results and metrics
        """
        self.logger.info("🚀 Starting Random Forest model training...")
        
        # Initialize model
        self.model = RockburstRandomForestModel()
        
        # Prepare features
        X, y = self.model.prepare_features(training_data, target_column)
        
        self.logger.info(f"📊 Training data prepared:")
        self.logger.info(f"   Samples: {len(X)}")
        self.logger.info(f"   Features: {X.shape[1]}")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model.config['training']['test_size'],
            random_state=self.model.config['training']['random_state'],
            stratify=y
        )
        
        # Train the model
        self.logger.info("🌲 Training Random Forest...")
        start_time = datetime.now()
        
        self.model.model.fit(X_train, y_train)
        
        training_time = datetime.now() - start_time
        self.model.is_trained = True
        self.model.last_training_time = datetime.now()
        
        self.logger.info(f"✅ Training completed in {training_time.total_seconds():.2f} seconds")
        
        # Evaluate on test set
        y_pred = self.model.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Get feature importance
        feature_importance = list(zip(X.columns, self.model.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        training_results = {
            'training_time_seconds': training_time.total_seconds(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': X.shape[1],
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'top_10_features': feature_importance[:10],
            'last_training_time': self.model.last_training_time.isoformat(),
            'model_parameters': self.model.config['model_params']
        }
        
        self.logger.info(f"📊 Model Performance:")
        self.logger.info(f"   Accuracy: {accuracy:.4f}")
        self.logger.info(f"   F1-Score: {f1:.4f}")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall: {recall:.4f}")
        
        return training_results
    
    def save_trained_model(self):
        """
        Save the trained model to disk.
        
        Returns:
            str: Path where model was saved
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("No trained model to save")
            
        model_path = self.model.save_model(self.model_dir)
        self.logger.info(f"💾 Model saved successfully to: {model_path}")
        return model_path
    
    def load_existing_model(self):
        """
        Load an existing model from disk if available.
        
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            self.model = RockburstRandomForestModel()
            self.model.load_model(self.model_dir)
            self.logger.info("✅ Existing model loaded successfully")
            return True
        except (FileNotFoundError, Exception) as e:
            self.logger.warning(f"Could not load existing model: {str(e)}")
            return False
    
    def retrain_if_needed(self, training_data, use_mlflow=False):
        """
        Check if retraining is needed (based on 24-hour rule) and retrain if necessary.
        
        Args:
            training_data: Training dataset as pandas DataFrame
            use_mlflow: Whether to integrate with MLflow for experiment tracking
            
        Returns:
            dict: Training results if retraining occurred, None if not needed
        """
        last_training_time = self._get_last_training_time()
        current_time = datetime.now()
        
        # Check if retraining is needed (24-hour rule)
        if last_training_time is None:
            self.logger.info("🔄 No previous training found - training new model")
            needs_retraining = True
        else:
            time_since_training = current_time - last_training_time
            needs_retraining = time_since_training > timedelta(hours=24)
            
            if needs_retraining:
                self.logger.info(f"🔄 Retraining needed - {time_since_training} since last training")
            else:
                self.logger.info(f"✅ Model is recent - {time_since_training} since last training")
        
        if needs_retraining:
            # Train new model
            training_results = self.train_model(training_data)
            if training_results:
                self.save_trained_model()
                return training_results
            else:
                self.logger.error("❌ Training failed")
                return None
        else:
            return None
    
    def _get_last_training_time(self):
        """
        Get the timestamp of the last training session.
        
        Returns:
            datetime or None: Last training time
        """
        if self.model and hasattr(self.model, 'last_training_time'):
            return self.model.last_training_time
        
        # Check for model config file
        config_path = os.path.join(self.model_dir, 'model_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    training_time_str = config.get('last_training_time')
                    if training_time_str:
                        return datetime.fromisoformat(training_time_str)
            except Exception as e:
                self.logger.warning(f"Could not read training time from config: {str(e)}")
        
        # Check file modification time of model file
        model_path = os.path.join(self.model_dir, 'rockburst_rf_model.pkl')
        if os.path.exists(model_path):
            mtime = os.path.getmtime(model_path)
            return datetime.fromtimestamp(mtime)
        
        return None
    
    def get_model_status(self):
        """
        Get current model status and information.
        
        Returns:
            dict: Model status information
        """
        if self.model is None:
            return {
                'model_exists': False,
                'is_trained': False,
                'needs_retraining': True,
                'last_training_time': None
            }
        
        return {
            'model_exists': True,
            'is_trained': self.model.is_trained,
            'needs_retraining': self.model.needs_retraining(),
            'last_training_time': self.model.last_training_time.isoformat() if self.model.last_training_time else None,
            'model_info': self.model.get_model_info()
        }


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest model for rockburst prediction')
    parser.add_argument('--data-path', type=str, help='Path to training data CSV file')
    parser.add_argument('--model-dir', type=str, default='./artifacts/models', 
                       help='Directory to save/load models')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force retraining even if model is recent')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of sample data points to generate (if no data file)')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.model_dir)
        
        # Load training data
        training_data = trainer.load_training_data(args.data_path)
        
        print("🚀 Rockburst Model Training Script")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if args.force_retrain:
            # Force retrain
            print("🔄 Force retraining model...")
            training_results = trainer.train_model(training_data)
            trainer.save_trained_model()
        else:
            # Smart retraining (only if needed)
            training_results = trainer.retrain_if_needed(training_data)
        
        # Get model status
        status = trainer.get_model_status()
        
        print("="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Model exists: {status['model_exists']}")
        print(f"Is trained: {status['is_trained']}")
        print(f"Needs retraining: {status['needs_retraining']}")
        print(f"Last training: {status.get('last_training_time', 'Never')}")
        
        if training_results:
            print(f"\\n🎯 Training Results:")
            print(f"   Accuracy: {training_results['accuracy']:.4f}")
            print(f"   F1-Score: {training_results['f1_score']:.4f}")
            print(f"   Training time: {training_results['training_time_seconds']:.2f}s")
            print(f"   Training samples: {training_results['training_samples']}")
            print(f"   Features used: {training_results['total_features']}")
        
        print("="*60)
        print("✅ Training script completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
