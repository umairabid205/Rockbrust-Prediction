#!/usr/bin/env python3
"""
Comprehensive Rockburst Prediction Model Training Script
========================================================
This script demonstrates the complete workflow for training rockburst prediction models
using the advanced ML pipeline developed in STEP 2.

Features:
- Loads actual user data from InfluxDB
- Applies comprehensive feature engineering
- Trains multiple ML models with evaluation
- Generates detailed reports and visualizations
- Integrates with MLflow for experiment tracking
- Saves all artifacts for production deployment

Author: Data Science Team GAMMA  
Created: August 10, 2025
"""

# Standard library imports
import sys
import os
import logging
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path

# Add project paths for imports
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')
sys.path.append('/opt/airflow/models')

# Data science and ML imports
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

# Custom logging setup
from exception_logging.logger import get_logger

# Import our custom ML pipeline modules
try:
    from models.rockburst_classifier import RockburstClassifier
    from models.feature_engineering import RockburstFeatureEngineering
    from models.model_trainer import RockburstModelTrainer
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODULES_AVAILABLE = False
    print(f"❌ Error importing custom modules: {str(e)}")
    print("Please ensure all models are in the /models directory")
    sys.exit(1)

# Database connection imports (conditional)
try:
    from influxdb_client import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("⚠️  InfluxDB client not available - using sample data instead")

class RockburstTrainingPipeline:
    """
    Complete training pipeline for rockburst prediction models.
    
    This class orchestrates the entire workflow from data loading
    to model deployment preparation.
    """
    
    def __init__(self, config_path=None, mlflow_enabled=True, logger_name="rockburst_training"):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file (optional)
            mlflow_enabled: Whether to use MLflow for experiment tracking
            logger_name: Name for the logger instance
        """
        self.logger = get_logger(logger_name)
        self.logger.info("🚀 Initializing Rockburst Training Pipeline")
        
        # MLflow configuration
        self.mlflow_enabled = mlflow_enabled
        if self.mlflow_enabled:
            mlflow.set_tracking_uri("http://localhost:5001")
            self.logger.info("📊 MLflow tracking configured")
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize pipeline components
        self.feature_engineer = None
        self.classifier = None  
        self.model_trainer = None
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.training_results = None
        
        # Output directories
        self._setup_output_directories()
        
    def _load_configuration(self, config_path=None):
        """Load training configuration from file or use defaults"""
        
        default_config = {
            'experiment': {
                'name': 'rockburst_production_training',
                'run_name': f'training_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'description': 'Production training of rockburst prediction models'
            },
            'data': {
                'source': 'influxdb',  # or 'csv_file' or 'sample'
                'influxdb': {
                    'url': 'http://localhost:8086',
                    'token': 'rockburst_token_2024',
                    'org': 'rockburst_org',
                    'bucket': 'rockburst_bucket',
                    'measurement': 'user_input_data',
                    'time_range': '-30d'  # Last 30 days
                },
                'csv_file': './data/processed/user_input_data.csv',
                'target_column': 'Intensity_Level_encoded',
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42
            },
            'feature_engineering': {
                'enabled': True,
                'geological_features': True,
                'temporal_features': True,
                'interaction_features': True,
                'polynomial_features': True,
                'feature_selection': {
                    'enabled': True,
                    'method': 'mutual_info',
                    'top_k_features': 50
                },
                'scaling': {
                    'enabled': True,
                    'method': 'robust'
                },
                'dimensionality_reduction': {
                    'enabled': True,
                    'method': 'pca',
                    'n_components': 0.95
                }
            },
            'model_training': {
                'models_to_train': [
                    'random_forest'
                ],
                'hyperparameter_tuning': {
                    'enabled': True,
                    'method': 'random_search',
                    'n_iterations': 50,
                    'cv_folds': 5,
                    'n_jobs': -1
                },
                'cross_validation': {
                    'enabled': True,
                    'folds': 5,
                    'scoring': 'f1_weighted'
                },
                'ensemble_methods': {
                    'voting_classifier': False,
                    'stacking_classifier': False,
                    'bagging_ensemble': False
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                }
            },
            'evaluation': {
                'metrics': [
                    'accuracy', 'precision', 'recall', 'f1_score',
                    'roc_auc', 'confusion_matrix', 'classification_report'
                ],
                'visualization': {
                    'enabled': True,
                    'plots': [
                        'confusion_matrix', 'roc_curves', 'precision_recall_curves',
                        'feature_importance', 'learning_curves', 'validation_curves'
                    ]
                },
                'reports': {
                    'detailed_report': True,
                    'model_comparison': True,
                    'feature_analysis': True,
                    'performance_summary': True
                }
            },
            'output': {
                'base_dir': './artifacts/rockburst_models',
                'models_dir': 'trained_models',
                'reports_dir': 'evaluation_reports', 
                'plots_dir': 'visualizations',
                'artifacts_dir': 'model_artifacts',
                'logs_dir': 'training_logs'
            },
            'deployment': {
                'model_registry': {
                    'enabled': True,
                    'model_name': 'rockburst_classifier',
                    'stage': 'Production'
                },
                'model_serving': {
                    'enabled': False,
                    'endpoint_name': 'rockburst_prediction_api'
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            self.logger.info(f"📄 Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations (user config overrides defaults)
            default_config.update(user_config)
        else:
            self.logger.info("⚙️  Using default configuration")
            
        return default_config
        
    def _setup_output_directories(self):
        """Create all necessary output directories"""
        
        base_dir = Path(self.config['output']['base_dir'])
        
        # Create all output directories
        for dir_name in ['models_dir', 'reports_dir', 'plots_dir', 'artifacts_dir', 'logs_dir']:
            dir_path = base_dir / self.config['output'][dir_name]
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {dir_path}")
            
    def load_data(self):
        """
        Load rockburst data from configured source.
        
        Returns:
            pd.DataFrame: Loaded rockburst data
        """
        self.logger.info("📥 Loading rockburst data...")
        
        data_source = self.config['data']['source']
        
        if data_source == 'influxdb' and INFLUXDB_AVAILABLE:
            self.raw_data = self._load_from_influxdb()
        elif data_source == 'csv_file':
            self.raw_data = self._load_from_csv()
        else:
            self.logger.warning("🔄 Using generated sample data")
            self.raw_data = self._generate_sample_data()
            
        self.logger.info(f"✅ Data loaded successfully: {self.raw_data.shape}")
        self._log_data_summary()
        
        return self.raw_data
        
    def _load_from_influxdb(self):
        """Load data from InfluxDB"""
        
        self.logger.info("🗄️  Loading data from InfluxDB...")
        
        influx_config = self.config['data']['influxdb']
        
        try:
            # Connect to InfluxDB
            client = InfluxDBClient(
                url=influx_config['url'],
                token=influx_config['token'],
                org=influx_config['org']
            )
            
            # Build query
            query = f'''
            from(bucket: "{influx_config['bucket']}")
                |> range(start: {influx_config['time_range']})
                |> filter(fn: (r) => r._measurement == "{influx_config['measurement']}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            # Execute query
            query_api = client.query_api()
            result = query_api.query_data_frame(query)
            
            client.close()
            
            if result.empty:
                self.logger.warning("⚠️  No data found in InfluxDB, generating sample data")
                return self._generate_sample_data()
                
            # Clean up the DataFrame
            result = result.drop(columns=['result', 'table', '_start', '_stop', '_measurement'], errors='ignore')
            result = result.reset_index(drop=True)
            
            self.logger.info(f"✅ Loaded {len(result)} records from InfluxDB")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error loading from InfluxDB: {str(e)}")
            self.logger.warning("🔄 Falling back to sample data")
            return self._generate_sample_data()
            
    def _load_from_csv(self):
        """Load data from CSV file"""
        
        csv_path = self.config['data']['csv_file']
        self.logger.info(f"📄 Loading data from CSV: {csv_path}")
        
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                self.logger.info(f"✅ Loaded {len(df)} records from CSV")
                return df
            else:
                self.logger.warning(f"⚠️  CSV file not found: {csv_path}")
                return self._generate_sample_data()
                
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV: {str(e)}")
            return self._generate_sample_data()
            
    def _generate_sample_data(self, n_samples=2000):
        """Generate realistic sample rockburst data"""
        
        self.logger.info(f"🎲 Generating {n_samples} sample data points...")
        
        np.random.seed(self.config['data']['random_state'])
        
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
        
        self.logger.info("✅ Sample data generated successfully")
        return df
        
    def _log_data_summary(self):
        """Log comprehensive data summary"""
        
        if self.raw_data is None:
            return
            
        self.logger.info("📊 Data Summary:")
        self.logger.info(f"   Shape: {self.raw_data.shape}")
        self.logger.info(f"   Features: {list(self.raw_data.columns)}")
        
        # Target distribution
        if 'Intensity_Level_encoded' in self.raw_data.columns:
            target_dist = self.raw_data['Intensity_Level_encoded'].value_counts().sort_index()
            self.logger.info("   Target Distribution:")
            for level, count in target_dist.items():
                percentage = (count / len(self.raw_data)) * 100
                intensity_name = ['Low', 'Medium', 'High'][level]
                self.logger.info(f"     {intensity_name} intensity (class {level}): {count} samples ({percentage:.1f}%)")
                
        # Missing values
        missing_values = self.raw_data.isnull().sum()
        if missing_values.any():
            self.logger.warning("⚠️  Missing values detected:")
            for col, missing in missing_values.items():
                if missing > 0:
                    self.logger.warning(f"     {col}: {missing} missing values")
                    
    def prepare_features(self):
        """Apply comprehensive feature engineering"""
        
        self.logger.info("🔧 Starting feature engineering pipeline...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Initialize feature engineer
        self.feature_engineer = RockburstFeatureEngineering()
        
        # Apply feature engineering based on configuration
        feature_config = self.config['feature_engineering']
        target_column = self.config['data']['target_column']
        
        # Create comprehensive features
        if feature_config['enabled']:
            self.processed_data = self.feature_engineer.create_comprehensive_features(
                self.raw_data, 
                target_column
            )
        else:
            self.processed_data = self.raw_data.copy()
            
        self.logger.info(f"✅ Feature engineering completed")
        self.logger.info(f"   Features expanded from {self.raw_data.shape[1]} to {self.processed_data.shape[1]}")
        
        return self.processed_data
        
    def train_models(self):
        """Train all configured models"""
        
        self.logger.info("🚀 Starting model training pipeline...")
        
        if self.processed_data is None:
            raise ValueError("No processed data available. Call prepare_features() first.")
            
        # Initialize model trainer
        self.model_trainer = RockburstModelTrainer(mlflow_enabled=self.mlflow_enabled)
        
        # Update model trainer paths to use our configuration
        self.model_trainer.config.update(self.config)
        
        # Override the hardcoded paths in model trainer
        base_dir = self.config['output']['base_dir']
        self.model_trainer.data_paths.update({
            'processed_data': f'{base_dir}/processed_data',
            'features_data': f'{base_dir}/features_data', 
            'final_data': f'{base_dir}/final_data',
            'models_output': f'{base_dir}/trained_models',
            'artifacts_output': f'{base_dir}/model_artifacts',
            'reports_output': f'{base_dir}/evaluation_reports'
        })
        
        self.model_trainer.output_paths.update({
            'models_dir': f'{base_dir}/trained_models',
            'artifacts_dir': f'{base_dir}/model_artifacts',
            'reports_dir': f'{base_dir}/evaluation_reports',
            'plots_dir': f'{base_dir}/visualizations'
        })
        
        # Set data
        self.model_trainer.training_data = self.processed_data
        
        # Start MLflow experiment if enabled
        if self.mlflow_enabled:
            mlflow.set_experiment(self.config['experiment']['name'])
            
        with mlflow.start_run(run_name=self.config['experiment']['run_name']) if self.mlflow_enabled else nullcontext():
            
            # Log experiment parameters
            if self.mlflow_enabled:
                mlflow.log_params({
                    'data_source': self.config['data']['source'],
                    'n_samples': len(self.processed_data),
                    'n_features': self.processed_data.shape[1] - 1,  # Exclude target
                    'feature_engineering_enabled': self.config['feature_engineering']['enabled'],
                    'models_to_train': self.config['model_training']['models_to_train'],
                    'hyperparameter_tuning_enabled': self.config['model_training']['hyperparameter_tuning']['enabled']
                })
            
            # Prepare data for training
            processed_train, processed_val = self.model_trainer.prepare_features(self.processed_data)
            
            # Train models
            self.training_results = self.model_trainer.train_models(processed_train)
            
            # Evaluate models
            evaluation_results = self.model_trainer.evaluate_models(save_reports=True)
            
            # Save artifacts
            artifacts_info = self.model_trainer.save_artifacts()
            
            # Log best model metrics to MLflow
            if self.mlflow_enabled and 'best_model' in self.training_results:
                best_model = self.training_results['best_model']
                mlflow.log_metrics({
                    'best_model_f1_score': best_model['f1_score'],
                    'best_model_accuracy': best_model['metrics'].get('accuracy', 0),
                    'best_model_precision': best_model['metrics'].get('precision_weighted', 0),
                    'best_model_recall': best_model['metrics'].get('recall_weighted', 0)
                })
                
                # Register best model
                if self.config['deployment']['model_registry']['enabled']:
                    model_name = self.config['deployment']['model_registry']['model_name']
                    mlflow.sklearn.log_model(
                        best_model['trained_model'],
                        'best_model',
                        registered_model_name=model_name
                    )
                    
        self.logger.info("✅ Model training completed successfully")
        return self.training_results
        
    def generate_final_report(self):
        """Generate comprehensive final training report"""
        
        self.logger.info("📋 Generating final training report...")
        
        if self.training_results is None:
            raise ValueError("No training results available. Call train_models() first.")
            
        # Prepare report data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'training_summary': {
                'timestamp': timestamp,
                'experiment_name': self.config['experiment']['name'],
                'data_source': self.config['data']['source'],
                'total_samples': len(self.raw_data) if self.raw_data is not None else 0,
                'total_features': self.processed_data.shape[1] - 1 if self.processed_data is not None else 0,
                'models_trained': len(self.training_results.get('training_summary', {}).get('successful_models', [])),
                'training_duration': self.training_results.get('training_summary', {}).get('total_training_time', 0)
            },
            'best_model_performance': self.training_results.get('best_model', {}),
            'all_models_performance': [],
            'feature_engineering_summary': {
                'original_features': self.raw_data.shape[1] if self.raw_data is not None else 0,
                'engineered_features': self.processed_data.shape[1] if self.processed_data is not None else 0,
                'feature_selection_enabled': self.config['feature_engineering']['feature_selection']['enabled'],
                'scaling_method': self.config['feature_engineering']['scaling']['method']
            },
            'data_quality_summary': {
                'missing_values': self.raw_data.isnull().sum().sum() if self.raw_data is not None else 0,
                'target_distribution': self.raw_data['Intensity_Level_encoded'].value_counts().to_dict() if self.raw_data is not None else {}
            },
            'configuration': self.config
        }
        
        # Add all models performance
        if 'model_results' in self.training_results:
            for model_name, model_info in self.training_results['model_results'].items():
                if 'evaluation_metrics' in model_info:
                    model_perf = {
                        'model_name': model_name,
                        'f1_score': model_info['evaluation_metrics'].get('f1_weighted', 0),
                        'accuracy': model_info['evaluation_metrics'].get('accuracy', 0),
                        'precision': model_info['evaluation_metrics'].get('precision_weighted', 0),
                        'recall': model_info['evaluation_metrics'].get('recall_weighted', 0),
                        'training_time': model_info.get('training_time', 0)
                    }
                    report['all_models_performance'].append(model_perf)
                    
        # Save report
        reports_dir = Path(self.config['output']['base_dir']) / self.config['output']['reports_dir']
        report_filename = f"final_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"📄 Final report saved to: {report_path}")
        
        # Print summary
        self._print_training_summary(report)
        
        return report_path
        
    def _print_training_summary(self, report):
        """Print a formatted training summary"""
        
        print("\n" + "="*80)
        print("🎯 ROCKBURST PREDICTION MODEL TRAINING SUMMARY")
        print("="*80)
        
        summary = report['training_summary']
        print(f"🕒 Training completed at: {summary['timestamp']}")
        print(f"📊 Data samples: {summary['total_samples']:,}")
        print(f"🔧 Features engineered: {summary['total_features']:,}")
        print(f"🤖 Models trained: {summary['models_trained']}")
        print(f"⏱️  Training duration: {summary['training_duration']:.2f} seconds")
        
        # Best model performance
        if 'best_model' in self.training_results:
            best_model = self.training_results['best_model']
            print(f"\n🏆 BEST MODEL: {best_model['name']}")
            print(f"   F1-Score: {best_model['f1_score']:.4f}")
            print(f"   Accuracy: {best_model['metrics'].get('accuracy', 0):.4f}")
            print(f"   Precision: {best_model['metrics'].get('precision_weighted', 0):.4f}")
            print(f"   Recall: {best_model['metrics'].get('recall_weighted', 0):.4f}")
            
        # Model comparison
        if report['all_models_performance']:
            print(f"\n📈 ALL MODELS PERFORMANCE:")
            print(f"{'Model':<20} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
            print("-" * 70)
            for model_perf in sorted(report['all_models_performance'], 
                                   key=lambda x: x['f1_score'], reverse=True):
                print(f"{model_perf['model_name']:<20} "
                      f"{model_perf['f1_score']:<10.4f} "
                      f"{model_perf['accuracy']:<10.4f} "
                      f"{model_perf['precision']:<10.4f} "
                      f"{model_perf['recall']:<10.4f}")
                      
        print(f"\n📁 Artifacts saved to: {self.config['output']['base_dir']}")
        print(f"🌐 MLflow UI: http://localhost:5001")
        print("="*80)


# Context manager for MLflow run handling
from contextlib import nullcontext


def main():
    """Main training script execution"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train rockburst prediction models')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    parser.add_argument('--data-source', choices=['influxdb', 'csv_file', 'sample'], 
                       help='Data source override')
    parser.add_argument('--models', nargs='+', help='Specific models to train (default: random_forest only)',
                       choices=['logistic_regression', 'random_forest', 'gradient_boosting', 
                               'xgboost', 'lightgbm', 'svm', 'mlp', 'extra_trees'],
                       default=['random_forest'])
    parser.add_argument('--quick', action='store_true', 
                       help='Quick training mode (fewer models, no hyperparameter tuning)')
    
    args = parser.parse_args()
    
    try:
        # Check if custom modules are available
        if not CUSTOM_MODULES_AVAILABLE:
            print("❌ Custom ML modules are not available. Please check the models directory.")
            sys.exit(1)
            
        # Initialize pipeline
        pipeline = RockburstTrainingPipeline(
            config_path=args.config,
            mlflow_enabled=not args.no_mlflow
        )
        
        # Apply command line overrides
        if args.data_source:
            pipeline.config['data']['source'] = args.data_source
            
        if args.models:
            pipeline.config['model_training']['models_to_train'] = args.models
            
        if args.quick:
            pipeline.config['model_training']['models_to_train'] = ['random_forest']
            pipeline.config['model_training']['hyperparameter_tuning']['enabled'] = False
            pipeline.config['model_training']['ensemble_methods'] = {
                'voting_classifier': False,
                'stacking_classifier': False,
                'bagging_ensemble': False
            }
        
        print("🚀 Starting Rockburst Prediction Model Training Pipeline")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute pipeline steps
        print("\n" + "="*60)
        print("STEP 1: Data Loading")
        print("="*60)
        pipeline.load_data()
        
        print("\n" + "="*60)
        print("STEP 2: Feature Engineering")
        print("="*60)
        pipeline.prepare_features()
        
        print("\n" + "="*60)
        print("STEP 3: Model Training & Evaluation")
        print("="*60)
        pipeline.train_models()
        
        print("\n" + "="*60)
        print("STEP 4: Final Report Generation")
        print("="*60)
        report_path = pipeline.generate_final_report()
        
        print("\n✅ Training pipeline completed successfully!")
        print(f"📄 Full report available at: {report_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
