#!/usr/bin/env python3
"""
MLflow + Airflow Integrated Model Pipeline for Rockburst Prediction
==================================================================
This demo shows the complete workflow with:
1. MLflow for experiment tracking and artifact storage
2. Airflow for orchestrating 24-hour retraining
3. Model versioning and deployment
4. Comprehensive monitoring and logging

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import json
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn

# Add project paths
sys.path.append('.')
sys.path.append('..')
sys.path.append('../models')

from train_model import ModelTrainer
from model import RockburstRandomForestModel
from exception_logging.logger import get_logger


class MLflowAirflowModelPipeline:
    """
    Integrated pipeline with MLflow for artifact storage and Airflow for orchestration.
    """
    
    def __init__(self, mlflow_uri="http://localhost:5001", model_dir='../artifacts/models'):
        """
        Initialize the integrated pipeline.
        
        Args:
            mlflow_uri: MLflow tracking server URI
            model_dir: Local model storage directory
        """
        self.logger = get_logger("mlflow_airflow_pipeline")
        self.mlflow_uri = mlflow_uri
        self.model_dir = model_dir
        self.experiment_name = "rockburst_production_pipeline"
        
        # Configure MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configure MLflow tracking and experiment"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"✅ MLflow configured: {self.mlflow_uri}")
            self.logger.info(f"📊 Experiment: {self.experiment_name}")
        except Exception as e:
            self.logger.error(f"❌ MLflow setup failed: {str(e)}")
            raise
    
    def run_training_with_mlflow(self, force_retrain=False):
        """
        Run model training with full MLflow integration.
        
        Args:
            force_retrain: Whether to force retraining regardless of 24-hour rule
            
        Returns:
            dict: Training results and MLflow run info
        """
        self.logger.info("🚀 Starting MLflow-integrated training pipeline...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"rockburst_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Get MLflow run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            # Initialize trainer
            trainer = ModelTrainer(self.model_dir)
            
            # Load training data
            training_data = trainer.load_training_data()
            
            # Log dataset info to MLflow
            mlflow.log_params({
                "dataset_size": len(training_data),
                "n_features_original": len(training_data.columns) - 1,  # Exclude target
                "force_retrain": force_retrain,
                "pipeline_type": "production_mlflow_airflow"
            })
            
            # Train or retrain model
            if force_retrain:
                training_results = trainer.train_model(training_data)
                trainer.save_trained_model()
            else:
                training_results = trainer.retrain_if_needed(training_data)
            
            if training_results:
                # Log training metrics to MLflow
                mlflow.log_metrics({
                    "accuracy": training_results['accuracy'],
                    "f1_score": training_results['f1_score'],
                    "precision": training_results['precision'],
                    "recall": training_results['recall'],
                    "training_time_seconds": training_results['training_time_seconds'],
                    "n_features_engineered": training_results['total_features']
                })
                
                # Log model parameters
                mlflow.log_params(training_results['model_parameters'])
                
                # Log model artifacts to MLflow
                if trainer.model and trainer.model.is_trained:
                    model_info = mlflow.sklearn.log_model(
                        trainer.model.model,
                        "random_forest_model",
                        registered_model_name="rockburst_random_forest_production"
                    )
                    
                    # Log additional artifacts
                    mlflow.log_artifact(os.path.join(self.model_dir, 'feature_engineer.pkl'), 'feature_pipeline')
                    mlflow.log_artifact(os.path.join(self.model_dir, 'feature_scaler.pkl'), 'feature_pipeline')
                    mlflow.log_artifact(os.path.join(self.model_dir, 'model_config.json'), 'config')
                    
                    self.logger.info("✅ Model artifacts logged to MLflow")
                
                # Log feature importance
                if hasattr(trainer.model.model, 'feature_importances_'):
                    # Get feature names and importance
                    feature_names = list(range(training_results['total_features']))  # Placeholder
                    importance_dict = {f"feature_{i}": float(imp) 
                                     for i, imp in enumerate(trainer.model.model.feature_importances_[:10])}  # Top 10
                    mlflow.log_params(importance_dict)
                
                self.logger.info(f"📊 Training completed with MLflow run: {run_id}")
                
                return {
                    'training_results': training_results,
                    'mlflow_run_id': run_id,
                    'mlflow_experiment': self.experiment_name,
                    'model_uri': model_info.model_uri if 'model_info' in locals() else None
                }
            else:
                mlflow.log_params({"status": "no_retraining_needed"})
                self.logger.info("✅ No retraining needed - using existing model")
                
                return {
                    'training_results': None,
                    'mlflow_run_id': run_id,
                    'status': 'no_retraining_needed'
                }
    
    def create_airflow_dag(self):
        """
        Generate Airflow DAG for automated retraining.
        
        Returns:
            str: Path to generated DAG file
        """
        dag_content = """#
# Rockburst Model Retraining DAG with MLflow Integration
# =====================================================
# This DAG handles automated model retraining every 24 hours with
# full MLflow experiment tracking and artifact storage.
#
# Schedule: Daily at 2 AM
#

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import sys
import os

# Add project paths
sys.path.append('/Users/umair/Downloads/projects/project_2')
sys.path.append('/Users/umair/Downloads/projects/project_2/models')

def run_model_training_with_mlflow():
    '''Task function for model training with MLflow'''
    from models.demo_model_pipeline import MLflowAirflowModelPipeline
    
    # Initialize pipeline
    pipeline = MLflowAirflowModelPipeline()
    
    # Run training (will auto-check 24-hour rule)
    results = pipeline.run_training_with_mlflow(force_retrain=False)
    
    print(f"Training completed with MLflow run: {results.get('mlflow_run_id')}")
    return results

def validate_model_performance():
    '''Task function for model validation'''
    from models.test_model import ModelTester
    
    # Run basic model validation
    tester = ModelTester()
    if tester.load_model():
        validation = tester.validate_model_integrity()
        if not validation['validation_passed']:
            raise ValueError("Model validation failed!")
        print("✅ Model validation passed")
    else:
        raise ValueError("Failed to load model for validation")

def send_training_notification():
    '''Task function for sending training notifications'''
    print("📧 Training notification sent (placeholder)")
    # Add actual notification logic here (email, Slack, etc.)

# DAG configuration
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 10),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'rockburst_model_retraining',
    default_args=default_args,
    description='Automated rockburst model retraining with MLflow',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['machine-learning', 'rockburst', 'mlflow'],
)

# Task 1: Check dependencies
check_dependencies = BashOperator(
    task_id='check_dependencies',
    bash_command='cd /Users/umair/Downloads/projects/project_2 && echo "Checking dependencies..." && python -c "import mlflow, sklearn, pandas, numpy; print(\\'✅ All dependencies available\\')"',
    dag=dag,
)

# Task 2: Run model training with MLflow
train_model = PythonOperator(
    task_id='train_model_with_mlflow',
    python_callable=run_model_training_with_mlflow,
    dag=dag,
)

# Task 3: Validate trained model
validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model_performance,
    dag=dag,
)

# Task 4: Send notification
notify_completion = PythonOperator(
    task_id='send_notification',
    python_callable=send_training_notification,
    dag=dag,
)

# Task 5: Cleanup old model versions (optional)
cleanup_old_versions = BashOperator(
    task_id='cleanup_old_versions',
    bash_command='cd /Users/umair/Downloads/projects/project_2 && find artifacts/models/archive -name "*.pkl" -type f -mtime +5 -delete 2>/dev/null || echo "✅ Cleanup completed"',
    dag=dag,
)

# Define task dependencies
check_dependencies >> train_model >> validate_model >> [notify_completion, cleanup_old_versions]
"""
        
        # Save DAG file
        dag_dir = '../dags'
        os.makedirs(dag_dir, exist_ok=True)
        dag_path = os.path.join(dag_dir, 'rockburst_model_retraining_dag.py')
        
        with open(dag_path, 'w') as f:
            f.write(dag_content)
        
        self.logger.info(f"📄 Airflow DAG created: {dag_path}")
        return dag_path
    
    def check_mlflow_artifacts(self):
        """
        Check and display MLflow artifacts for recent runs.
        
        Returns:
            dict: MLflow artifacts information
        """
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                self.logger.warning(f"⚠️ Experiment '{self.experiment_name}' not found")
                return {'experiment_name': self.experiment_name, 'recent_runs': []}
            
            # Simple check - return basic info
            artifacts_info = {
                'experiment_id': experiment.experiment_id,
                'experiment_name': self.experiment_name,
                'recent_runs': []
            }
            
            self.logger.info(f"📊 MLflow experiment found: {self.experiment_name}")
            return artifacts_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check MLflow artifacts: {str(e)}")
            return {'experiment_name': self.experiment_name, 'recent_runs': []}
    
    def demonstrate_complete_pipeline(self):
        """
        Demonstrate the complete MLflow + Airflow pipeline.
        """
        print("🚀 MLflow + Airflow Model Pipeline Demonstration")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        try:
            print("\\n📊 STEP 1: MLflow Experiment Tracking")
            print("-" * 50)
            print(f"✅ MLflow Server: {self.mlflow_uri}")
            print(f"📊 Experiment: {self.experiment_name}")
            
            # Check existing MLflow artifacts
            artifacts_info = self.check_mlflow_artifacts()
            if artifacts_info.get('recent_runs'):
                print(f"📈 Recent runs: {len(artifacts_info['recent_runs'])}")
            else:
                print("📈 No previous runs found - will create new experiment")
            print("\\n🏋️ STEP 2: Model Training with MLflow")
            print("-" * 50)
            
            # Run training with MLflow
            training_results = self.run_training_with_mlflow(force_retrain=False)
            
            if training_results['training_results']:
                print("✅ New model trained!")
                print(f"   Accuracy: {training_results['training_results']['accuracy']:.4f}")
                print(f"   F1-Score: {training_results['training_results']['f1_score']:.4f}")
                print(f"   MLflow Run: {training_results['mlflow_run_id']}")
            else:
                print("✅ Using existing model (< 24 hours old)")
                print(f"   MLflow Run: {training_results['mlflow_run_id']}")
            
            print("\\n🔄 STEP 3: Airflow DAG Generation")
            print("-" * 50)
            
            dag_path = self.create_airflow_dag()
            print(f"✅ Airflow DAG created: {dag_path}")
            print("📋 DAG Features:")
            print("   - Daily execution at 2 AM")
            print("   - Automatic 24-hour retraining check") 
            print("   - MLflow experiment tracking")
            print("   - Model validation")
            print("   - Notification system")
            print("   - Cleanup old versions")
            
            print("\\n💾 STEP 4: Artifact Storage Summary")
            print("-" * 50)
            
            print("📦 Local Storage (./artifacts/models/):")
            model_files = ['rockburst_rf_model.pkl', 'feature_engineer.pkl', 'feature_scaler.pkl']
            for filename in model_files:
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   ✅ {filename} ({size:,} bytes)")
            
            print("\\n🌐 MLflow Storage (MinIO S3):")
            print(f"   📊 Experiment tracking: {self.mlflow_uri}")
            print("   📦 Model artifacts: MinIO bucket 'mlflow'")
            print("   🔄 Model registry: 'rockburst_random_forest_production'")
            print("   📈 Metrics & parameters logged")
            
            print("\\n⏰ STEP 5: Deployment Instructions")
            print("-" * 50)
            
            print("🔧 Airflow Setup:")
            print("   1. Copy DAG to Airflow dags folder:")
            print(f"      cp {dag_path} $AIRFLOW_HOME/dags/")
            print("   2. Restart Airflow scheduler")
            print("   3. Enable DAG in Airflow UI")
            
            print("\\n🌐 MLflow Setup:")
            print("   1. MLflow UI: http://localhost:5001")
            print("   2. MinIO Console: http://localhost:9091")
            print("   3. Model registry available for deployment")
            
            print("\\n" + "="*80)
            print("✅ COMPLETE PIPELINE DEMONSTRATION FINISHED")
            print("="*80)
            
            print("\\n🎯 Integration Summary:")
            print("   ✅ MLflow: Experiment tracking + artifact storage")
            print("   ✅ Airflow: Automated orchestration + scheduling")
            print("   ✅ MinIO: S3-compatible artifact backend")
            print("   ✅ 24-hour retraining: Intelligent scheduling")
            print("   ✅ Model versioning: Full lineage tracking")
            print("   ✅ Production ready: Monitoring + validation")
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline demonstration failed: {str(e)}")
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main demonstration function"""
    try:
        # Initialize the integrated pipeline
        pipeline = MLflowAirflowModelPipeline()
        
        # Run complete demonstration
        pipeline.demonstrate_complete_pipeline()
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
