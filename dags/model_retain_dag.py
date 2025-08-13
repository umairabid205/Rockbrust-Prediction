"""
Rockburst Model Retraining DAG
==============================
This DAG handles daily retraining of the rockburst prediction model 
using Random Forest with incremental feedback data.

Author: Team GAMMA
Created: August 10, 2025
"""

# Standard library imports
import sys
import os
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Add project root to Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')

# Import custom logging
try:
    from exception_logging.logger import logging as custom_logging
    from exception_logging.exception import CustomException
except ImportError:
    custom_logging = logging
    CustomException = Exception

# DAG configuration
default_args = {
    'owner': 'TEAM_GAMMA',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
    'max_active_runs': 1,
}

dag = DAG(
    'rockburst_daily_retraining',
    default_args=default_args,
    description='Daily retraining of rockburst prediction model with Random Forest',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['rockburst', 'retraining', 'random-forest', 'mlops'],
    max_active_tasks=3,
)

# Configuration
MODEL_DIR = '/opt/airflow/models'
DATA_DIR = '/opt/airflow/data/processed'
FEEDBACK_DB_CONFIG = {
    'host': Variable.get("FEEDBACK_DB_HOST", default_var="feedback-db"),
    'port': Variable.get("FEEDBACK_DB_PORT", default_var=5432),
    'user': Variable.get("FEEDBACK_DB_USER"),
    'password': Variable.get("FEEDBACK_DB_PASSWORD"),
    'database': Variable.get("FEEDBACK_DB_NAME", default_var="feedback_db")
}
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 8,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# ============================
# Task 1: Fetch New Feedback
# ============================
def fetch_new_feedback(**context):
    """Retrieve engineer feedback since last retraining"""
    try:
        custom_logging.info("Fetching new engineer feedback")
        last_run = context['dag_run'].get_previous_dagrun()
        last_date = last_run.execution_date if last_run else datetime(2025, 1, 1)
        
        # Simulated feedback retrieval (replace with actual DB query)
        feedback_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(24)],
            'sensor_data': [np.random.rand(10).tobytes() for _ in range(24)],
            'predicted_class': np.random.randint(0, 3, 24),
            'corrected_class': np.random.randint(0, 3, 24),
            'engineer_id': [f"eng{np.random.randint(100,999)}" for _ in range(24)]
        })
        
        # Save to parquet for downstream tasks
        os.makedirs(DATA_DIR, exist_ok=True)
        feedback_path = f"{DATA_DIR}/feedback_{context['ds']}.parquet"
        feedback_data.to_parquet(feedback_path)
        
        custom_logging.info(f"Fetched {len(feedback_data)} new feedback records")
        return {
            'feedback_path': feedback_path,
            'feedback_count': len(feedback_data),
            'start_date': last_date.isoformat(),
            'end_date': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error fetching feedback: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# =============================
# Task 2: Prepare Training Data
# =============================
def prepare_training_data(**context):
    """Combine historical data with new feedback"""
    try:
        feedback_info = context['task_instance'].xcom_pull(task_ids='fetch_feedback_task')
        feedback_path = feedback_info['feedback_path']
        
        custom_logging.info("Loading historical training data")
        # Load base dataset (replace with actual data source)
        base_data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000),
            'target': np.random.randint(0, 3, 1000)
        })
        
        custom_logging.info("Loading new feedback data")
        feedback_data = pd.read_parquet(feedback_path)
        # Simulate feature engineering from sensor data
        feedback_data['feature1'] = np.random.rand(len(feedback_data))
        feedback_data['feature2'] = np.random.rand(len(feedback_data))
        feedback_data['target'] = feedback_data['corrected_class']
        
        # Combine datasets
        combined_data = pd.concat([base_data, feedback_data[['feature1', 'feature2', 'target']]])
        combined_path = f"{DATA_DIR}/training_data_{context['ds']}.parquet"
        combined_data.to_parquet(combined_path)
        
        custom_logging.info(f"Combined dataset size: {len(combined_data)} records")
        return {
            'dataset_path': combined_path,
            'base_records': len(base_data),
            'new_records': len(feedback_data)
        }
        
    except Exception as e:
        error_msg = f"Error preparing data: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# =============================
# Task 3: Train Random Forest
# =============================
def train_random_forest(**context):
    """Train and validate Random Forest model"""
    try:
        data_info = context['task_instance'].xcom_pull(task_ids='prepare_data_task')
        data_path = data_info['dataset_path']
        
        custom_logging.info("Loading training data")
        df = pd.read_parquet(data_path)
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=y_train
        )
        class_weights = dict(zip(classes, weights))
        
        custom_logging.info("Training Random Forest model")
        model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG['n_estimators'],
            max_depth=MODEL_CONFIG['max_depth'],
            class_weight=class_weights,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=MODEL_CONFIG['n_jobs']
        )
        model.fit(X_train, y_train)
        
        custom_logging.info("Evaluating model")
        val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred, average='weighted')
        report = classification_report(y_val, val_pred, output_dict=True)
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/model_{context['ds']}.joblib"
        joblib.dump(model, model_path)
        
        custom_logging.info(f"Model saved: {model_path}")
        return {
            'model_path': model_path,
            'accuracy': accuracy,
            'f1_score': f1,
            'class_report': report,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# =============================
# Task 4: Validate Model
# =============================
def validate_model(**context):
    """Compare new model with current production model"""
    try:
        model_info = context['task_instance'].xcom_pull(task_ids='train_model_task')
        
        # Simulate comparison with current model
        current_model_accuracy = 0.82  # Would come from model registry
        new_model_accuracy = model_info['accuracy']
        
        improvement = new_model_accuracy - current_model_accuracy
        deploy = improvement >= 0.01  # 1% improvement threshold
        
        custom_logging.info(f"Accuracy improvement: {improvement:.4f}")
        return {
            'deploy': deploy,
            'accuracy': new_model_accuracy,
            'improvement': improvement,
            'threshold': 0.01
        }
        
    except Exception as e:
        error_msg = f"Error validating model: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# =============================
# Task 5: Deploy Model
# =============================
def deploy_model(**context):
    """Deploy validated model to production"""
    try:
        validation = context['task_instance'].xcom_pull(task_ids='validate_model_task')
        model_info = context['task_instance'].xcom_pull(task_ids='train_model_task')
        
        if not validation['deploy']:
            custom_logging.warning("Model not deployed - insufficient improvement")
            return {'status': 'skipped', 'reason': 'Insufficient improvement'}
        
        # Simulate deployment process
        model_path = model_info['model_path']
        prod_path = f"{MODEL_DIR}/production_model.joblib"
        
        # In production: would use model registry and blue/green deployment
        os.rename(model_path, prod_path)
        
        custom_logging.info(f"Model deployed to production: {prod_path}")
        return {
            'status': 'deployed',
            'model_path': prod_path,
            'deployment_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error deploying model: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# =============================
# Task 6: Generate Retraining Report
# =============================
def generate_retraining_report(**context):
    """Create comprehensive retraining report"""
    try:
        custom_logging.info("Generating retraining report")
        
        # Collect data from all tasks
        feedback_info = context['task_instance'].xcom_pull(task_ids='fetch_feedback_task')
        data_info = context['task_instance'].xcom_pull(task_ids='prepare_data_task')
        model_info = context['task_instance'].xcom_pull(task_ids='train_model_task')
        validation = context['task_instance'].xcom_pull(task_ids='validate_model_task')
        deployment = context['task_instance'].xcom_pull(task_ids='deploy_model_task')
        
        # Build report
        report = {
            'execution_date': context['ds'],
            'feedback': feedback_info,
            'dataset': data_info,
            'model_training': model_info,
            'validation': validation,
            'deployment': deployment,
            'system_metrics': {
                'memory_usage': '1.2GB',  # Would use psutil in real implementation
                'duration': '45 minutes'
            }
        }
        
        # Save report
        report_dir = '/opt/airflow/reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/retraining_report_{context['ds']}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        custom_logging.info(f"Report saved: {report_path}")
        return {'report_path': report_path}
        
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        custom_logging.error(error_msg)
        raise CustomException(e, sys)

# ========================
# Define DAG Tasks
# ========================
fetch_feedback_task = PythonOperator(
    task_id='fetch_feedback_task',
    python_callable=fetch_new_feedback,
    dag=dag,
    doc_md="""**Fetch New Feedback**\n\nRetrieves engineer corrections since last retraining run from feedback database"""
)

prepare_data_task = PythonOperator(
    task_id='prepare_data_task',
    python_callable=prepare_training_data,
    dag=dag,
    doc_md="""**Prepare Training Data**\n\nCombines historical data with new feedback for training"""
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_random_forest,
    dag=dag,
    doc_md="""**Train Random Forest**\n\nTrains new model with updated dataset and validates performance"""
)

validate_model_task = PythonOperator(
    task_id='validate_model_task',
    python_callable=validate_model,
    dag=dag,
    doc_md="""**Validate Model**\n\nCompares new model with current production version using accuracy metrics"""
)

deploy_model_task = PythonOperator(
    task_id='deploy_model_task',
    python_callable=deploy_model,
    dag=dag,
    doc_md="""**Deploy Model**\n\nDeploys new model to production if it meets improvement thresholds"""
)

generate_report_task = PythonOperator(
    task_id='generate_report_task',
    python_callable=generate_retraining_report,
    dag=dag,
    doc_md="""**Generate Report**\n\nCreates comprehensive retraining report with metrics"""
)

# ========================
# Task Dependencies
# ========================
fetch_feedback_task >> prepare_data_task >> train_model_task
train_model_task >> validate_model_task >> deploy_model_task
deploy_model_task >> generate_report_task

# Log DAG initialization
custom_logging.info("Rockburst retraining DAG initialized successfully")