"""
Rockburst Feature Engineering DAG
==================================
This DAG handles advanced feature engineering for rockburst prediction using user input data.
It creates derived features from the 9 base columns provided by users.

Author: Data Science Team GAMMA 

"""
# Standard library imports - for basic Python functionality
import sys
import os
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Airflow imports - for workflow orchestration
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Add project root to Python path for custom imports
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')

# Import custom logging and exception handling
try:
    from exception_logging.logger import logging as custom_logging
    from exception_logging.exception import CustomException
    # Log successful import of custom modules
    logging.info("Successfully imported custom logging for feature engineering")
except ImportError as e:
    # Fallback to standard logging if custom modules not available
    logging.warning(f"Could not import custom logging modules: {str(e)}")
    custom_logging = logging
    CustomException = Exception



# DAG default arguments - configuration for all tasks in this DAG
default_args = {
    'owner': 'Data Science Team GAMMA',  # Owner of the feature engineering DAG
    'depends_on_past': False,  # Don't wait for previous runs to complete
    'start_date': datetime(2025, 8, 10),  # When this DAG should start running
    'email_on_failure': False,  # Don't send email on task failure
    'email_on_retry': False,  # Don't send email on task retry
    'retries': 2,  # Number of retries if task fails
    'retry_delay': timedelta(minutes=5),  # Wait time between retries
    'max_active_runs': 1,  # Only one instance of this DAG can run at a time
}



# Create the feature engineering DAG instance
dag = DAG(
    'rockburst_feature_engineering',  # Unique identifier for this DAG
    default_args=default_args,  # Use the default arguments defined above
    description='Feature engineering for user-input rockburst prediction data',  # Human-readable description
    schedule=timedelta(hours=12),  # Run every 12 hours
    catchup=False,  # Don't run for past dates when DAG is first created
    tags=['rockbrust', 'feature-engineering', 'user-data', 'ml-features'],  # Tags for organization
    max_active_tasks=2,  # Maximum number of tasks that can run simultaneously
)



# User Data Feature Configuration - Based on actual database columns
USER_DATA_FEATURES = {
    'base_features': [
        'Duration_days',                    # Duration of observation period in days
        'Energy_Unit_log',                  # Logarithmic energy units
        'Energy_density_Joule_sqr',         # Energy density in Joules squared  
        'Volume_m3_sqr',                    # Volume in cubic meters squared
        'Event_freq_unit_per_day_log',      # Logarithmic event frequency per day
        'Energy_Joule_per_day_sqr',         # Energy per day in Joules squared
        'Volume_m3_per_day_sqr',            # Volume per day in cubic meters squared
        'Energy_per_Volume_log',            # Logarithmic energy per volume ratio
        'Intensity_Level_encoded'           # Encoded intensity level (target variable)
    ],
    'derived_features': {
        'ratios': ['Energy_per_Duration', 'Volume_per_Duration', 'Freq_per_Duration'],
        'interactions': ['Energy_Volume_interaction', 'Freq_Energy_interaction'],
        'polynomial': ['Energy_squared', 'Volume_squared', 'Duration_squared'],
        'statistical': ['Energy_Volume_ratio', 'Normalized_frequency', 'Density_ratio']
    },
    'scaling_methods': ['standard', 'minmax', 'robust']
}

# Log the feature configuration
custom_logging.info(f"User data feature engineering configured with {len(USER_DATA_FEATURES['base_features'])} base features")





def check_processed_data_availability(**context):
    """
    Task 1: Check for availability of user data processed by ingestion DAG
    
    This function:
    1. Checks for processed user data files
    2. Validates they contain the required columns
    3. Prepares data for feature engineering
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Status of processed user data availability
    """
    try:
        # Log start of data availability check
        custom_logging.info("Starting processed user data availability check")
        
        # Define paths for processed data
        processed_data_path = '/opt/airflow/data/processed'
        features_data_path = '/opt/airflow/data/features'
        
        custom_logging.info(f"Checking processed data path: {processed_data_path}")
        
        # Create features directory if it doesn't exist
        if not os.path.exists(features_data_path):
            os.makedirs(features_data_path, exist_ok=True)
            custom_logging.info(f"Created features directory: {features_data_path}")
        
        # Check for processed user data files
        user_data_files = []
        if os.path.exists(processed_data_path):
            all_files = os.listdir(processed_data_path)
            user_data_files = [f for f in all_files if f.lower().endswith('.csv') and 'preprocessed' in f.lower()]
            
            custom_logging.info(f"Found {len(user_data_files)} processed user data files")
            for file in user_data_files:
                file_path = os.path.join(processed_data_path, file)
                file_size = os.path.getsize(file_path)
                custom_logging.info(f"User data file: {file} (Size: {file_size} bytes)")
        
        # Prepare return data
        data_status = {
            'processed_files_found': user_data_files,
            'file_count': len(user_data_files),
            'data_ready_for_features': len(user_data_files) > 0,
            'check_timestamp': datetime.now().isoformat(),
            'task_status': 'success'
        }
        
        custom_logging.info(f"Data availability check completed: {data_status}")
        return data_status
        
    except Exception as e:
        error_message = f"Error in check_processed_data_availability: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)
    




def create_derived_features(**context):
    """
    Task 2: Create derived features from the 9 base user input columns
    
    This function:
    1. Loads processed user data
    2. Creates ratio features between different measurements
    3. Generates interaction features
    4. Creates polynomial features for non-linear relationships
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Results of derived feature creation
    """
    try:
        # Log start of feature creation
        custom_logging.info("Starting derived feature creation from user data")
        
        # Get data availability from previous task
        data_status = context['task_instance'].xcom_pull(task_ids='check_data_task')
        custom_logging.info("Retrieved data availability status")
        
        # Check if data is available
        if not data_status['data_ready_for_features']:
            custom_logging.warning("No processed user data available for feature engineering")
            return {
                'feature_creation_status': 'no_data',
                'message': 'No processed user data available',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize results
        feature_results = {
            'processed_files': [],
            'total_derived_features': 0,
            'feature_categories': [],
            'output_files': [],
            'creation_summary': {},
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # Define paths
        processed_data_path = '/opt/airflow/data/processed'
        features_data_path = '/opt/airflow/data/features'
        
        # Get base features from configuration
        base_features = USER_DATA_FEATURES['base_features']
        target_column = 'Intensity_Level_encoded'
        numeric_features = [col for col in base_features if col != target_column]
        
        custom_logging.info(f"Working with {len(numeric_features)} numeric features and target: {target_column}")
        
        # Process each user data file
        for data_file in data_status['processed_files_found']:
            custom_logging.info(f"Creating derived features for: {data_file}")
            
            try:
                # Load user data
                input_path = os.path.join(processed_data_path, data_file)
                df = pd.read_csv(input_path)
                
                custom_logging.info(f"Loaded user data with shape: {df.shape}")
                
                # Verify required columns are present
                available_features = [col for col in base_features if col in df.columns]
                missing_features = [col for col in base_features if col not in df.columns]
                
                if missing_features:
                    custom_logging.warning(f"Missing features in {data_file}: {missing_features}")
                
                available_numeric = [col for col in available_features if col != target_column]
                custom_logging.info(f"Available numeric features: {available_numeric}")
                
                # Initialize derived features DataFrame
                derived_df = df.copy()  # Start with original data
                
                # Category 1: Ratio Features
                custom_logging.info("Creating ratio-based derived features")
                ratio_features_created = 0
                
                if 'Energy_Unit_log' in available_numeric and 'Duration_days' in available_numeric:
                    derived_df['Energy_per_Duration'] = derived_df['Energy_Unit_log'] / (derived_df['Duration_days'] + 1e-8)
                    ratio_features_created += 1
                    custom_logging.info("Created Energy_per_Duration ratio feature")
                
                if 'Volume_m3_sqr' in available_numeric and 'Duration_days' in available_numeric:
                    derived_df['Volume_per_Duration'] = derived_df['Volume_m3_sqr'] / (derived_df['Duration_days'] + 1e-8)
                    ratio_features_created += 1
                    custom_logging.info("Created Volume_per_Duration ratio feature")
                
                if 'Event_freq_unit_per_day_log' in available_numeric and 'Duration_days' in available_numeric:
                    derived_df['Freq_per_Duration'] = derived_df['Event_freq_unit_per_day_log'] / (derived_df['Duration_days'] + 1e-8)
                    ratio_features_created += 1
                    custom_logging.info("Created Freq_per_Duration ratio feature")
                
                # Category 2: Interaction Features
                custom_logging.info("Creating interaction derived features")
                interaction_features_created = 0
                
                if 'Energy_density_Joule_sqr' in available_numeric and 'Volume_m3_sqr' in available_numeric:
                    derived_df['Energy_Volume_interaction'] = derived_df['Energy_density_Joule_sqr'] * derived_df['Volume_m3_sqr']
                    interaction_features_created += 1
                    custom_logging.info("Created Energy_Volume_interaction feature")
                
                if 'Event_freq_unit_per_day_log' in available_numeric and 'Energy_Unit_log' in available_numeric:
                    derived_df['Freq_Energy_interaction'] = derived_df['Event_freq_unit_per_day_log'] * derived_df['Energy_Unit_log']
                    interaction_features_created += 1
                    custom_logging.info("Created Freq_Energy_interaction feature")
                
                # Category 3: Polynomial Features (squared terms for capturing non-linear relationships)
                custom_logging.info("Creating polynomial derived features")
                polynomial_features_created = 0
                
                if 'Energy_Unit_log' in available_numeric:
                    derived_df['Energy_squared'] = derived_df['Energy_Unit_log'] ** 2
                    polynomial_features_created += 1
                    custom_logging.info("Created Energy_squared polynomial feature")
                
                if 'Volume_m3_sqr' in available_numeric:
                    derived_df['Volume_squared'] = derived_df['Volume_m3_sqr'] ** 2
                    polynomial_features_created += 1
                    custom_logging.info("Created Volume_squared polynomial feature")
                
                if 'Duration_days' in available_numeric:
                    derived_df['Duration_squared'] = derived_df['Duration_days'] ** 2
                    polynomial_features_created += 1
                    custom_logging.info("Created Duration_squared polynomial feature")
                
                # Category 4: Statistical Derived Features
                custom_logging.info("Creating statistical derived features")
                statistical_features_created = 0
                
                if 'Energy_density_Joule_sqr' in available_numeric and 'Volume_m3_sqr' in available_numeric:
                    derived_df['Energy_Volume_ratio'] = derived_df['Energy_density_Joule_sqr'] / (derived_df['Volume_m3_sqr'] + 1e-8)
                    statistical_features_created += 1
                    custom_logging.info("Created Energy_Volume_ratio statistical feature")
                
                if 'Event_freq_unit_per_day_log' in available_numeric:
                    # Normalize frequency by dataset statistics
                    freq_mean = derived_df['Event_freq_unit_per_day_log'].mean()
                    freq_std = derived_df['Event_freq_unit_per_day_log'].std()
                    derived_df['Normalized_frequency'] = (derived_df['Event_freq_unit_per_day_log'] - freq_mean) / (freq_std + 1e-8)
                    statistical_features_created += 1
                    custom_logging.info("Created Normalized_frequency statistical feature")
                
                if 'Energy_density_Joule_sqr' in available_numeric and 'Energy_per_Volume_log' in available_numeric:
                    derived_df['Density_ratio'] = derived_df['Energy_density_Joule_sqr'] / (np.exp(derived_df['Energy_per_Volume_log']) + 1e-8)
                    statistical_features_created += 1
                    custom_logging.info("Created Density_ratio statistical feature")
                
                # Calculate total derived features created
                total_derived = ratio_features_created + interaction_features_created + polynomial_features_created + statistical_features_created
                
                # Add metadata
                derived_df['original_file'] = data_file
                derived_df['feature_creation_timestamp'] = datetime.now().isoformat()
                derived_df['original_feature_count'] = len(available_features)
                derived_df['derived_feature_count'] = total_derived
                
                # Log feature creation statistics
                custom_logging.info(f"Derived feature creation for {data_file}:")
                custom_logging.info(f"  - Ratio features: {ratio_features_created}")
                custom_logging.info(f"  - Interaction features: {interaction_features_created}")
                custom_logging.info(f"  - Polynomial features: {polynomial_features_created}")
                custom_logging.info(f"  - Statistical features: {statistical_features_created}")
                custom_logging.info(f"  - Total derived features: {total_derived}")
                
                # Save derived features
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = os.path.splitext(data_file)[0]
                derived_filename = f"{base_filename}_derived_features_{timestamp}.csv"
                derived_path = os.path.join(features_data_path, derived_filename)
                
                derived_df.to_csv(derived_path, index=False)
                
                if os.path.exists(derived_path):
                    output_file_size = os.path.getsize(derived_path)
                    custom_logging.info(f"Saved derived features: {derived_filename} (Size: {output_file_size} bytes)")
                    
                    # Add to results
                    file_result = {
                        'input_filename': data_file,
                        'output_filename': derived_filename,
                        'original_features': len(available_features),
                        'derived_features': total_derived,
                        'total_features': len(derived_df.columns) - 4,  # Exclude metadata columns
                        'feature_categories': ['ratio', 'interaction', 'polynomial', 'statistical'],
                        'creation_timestamp': datetime.now().isoformat()
                    }
                    
                    feature_results['processed_files'].append(file_result)
                    feature_results['total_derived_features'] += total_derived
                    feature_results['output_files'].append(derived_filename)
                
                custom_logging.info(f"Derived feature creation completed for {data_file}")
                
            except Exception as file_error:
                error_msg = f"Error creating derived features for {data_file}: {str(file_error)}"
                custom_logging.error(error_msg)
                
                feature_results['processed_files'].append({
                    'input_filename': data_file,
                    'creation_status': 'error',
                    'error_message': error_msg,
                    'creation_timestamp': datetime.now().isoformat()
                })
        
        # Generate creation summary
        if feature_results['processed_files']:
            successful_files = [f for f in feature_results['processed_files'] if f.get('creation_status') != 'error']
            
            feature_results['creation_summary'] = {
                'total_files_processed': len(feature_results['processed_files']),
                'successful_creations': len(successful_files),
                'total_derived_features_created': feature_results['total_derived_features'],
                'average_features_per_file': round(feature_results['total_derived_features'] / len(successful_files), 2) if successful_files else 0
            }
        
        # Set feature categories used
        feature_results['feature_categories'] = ['ratio', 'interaction', 'polynomial', 'statistical']
        
        custom_logging.info(f"Derived feature creation summary: {feature_results['creation_summary']}")
        custom_logging.info("Derived feature creation task completed successfully")
        
        return feature_results
        
    except Exception as e:
        error_message = f"Error in create_derived_features: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)
    





def apply_feature_scaling(**context):
    """
    Task 3: Apply different scaling methods to the derived features
    
    This function:
    1. Loads files with derived features
    2. Applies StandardScaler, MinMaxScaler, and RobustScaler
    3. Creates separate datasets for each scaling method
    4. Prepares data for ML model training
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Results of feature scaling operations
    """
    try:
        # Log start of feature scaling
        custom_logging.info("Starting feature scaling task")
        
        # Get derived features results from previous task
        feature_results = context['task_instance'].xcom_pull(task_ids='create_features_task')
        custom_logging.info("Retrieved derived features results")
        
        # Check if features were created
        if not feature_results.get('output_files'):
            custom_logging.warning("No derived features available for scaling")
            return {
                'scaling_status': 'no_features',
                'message': 'No derived features available for scaling',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize scaling results
        scaling_results = {
            'scaled_datasets': [],
            'scaling_methods_applied': [],
            'total_features_scaled': 0,
            'output_files': [],
            'scaling_summary': {},
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # Define paths
        features_data_path = '/opt/airflow/data/features'
        
        # Get scaling methods from configuration
        scaling_methods = USER_DATA_FEATURES['scaling_methods']
        custom_logging.info(f"Will apply scaling methods: {scaling_methods}")
        
        # Process each derived feature file
        for file_info in feature_results['processed_files']:
            if file_info.get('creation_status') == 'error':
                custom_logging.warning(f"Skipping file with creation errors: {file_info['input_filename']}")
                continue
                
            if 'output_filename' not in file_info:
                custom_logging.warning(f"No output filename found for: {file_info['input_filename']}")
                continue
                
            feature_filename = file_info['output_filename']
            custom_logging.info(f"Starting scaling for feature file: {feature_filename}")
            
            try:
                # Load derived features
                feature_path = os.path.join(features_data_path, feature_filename)
                df = pd.read_csv(feature_path)
                
                custom_logging.info(f"Loaded feature data with shape: {df.shape}")
                
                # Identify feature columns (exclude metadata and target)
                metadata_columns = [col for col in df.columns if any(keyword in col.lower() 
                                   for keyword in ['timestamp', 'file', 'original', 'creation', 'count'])]
                target_column = 'Intensity_Level_encoded'
                
                feature_columns = [col for col in df.columns 
                                 if col not in metadata_columns and col != target_column]
                
                custom_logging.info(f"Identified {len(feature_columns)} feature columns for scaling")
                custom_logging.info(f"Metadata columns: {len(metadata_columns)}")
                
                # Prepare base DataFrame with metadata and target
                base_df = pd.DataFrame()
                for meta_col in metadata_columns:
                    if meta_col in df.columns:
                        base_df[meta_col] = df[meta_col]
                
                if target_column in df.columns:
                    base_df[target_column] = df[target_column]
                    custom_logging.info("Target column preserved for scaled datasets")
                
                # Apply each scaling method
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = os.path.splitext(feature_filename)[0]
                
                # Apply StandardScaler
                if 'standard' in scaling_methods and feature_columns:
                    custom_logging.info("Applying StandardScaler")
                    standard_scaler = StandardScaler()
                    scaled_features_standard = standard_scaler.fit_transform(df[feature_columns].fillna(0))
                    
                    # Create scaled DataFrame
                    scaled_df_standard = base_df.copy()
                    scaled_feature_names = [f'{col}_std' for col in feature_columns]
                    
                    for i, col_name in enumerate(scaled_feature_names):
                        scaled_df_standard[col_name] = scaled_features_standard[:, i]
                    
                    # Add scaling metadata
                    scaled_df_standard['scaling_method'] = 'standard'
                    scaled_df_standard['scaling_timestamp'] = datetime.now().isoformat()
                    scaled_df_standard['features_scaled'] = len(feature_columns)
                    
                    # Save StandardScaler version
                    standard_filename = f"{base_filename}_scaled_standard_{timestamp}.csv"
                    standard_path = os.path.join(features_data_path, standard_filename)
                    scaled_df_standard.to_csv(standard_path, index=False)
                    
                    if os.path.exists(standard_path):
                        custom_logging.info(f"Saved StandardScaler dataset: {standard_filename}")
                        scaling_results['output_files'].append(standard_filename)
                        if 'standard' not in scaling_results['scaling_methods_applied']:
                            scaling_results['scaling_methods_applied'].append('standard')
                
                # Apply MinMaxScaler
                if 'minmax' in scaling_methods and feature_columns:
                    custom_logging.info("Applying MinMaxScaler")
                    minmax_scaler = MinMaxScaler()
                    scaled_features_minmax = minmax_scaler.fit_transform(df[feature_columns].fillna(0))
                    
                    # Create scaled DataFrame
                    scaled_df_minmax = base_df.copy()
                    scaled_feature_names = [f'{col}_minmax' for col in feature_columns]
                    
                    for i, col_name in enumerate(scaled_feature_names):
                        scaled_df_minmax[col_name] = scaled_features_minmax[:, i]
                    
                    # Add scaling metadata
                    scaled_df_minmax['scaling_method'] = 'minmax'
                    scaled_df_minmax['scaling_timestamp'] = datetime.now().isoformat()
                    scaled_df_minmax['features_scaled'] = len(feature_columns)
                    
                    # Save MinMaxScaler version
                    minmax_filename = f"{base_filename}_scaled_minmax_{timestamp}.csv"
                    minmax_path = os.path.join(features_data_path, minmax_filename)
                    scaled_df_minmax.to_csv(minmax_path, index=False)
                    
                    if os.path.exists(minmax_path):
                        custom_logging.info(f"Saved MinMaxScaler dataset: {minmax_filename}")
                        scaling_results['output_files'].append(minmax_filename)
                        if 'minmax' not in scaling_results['scaling_methods_applied']:
                            scaling_results['scaling_methods_applied'].append('minmax')
                
                # Apply RobustScaler
                if 'robust' in scaling_methods and feature_columns:
                    custom_logging.info("Applying RobustScaler")
                    robust_scaler = RobustScaler()
                    scaled_features_robust = robust_scaler.fit_transform(df[feature_columns].fillna(0))
                    
                    # Create scaled DataFrame
                    scaled_df_robust = base_df.copy()
                    scaled_feature_names = [f'{col}_robust' for col in feature_columns]
                    
                    for i, col_name in enumerate(scaled_feature_names):
                        scaled_df_robust[col_name] = scaled_features_robust[:, i]
                    
                    # Add scaling metadata
                    scaled_df_robust['scaling_method'] = 'robust'
                    scaled_df_robust['scaling_timestamp'] = datetime.now().isoformat()
                    scaled_df_robust['features_scaled'] = len(feature_columns)
                    
                    # Save RobustScaler version
                    robust_filename = f"{base_filename}_scaled_robust_{timestamp}.csv"
                    robust_path = os.path.join(features_data_path, robust_filename)
                    scaled_df_robust.to_csv(robust_path, index=False)
                    
                    if os.path.exists(robust_path):
                        custom_logging.info(f"Saved RobustScaler dataset: {robust_filename}")
                        scaling_results['output_files'].append(robust_filename)
                        if 'robust' not in scaling_results['scaling_methods_applied']:
                            scaling_results['scaling_methods_applied'].append('robust')
                
                # Record dataset information
                dataset_info = {
                    'source_feature_file': feature_filename,
                    'features_scaled': len(feature_columns),
                    'scaling_methods_applied': scaling_methods,
                    'metadata_preserved': len(metadata_columns),
                    'target_preserved': target_column in df.columns,
                    'scaling_timestamp': datetime.now().isoformat()
                }
                
                scaling_results['scaled_datasets'].append(dataset_info)
                scaling_results['total_features_scaled'] += len(feature_columns)
                
                custom_logging.info(f"Feature scaling completed for {feature_filename}")
                custom_logging.info(f"  - Features scaled: {len(feature_columns)}")
                custom_logging.info(f"  - Scaling methods applied: {len(scaling_methods)}")
                
            except Exception as file_error:
                error_msg = f"Error scaling features for {feature_filename}: {str(file_error)}"
                custom_logging.error(error_msg)
                
                scaling_results['scaled_datasets'].append({
                    'source_feature_file': feature_filename,
                    'scaling_status': 'error',
                    'error_message': error_msg,
                    'scaling_timestamp': datetime.now().isoformat()
                })
        
        # Generate scaling summary
        if scaling_results['scaled_datasets']:
            successful_scaling = [d for d in scaling_results['scaled_datasets'] if d.get('scaling_status') != 'error']
            
            scaling_results['scaling_summary'] = {
                'total_feature_files': len(scaling_results['scaled_datasets']),
                'successful_scalings': len(successful_scaling),
                'scaling_methods_used': len(scaling_results['scaling_methods_applied']),
                'total_output_files': len(scaling_results['output_files']),
                'total_features_processed': scaling_results['total_features_scaled']
            }
        
        custom_logging.info(f"Feature scaling summary: {scaling_results['scaling_summary']}")
        custom_logging.info("Feature scaling task completed successfully")
        
        return scaling_results
        
    except Exception as e:
        error_message = f"Error in apply_feature_scaling: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)
    




def combine_and_scale_features(**context):
    """
    Task 4: Final validation and summary of all feature engineering results
    
    This function:
    1. Validates all scaled feature datasets 
    2. Creates a comprehensive feature engineering summary
    3. Prepares final datasets for ML model training
    4. Performs data quality checks
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Final feature engineering results and validation summary
    """
    try:
        # Log start of final feature validation
        custom_logging.info("Starting final feature validation and summary task")
        
        # Get results from previous scaling task
        scaling_results = context['task_instance'].xcom_pull(task_ids='apply_scaling_task')
        custom_logging.info("Retrieved scaling results for final validation")
        
        # Check if scaling was completed
        if not scaling_results.get('output_files'):
            custom_logging.warning("No scaled features available for validation")
            return {
                'validation_status': 'no_scaled_features',
                'message': 'No scaled features available for validation',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize final validation results
        final_results = {
            'validated_datasets': [],
            'final_feature_counts': {},
            'data_quality_summary': {},
            'ready_for_ml_training': False,
            'output_summary': {},
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # Define paths
        features_data_path = '/opt/airflow/data/features'
        
        # Validate each scaled dataset
        custom_logging.info("Starting final dataset validation")
        
        total_datasets = 0
        valid_datasets = 0
        
        for output_file in scaling_results['output_files']:
            custom_logging.info(f"Validating scaled dataset: {output_file}")
            
            try:
                # Load and validate the scaled dataset
                file_path = os.path.join(features_data_path, output_file)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    # Basic validation checks
                    validation_checks = {
                        'file_exists': True,
                        'data_loaded': not df.empty,
                        'has_features': len(df.columns) > 5,  # Should have features + metadata
                        'has_target': 'Intensity_Level_encoded' in df.columns,
                        'no_all_nan_columns': not df.isnull().all().any(),
                        'reasonable_size': len(df) > 0 and len(df.columns) > 0
                    }
                    
                    # Count different types of columns
                    metadata_cols = [col for col in df.columns if any(keyword in col.lower() 
                                    for keyword in ['timestamp', 'file', 'original', 'method', 'scaled'])]
                    target_cols = [col for col in df.columns if 'intensity_level' in col.lower()]
                    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
                    
                    dataset_info = {
                        'filename': output_file,
                        'total_columns': len(df.columns),
                        'feature_columns': len(feature_cols),
                        'metadata_columns': len(metadata_cols),
                        'target_columns': len(target_cols),
                        'total_rows': len(df),
                        'validation_passed': all(validation_checks.values()),
                        'validation_checks': validation_checks,
                        'scaling_method': df['scaling_method'].iloc[0] if 'scaling_method' in df.columns else 'unknown',
                        'validation_timestamp': datetime.now().isoformat()
                    }
                    
                    if dataset_info['validation_passed']:
                        valid_datasets += 1
                        custom_logging.info(f"Dataset validation PASSED: {output_file}")
                        custom_logging.info(f"  - Features: {len(feature_cols)}, Rows: {len(df)}")
                    else:
                        custom_logging.warning(f"Dataset validation FAILED: {output_file}")
                        failed_checks = [check for check, result in validation_checks.items() if not result]
                        custom_logging.warning(f"  - Failed checks: {failed_checks}")
                    
                    final_results['validated_datasets'].append(dataset_info)
                    total_datasets += 1
                    
                else:
                    custom_logging.error(f"Scaled dataset file not found: {output_file}")
                    final_results['validated_datasets'].append({
                        'filename': output_file,
                        'validation_passed': False,
                        'error': 'File not found',
                        'validation_timestamp': datetime.now().isoformat()
                    })
                    total_datasets += 1
                    
            except Exception as file_error:
                error_msg = f"Error validating {output_file}: {str(file_error)}"
                custom_logging.error(error_msg)
                
                final_results['validated_datasets'].append({
                    'filename': output_file,
                    'validation_passed': False,
                    'error': error_msg,
                    'validation_timestamp': datetime.now().isoformat()
                })
                total_datasets += 1
        
        # Calculate validation summary
        validation_rate = (valid_datasets / total_datasets) * 100 if total_datasets > 0 else 0
        
        # Create final feature counts summary
        scaling_methods = set()
        total_features_by_method = {}
        
        for dataset in final_results['validated_datasets']:
            if dataset.get('validation_passed') and 'scaling_method' in dataset:
                method = dataset['scaling_method']
                scaling_methods.add(method)
                if method not in total_features_by_method:
                    total_features_by_method[method] = 0
                total_features_by_method[method] += dataset.get('feature_columns', 0)
        
        final_results['final_feature_counts'] = {
            'scaling_methods_available': list(scaling_methods),
            'features_per_method': total_features_by_method,
            'total_valid_datasets': valid_datasets,
            'total_datasets': total_datasets
        }
        
        # Data quality summary
        final_results['data_quality_summary'] = {
            'validation_rate': round(validation_rate, 2),
            'datasets_ready_for_training': valid_datasets,
            'datasets_with_issues': total_datasets - valid_datasets,
            'scaling_methods_successful': len(scaling_methods),
            'feature_engineering_complete': valid_datasets > 0
        }
        
        # Determine if ready for ML training
        final_results['ready_for_ml_training'] = (
            valid_datasets > 0 and 
            validation_rate >= 80 and  # At least 80% datasets valid
            len(scaling_methods) > 0   # At least one scaling method successful
        )
        
        # Create output summary
        final_results['output_summary'] = {
            'feature_datasets_created': len(scaling_results.get('output_files', [])),
            'feature_datasets_validated': valid_datasets,
            'total_features_engineered': scaling_results.get('total_features_scaled', 0),
            'scaling_methods_applied': scaling_results.get('scaling_methods_applied', []),
            'ready_for_next_stage': final_results['ready_for_ml_training']
        }
        
        # Log final summary
        custom_logging.info("="*60)
        custom_logging.info("FEATURE ENGINEERING PIPELINE SUMMARY")
        custom_logging.info("="*60)
        custom_logging.info(f"Total datasets processed: {total_datasets}")
        custom_logging.info(f"Valid datasets: {valid_datasets}")
        custom_logging.info(f"Validation rate: {validation_rate:.1f}%")
        custom_logging.info(f"Scaling methods: {list(scaling_methods)}")
        custom_logging.info(f"Ready for ML training: {final_results['ready_for_ml_training']}")
        custom_logging.info("="*60)
        
        if final_results['ready_for_ml_training']:
            custom_logging.info("✅ Feature engineering pipeline completed successfully!")
            custom_logging.info("📊 All datasets are ready for model training")
        else:
            custom_logging.warning("⚠️  Feature engineering completed with issues")
            custom_logging.warning("🔍 Please review dataset validation results")
        
        return final_results
        
    except Exception as e:
        error_message = f"Error in combine_and_scale_features task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)

# Define all preprocessing DAG tasks with detailed configuration



# Task 1: Check for processed data availability
check_data_task = PythonOperator(
    task_id='check_data_task',  # Unique identifier for this task
    python_callable=check_processed_data_availability,  # Function to call when task runs
    dag=dag,  # Associate with our preprocessing DAG
    doc_md="""
    **Check Processed Data Task**
    
    This task checks for the availability of processed user input data from the ingestion pipeline.
    
    Responsibilities:
    - Verify processed data directory exists
    - Check for preprocessed CSV files containing user input data
    - Create features directory structure
    - Return file availability status for downstream tasks
    """,
    retries=3,  # Retry multiple times as we're waiting for upstream data
    retry_delay=timedelta(minutes=10)  # Wait longer between retries
)


# Task 2: Create derived features from user data
create_features_task = PythonOperator(
    task_id='create_features_task',  # Unique identifier
    python_callable=create_derived_features,  # Function to call
    dag=dag,  # Associate with preprocessing DAG
    doc_md="""
    **Create Derived Features Task**
    
    This task creates derived features from the 9 user input columns.
    
    Responsibilities:
    - Create ratio features from user data
    - Generate interaction features between columns
    - Calculate polynomial features
    - Add statistical derived features
    - Save derived feature datasets
    """,
    retries=2,  # Retry if feature creation fails
    retry_delay=timedelta(minutes=5)
)

# Task 3: Apply feature scaling methods
apply_scaling_task = PythonOperator(
    task_id='apply_scaling_task',  # Unique identifier
    python_callable=apply_feature_scaling,  # Function to call
    dag=dag,  # Associate with preprocessing DAG
    doc_md="""
    **Apply Feature Scaling Task**
    
    This task applies different scaling methods to derived features.
    
    Responsibilities:
    - Apply StandardScaler to features
    - Apply MinMaxScaler to features  
    - Apply RobustScaler to features
    - Create separate scaled datasets
    - Prepare data for ML model training
    """,
    retries=2,  # Retry if scaling fails
    retry_delay=timedelta(minutes=5)
)


# Task 4: Final validation and summary
final_validation_task = PythonOperator(
    task_id='final_validation_task',  # Unique identifier
    python_callable=combine_and_scale_features,  # Function to call
    dag=dag,  # Associate with preprocessing DAG
    doc_md="""
    **Final Validation and Summary Task**
    
    This task performs final validation of all scaled feature datasets.
    
    Responsibilities:
    - Validate all scaled feature datasets
    - Perform data quality checks
    - Create comprehensive feature engineering summary
    - Determine readiness for ML model training
    - Generate final pipeline reports
    """,
    retries=2,  # Retry if validation fails
    retry_delay=timedelta(minutes=5)
)


# Define preprocessing task dependencies
# Create a sequential processing structure for user data feature engineering

custom_logging.info("Setting up preprocessing task dependencies")

# Task flow: data check -> feature creation -> scaling -> final validation
# Set task dependencies for Airflow DAG
check_data_task.set_downstream(create_features_task)
create_features_task.set_downstream(apply_scaling_task)
apply_scaling_task.set_downstream(final_validation_task)

# Log preprocessing DAG initialization completion
custom_logging.info("Rockburst data preprocessing DAG initialized successfully")
custom_logging.info("DAG contains 4 tasks with sequential feature engineering pipeline")
custom_logging.info("DAG processes user input data through derivation, scaling, and combination")
custom_logging.info("DAG scheduled to run every 12 hours after data ingestion completes")
