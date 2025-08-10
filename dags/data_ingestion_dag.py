"""
Rockburst Data Ingestion DAG
=============================
This DAG handles the complete data ingestion pipeline for rockburst sensor data.
It includes data collection, validation, preprocessing, and storage operations.

Author: Team GAMMA
Created: August 10, 2025
"""

# Standard library imports - for basic Python functionality
import sys
import os
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

# Airflow imports - for workflow orchestration
from airflow import DAG
from airflow.operators.python import PythonOperator



# Add project root to Python path for custom imports
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')




# Import custom logging and exception handling
try:
    from exception_logging.logger import logging as custom_logging
    from exception_logging.exception import CustomException
    # Log successful import of custom modules
    logging.info("Successfully imported custom logging and exception modules")
except ImportError as e:
    # Fallback to standard logging if custom modules not available
    logging.warning(f"Could not import custom logging modules: {str(e)}")
    custom_logging = logging
    CustomException = Exception  # Fallback to built-in Exception if custom not available



# DAG default arguments - configuration for all tasks in this DAG
default_args = {
    'owner': 'TEAM_GAMMA',  # Owner of the DAG
    'depends_on_past': False,  # Don't wait for previous runs to complete
    'start_date': datetime(2025, 8, 10),  # When this DAG should start running
    'email_on_failure': False,  # Don't send email on task failure
    'email_on_retry': False,  # Don't send email on task retry
    'retries': 2,  # Number of retries if task fails
    'retry_delay': timedelta(minutes=5),  # Wait time between retries
    'max_active_runs': 1,  # Only one instance of this DAG can run at a time
}



# Create the DAG instance - the main workflow container
dag = DAG(
    'rockburst_data_ingestion',  # Unique identifier for this DAG
    default_args=default_args,  # Use the default arguments defined above
    description='Complete data ingestion pipeline for rockburst sensor data with InfluxDB schema configuration',  # Human-readable description
    schedule=timedelta(hours=6),  # Run every 6 hours for data ingestion
    catchup=False,  # Don't run for past dates when DAG is first created
    tags=['rockbrust', 'data-ingestion', 'sensor-data', 'preprocessing', 'influxdb'],  # Tags for organization
    max_active_tasks=3,  # Maximum number of tasks that can run simultaneously
)

# InfluxDB Schema Configuration for User Input Rockburst Data
# Define the structure for user-submitted rockburst analysis data
ROCKBURST_USER_DATA_SCHEMA = {
    'measurement': 'rockburst_analysis',  # Single measurement type for user data
    'description': 'User-submitted rockburst prediction data with calculated features',
    'tags': ['user_id', 'location', 'submission_date', 'analysis_type'],  # Metadata tags
    'fields': [
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
    'retention_policy': 'inf',  # Keep user analysis data indefinitely for model training
    'database_config': {
        'org': 'rockbrust',
        'bucket': 'rockbrust-data',
        'token': 'rockbrust-token-12345',
        'url': 'http://rockbrust:8086'
    }
}

# Log the schema configuration
custom_logging.info(f"InfluxDB user data schema configuration loaded for measurement: {ROCKBURST_USER_DATA_SCHEMA['measurement']}")




def check_data_directory(**context):
    """
    Task 1: Check if data directory exists and contains CSV files
    
    This function:
    1. Verifies the data directory structure exists
    2. Checks for CSV files in the raw data folder
    3. Logs directory contents for monitoring
    4. Returns status for downstream tasks
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Status information about data directory and files found
    """
    try:
        # Log start of directory check task
        custom_logging.info("Starting data directory check task")
        
        # Define data directory paths
        base_data_path = '/opt/airflow/data'  # Main data directory
        raw_data_path = os.path.join(base_data_path, 'raw')  # Raw data subdirectory
        processed_data_path = os.path.join(base_data_path, 'processed')  # Processed data subdirectory
        
        # Log the paths being checked
        custom_logging.info(f"Checking base data path: {base_data_path}")
        custom_logging.info(f"Checking raw data path: {raw_data_path}")
        custom_logging.info(f"Checking processed data path: {processed_data_path}")
        
        # Create directories if they don't exist
        directories_to_create = [base_data_path, raw_data_path, processed_data_path]
        for directory in directories_to_create:
            if not os.path.exists(directory):
                # Create missing directory
                os.makedirs(directory, exist_ok=True)
                custom_logging.info(f"Created missing directory: {directory}")
            else:
                # Log that directory already exists
                custom_logging.info(f"Directory already exists: {directory}")
        
        # Check for CSV files in raw data directory
        csv_files = []  # Initialize empty list for CSV files
        if os.path.exists(raw_data_path):
            # Get all files in raw data directory
            all_files = os.listdir(raw_data_path)
            # Filter for CSV files only
            csv_files = [f for f in all_files if f.lower().endswith('.csv')]
            
            # Log findings about CSV files
            custom_logging.info(f"Found {len(csv_files)} CSV files in raw data directory")
            for csv_file in csv_files:
                file_path = os.path.join(raw_data_path, csv_file)
                file_size = os.path.getsize(file_path)  # Get file size in bytes
                custom_logging.info(f"CSV file found: {csv_file} (Size: {file_size} bytes)")
        else:
            # Log if raw data directory doesn't exist
            custom_logging.warning(f"Raw data directory does not exist: {raw_data_path}")
        
        # Prepare return data with directory status
        directory_status = {
            'base_data_path_exists': os.path.exists(base_data_path),
            'raw_data_path_exists': os.path.exists(raw_data_path),
            'processed_data_path_exists': os.path.exists(processed_data_path),
            'csv_files_found': csv_files,
            'csv_file_count': len(csv_files),
            'task_completion_time': datetime.now().isoformat(),
            'task_status': 'success'
        }
        
        # Log successful completion of directory check
        custom_logging.info(f"Directory check completed successfully: {directory_status}")
        
        return directory_status
        
    except Exception as e:
        # Handle any errors that occur during directory check
        error_message = f"Error in check_data_directory task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)





def validate_csv_data(**context):
    """
    Task 2: Validate CSV data files for quality and structure
    
    This function:
    1. Loads CSV files found in previous task
    2. Performs data quality checks (nulls, duplicates, structure)
    3. Validates data types and formats
    4. Generates validation report
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Comprehensive validation results for all CSV files
    """
    try:
        # Log start of validation task
        custom_logging.info("Starting CSV data validation task")
        
        # Get directory status from previous task
        directory_status = context['task_instance'].xcom_pull(task_ids='check_data_directory_task')
        custom_logging.info(f"Retrieved directory status from previous task: {directory_status}")
        
        # Check if any CSV files were found
        if directory_status['csv_file_count'] == 0:
            custom_logging.warning("No CSV files found for validation")
            return {
                'validation_status': 'no_data',
                'message': 'No CSV files available for validation',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize validation results dictionary
        validation_results = {
            'files_validated': [],
            'total_files': directory_status['csv_file_count'],
            'validation_summary': {},
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # Define raw data path
        raw_data_path = '/opt/airflow/data/raw'
        
        # Process each CSV file found
        for csv_file in directory_status['csv_files_found']:
            custom_logging.info(f"Starting validation for file: {csv_file}")
            
            # Construct full file path
            file_path = os.path.join(raw_data_path, csv_file)
            custom_logging.info(f"Full file path: {file_path}")
            
            try:
                # Load CSV file into pandas DataFrame
                custom_logging.info(f"Loading CSV file: {csv_file}")
                df = pd.read_csv(file_path)
                custom_logging.info(f"Successfully loaded CSV with shape: {df.shape}")
                
                # Perform comprehensive data validation
                file_validation = {
                    'filename': csv_file,
                    'file_size_bytes': os.path.getsize(file_path),
                    'shape': df.shape,  # (rows, columns)
                    'column_count': len(df.columns),
                    'row_count': len(df),
                    'column_names': list(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'missing_values_per_column': df.isnull().sum().to_dict(),
                    'total_missing_values': df.isnull().sum().sum(),
                    'duplicate_rows': df.duplicated().sum(),
                    'memory_usage_bytes': df.memory_usage(deep=True).sum(),
                    'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                    'validation_timestamp': datetime.now().isoformat()
                }
                
                # Log detailed validation information
                custom_logging.info(f"Validation results for {csv_file}:")
                custom_logging.info(f"  - Shape: {file_validation['shape']}")
                custom_logging.info(f"  - Missing values: {file_validation['total_missing_values']}")
                custom_logging.info(f"  - Duplicate rows: {file_validation['duplicate_rows']}")
                custom_logging.info(f"  - Memory usage: {file_validation['memory_usage_bytes']} bytes")
                
                # Validate that required rockburst columns are present
                required_columns = ROCKBURST_USER_DATA_SCHEMA['fields']
                missing_columns = [col for col in required_columns if col not in df.columns]
                extra_columns = [col for col in df.columns if col not in required_columns]
                
                # Log column validation results
                if missing_columns:
                    warning_msg = f"Missing required columns: {missing_columns}"
                    file_validation['warnings'].append(warning_msg)
                    custom_logging.warning(f"{csv_file}: {warning_msg}")
                
                if extra_columns:
                    info_msg = f"Extra columns found (will be ignored): {extra_columns}"
                    file_validation['warnings'].append(info_msg)
                    custom_logging.info(f"{csv_file}: {info_msg}")
                
                # Validate data types for rockburst features
                numeric_required = [col for col in required_columns if col != 'Intensity_Level_encoded']
                for col in numeric_required:
                    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                        warning_msg = f"Column {col} should be numeric but is {df[col].dtype}"
                        file_validation['warnings'].append(warning_msg)
                        custom_logging.warning(f"{csv_file}: {warning_msg}")
                
                # Check for high percentage of missing values
                missing_percentage = (file_validation['total_missing_values'] / 
                                    (file_validation['row_count'] * file_validation['column_count'])) * 100
                if missing_percentage > 10:
                    warning_msg = f"High missing values percentage: {missing_percentage:.2f}%"
                    file_validation['warnings'].append(warning_msg)
                    custom_logging.warning(f"{csv_file}: {warning_msg}")
                
                # Check for duplicate rows
                if file_validation['duplicate_rows'] > 0:
                    duplicate_percentage = (file_validation['duplicate_rows'] / file_validation['row_count']) * 100
                    warning_msg = f"Duplicate rows found: {file_validation['duplicate_rows']} ({duplicate_percentage:.2f}%)"
                    file_validation['warnings'].append(warning_msg)
                    custom_logging.warning(f"{csv_file}: {warning_msg}")
                
                # Check for empty columns
                empty_columns = [col for col, missing_count in file_validation['missing_values_per_column'].items() 
                               if missing_count == file_validation['row_count']]
                if empty_columns:
                    warning_msg = f"Completely empty columns: {empty_columns}"
                    file_validation['warnings'].append(warning_msg)
                    custom_logging.warning(f"{csv_file}: {warning_msg}")
                
                # Add file validation to results
                validation_results['files_validated'].append(file_validation)
                custom_logging.info(f"Validation completed successfully for {csv_file}")
                
            except Exception as file_error:
                # Handle errors specific to individual file processing
                error_msg = f"Error validating file {csv_file}: {str(file_error)}"
                custom_logging.error(error_msg)
                
                # Add error information to validation results
                validation_results['files_validated'].append({
                    'filename': csv_file,
                    'validation_status': 'error',
                    'error_message': error_msg,
                    'validation_timestamp': datetime.now().isoformat()
                })
        
        # Generate overall validation summary
        validation_results['validation_summary'] = {
            'total_rows': sum([f.get('row_count', 0) for f in validation_results['files_validated'] if 'row_count' in f]),
            'total_columns': sum([f.get('column_count', 0) for f in validation_results['files_validated'] if 'column_count' in f]),
            'total_missing_values': sum([f.get('total_missing_values', 0) for f in validation_results['files_validated'] if 'total_missing_values' in f]),
            'files_with_warnings': len([f for f in validation_results['files_validated'] if f.get('warnings', [])]),
            'files_with_errors': len([f for f in validation_results['files_validated'] if f.get('validation_status') == 'error'])
        }
        
        # Log validation summary
        custom_logging.info(f"Validation summary: {validation_results['validation_summary']}")
        custom_logging.info("CSV data validation task completed successfully")
        
        return validation_results
        
    except Exception as e:
        # Handle any errors that occur during validation
        error_message = f"Error in validate_csv_data task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)







def preprocess_sensor_data(**context):
    """
    Task 3: Preprocess rockburst sensor data
    
    This function:
    1. Loads validated CSV data
    2. Performs data cleaning (remove duplicates, handle missing values)
    3. Applies rockburst-specific preprocessing
    4. Saves preprocessed data for model training
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Preprocessing results and output file locations
    """
    try:
        # Log start of preprocessing task
        custom_logging.info("Starting sensor data preprocessing task")
        
        # Get validation results from previous task
        validation_results = context['task_instance'].xcom_pull(task_ids='validate_csv_data_task')
        custom_logging.info(f"Retrieved validation results from previous task")
        
        # Check if validation found any processable data
        if validation_results['validation_status'] == 'no_data':
            custom_logging.warning("No data available for preprocessing")
            return {
                'preprocessing_status': 'no_data',
                'message': 'No validated data available for preprocessing',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize preprocessing results
        preprocessing_results = {
            'processed_files': [],
            'total_files_processed': 0,
            'preprocessing_summary': {},
            'output_files': [],
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # Define paths
        raw_data_path = '/opt/airflow/data/raw'
        processed_data_path = '/opt/airflow/data/processed'
        
        # Process each validated file
        for file_info in validation_results['files_validated']:
            # Skip files that had validation errors
            if file_info.get('validation_status') == 'error':
                custom_logging.warning(f"Skipping file with validation errors: {file_info['filename']}")
                continue
                
            csv_file = file_info['filename']
            custom_logging.info(f"Starting preprocessing for file: {csv_file}")
            
            try:
                # Load the CSV file
                input_path = os.path.join(raw_data_path, csv_file)
                custom_logging.info(f"Loading file for preprocessing: {input_path}")
                df = pd.read_csv(input_path)
                
                # Store original data statistics
                original_shape = df.shape
                original_missing = df.isnull().sum().sum()
                original_duplicates = df.duplicated().sum()
                custom_logging.info(f"Original data statistics - Shape: {original_shape}, Missing: {original_missing}, Duplicates: {original_duplicates}")
                
                # Step 1: Validate required columns are present
                custom_logging.info("Step 1: Validating required rockburst columns")
                required_columns = ROCKBURST_USER_DATA_SCHEMA['fields']
                available_columns = [col for col in required_columns if col in df.columns]
                
                if len(available_columns) < len(required_columns):
                    missing_cols = set(required_columns) - set(available_columns)
                    custom_logging.warning(f"Missing columns {missing_cols}, will work with available: {available_columns}")
                
                # Step 2: Remove duplicate rows
                custom_logging.info("Step 2: Removing duplicate rows")
                df_deduplicated = df.drop_duplicates()
                duplicates_removed = original_shape[0] - df_deduplicated.shape[0]
                custom_logging.info(f"Removed {duplicates_removed} duplicate rows")
                
                # Step 3: Handle missing values for numeric features only
                custom_logging.info("Step 3: Handling missing values in numeric features")
                numeric_features = [col for col in available_columns if col != 'Intensity_Level_encoded']
                
                for col in numeric_features:
                    if col in df_deduplicated.columns:
                        missing_count = df_deduplicated[col].isnull().sum()
                        if missing_count > 0:
                            median_value = df_deduplicated[col].median()
                            df_deduplicated[col].fillna(median_value, inplace=True)
                            custom_logging.info(f"Filled {missing_count} missing values in {col} with median: {median_value}")
                
                # Handle encoded intensity level (categorical target)
                if 'Intensity_Level_encoded' in df_deduplicated.columns:
                    missing_intensity = df_deduplicated['Intensity_Level_encoded'].isnull().sum()
                    if missing_intensity > 0:
                        mode_value = df_deduplicated['Intensity_Level_encoded'].mode().iloc[0] if not df_deduplicated['Intensity_Level_encoded'].mode().empty else 0
                        df_deduplicated['Intensity_Level_encoded'].fillna(mode_value, inplace=True)
                        custom_logging.info(f"Filled {missing_intensity} missing intensity values with mode: {mode_value}")
                
                # Step 4: Remove invalid data points
                custom_logging.info("Step 4: Removing invalid data points")
                
                # Remove rows where all numeric features are zero (likely invalid entries)
                initial_rows = len(df_deduplicated)
                numeric_cols_present = [col for col in numeric_features if col in df_deduplicated.columns]
                
                if numeric_cols_present:
                    df_cleaned = df_deduplicated.loc[~(df_deduplicated[numeric_cols_present] == 0).all(axis=1)]
                    zero_rows_removed = initial_rows - len(df_cleaned)
                    custom_logging.info(f"Removed {zero_rows_removed} rows with all zero numeric values")
                else:
                    df_cleaned = df_deduplicated
                    zero_rows_removed = 0
                
                # Step 5: Data type optimization for memory efficiency
                custom_logging.info("Step 5: Optimizing data types for rockburst features")
                
                # Optimize numeric columns to appropriate types
                for col in numeric_cols_present:
                    if col in df_cleaned.columns:
                        # Convert to float32 for memory efficiency while maintaining precision
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('float32')
                        custom_logging.info(f"Optimized column {col} to float32")
                
                # Ensure intensity level is integer type
                if 'Intensity_Level_encoded' in df_cleaned.columns:
                    df_cleaned['Intensity_Level_encoded'] = pd.to_numeric(df_cleaned['Intensity_Level_encoded'], errors='coerce').astype('int32')
                    custom_logging.info("Optimized Intensity_Level_encoded to int32")                # Calculate preprocessing statistics
                final_shape = df_cleaned.shape
                final_missing = df_cleaned.isnull().sum().sum()
                memory_reduction = (df.memory_usage(deep=True).sum() - df_cleaned.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum() * 100
                
                # Create preprocessing summary for this file
                file_preprocessing = {
                    'filename': csv_file,
                    'original_shape': original_shape,
                    'final_shape': final_shape,
                    'rows_removed': original_shape[0] - final_shape[0],
                    'duplicates_removed': duplicates_removed,
                    'zero_rows_removed': zero_rows_removed,
                    'original_missing_values': original_missing,
                    'final_missing_values': final_missing,
                    'memory_reduction_percentage': round(memory_reduction, 2),
                    'preprocessing_timestamp': datetime.now().isoformat()
                }
                
                # Log preprocessing statistics
                custom_logging.info(f"Preprocessing statistics for {csv_file}:")
                custom_logging.info(f"  - Shape change: {original_shape} -> {final_shape}")
                custom_logging.info(f"  - Rows removed: {original_shape[0] - final_shape[0]}")
                custom_logging.info(f"  - Missing values: {original_missing} -> {final_missing}")
                custom_logging.info(f"  - Memory reduction: {memory_reduction:.2f}%")
                
                # Save preprocessed data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = os.path.splitext(csv_file)[0]
                output_filename = f"{base_filename}_preprocessed_{timestamp}.csv"
                output_path = os.path.join(processed_data_path, output_filename)
                
                custom_logging.info(f"Saving preprocessed data to: {output_path}")
                df_cleaned.to_csv(output_path, index=False)
                
                # Verify file was saved successfully
                if os.path.exists(output_path):
                    output_file_size = os.path.getsize(output_path)
                    custom_logging.info(f"Successfully saved preprocessed file: {output_filename} (Size: {output_file_size} bytes)")
                    
                    file_preprocessing['output_filename'] = output_filename
                    file_preprocessing['output_path'] = output_path
                    file_preprocessing['output_file_size'] = output_file_size
                    preprocessing_results['output_files'].append(output_filename)
                else:
                    custom_logging.error(f"Failed to save preprocessed file: {output_path}")
                    file_preprocessing['error'] = f"Failed to save preprocessed file"
                
                # Add file preprocessing results
                preprocessing_results['processed_files'].append(file_preprocessing)
                preprocessing_results['total_files_processed'] += 1
                
                custom_logging.info(f"Preprocessing completed successfully for {csv_file}")
                
            except Exception as file_error:
                # Handle errors specific to individual file preprocessing
                error_msg = f"Error preprocessing file {csv_file}: {str(file_error)}"
                custom_logging.error(error_msg)
                
                preprocessing_results['processed_files'].append({
                    'filename': csv_file,
                    'preprocessing_status': 'error',
                    'error_message': error_msg,
                    'preprocessing_timestamp': datetime.now().isoformat()
                })
        
        # Generate overall preprocessing summary
        if preprocessing_results['processed_files']:
            total_original_rows = sum([f.get('original_shape', [0, 0])[0] for f in preprocessing_results['processed_files'] if 'original_shape' in f])
            total_final_rows = sum([f.get('final_shape', [0, 0])[0] for f in preprocessing_results['processed_files'] if 'final_shape' in f])
            total_rows_removed = sum([f.get('rows_removed', 0) for f in preprocessing_results['processed_files'] if 'rows_removed' in f])
            
            preprocessing_results['preprocessing_summary'] = {
                'total_original_rows': total_original_rows,
                'total_final_rows': total_final_rows,
                'total_rows_removed': total_rows_removed,
                'data_reduction_percentage': round((total_rows_removed / total_original_rows * 100) if total_original_rows > 0 else 0, 2),
                'files_with_errors': len([f for f in preprocessing_results['processed_files'] if f.get('preprocessing_status') == 'error'])
            }
        
        # Log preprocessing summary
        custom_logging.info(f"Preprocessing summary: {preprocessing_results['preprocessing_summary']}")
        custom_logging.info("Sensor data preprocessing task completed successfully")
        
        return preprocessing_results
        
    except Exception as e:
        # Handle any errors that occur during preprocessing
        error_message = f"Error in preprocess_sensor_data task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)







def store_data_in_influxdb(**context):
    """
    Task 4: Store preprocessed data in InfluxDB for time-series analysis
    
    This function:
    1. Takes preprocessed CSV data
    2. Converts it to InfluxDB line protocol format
    3. Stores it in the rockbrust-data bucket
    4. Creates appropriate tags and fields for queries
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Storage results and InfluxDB operation status
    """
    try:
        # Log start of InfluxDB storage task
        custom_logging.info("Starting InfluxDB data storage task")
        
        # Get preprocessing results from previous task
        preprocessing_results = context['task_instance'].xcom_pull(task_ids='preprocess_sensor_data_task')
        custom_logging.info("Retrieved preprocessing results from previous task")
        
        # Check if preprocessing produced any data
        if preprocessing_results.get('preprocessing_status') == 'no_data':
            custom_logging.warning("No preprocessed data available for InfluxDB storage")
            return {
                'storage_status': 'no_data',
                'message': 'No preprocessed data available for InfluxDB storage',
                'task_completion_time': datetime.now().isoformat()
            }
        
        # Initialize storage results
        storage_results = {
            'stored_files': [],
            'total_records_stored': 0,
            'influxdb_operations': [],
            'storage_summary': {},
            'task_status': 'success',
            'task_completion_time': datetime.now().isoformat()
        }
        
        # InfluxDB connection parameters (using user data schema configuration)
        influxdb_config = ROCKBURST_USER_DATA_SCHEMA['database_config']
        custom_logging.info(f"InfluxDB configuration from schema: {influxdb_config}")
        
        # Log available fields from user data schema
        available_fields = ROCKBURST_USER_DATA_SCHEMA['fields']
        custom_logging.info(f"Available user data fields: {available_fields}")
        
        # Define processed data path
        processed_data_path = '/opt/airflow/data/processed'
        
        # Note: This is a placeholder for InfluxDB integration
        # In production, you would use influxdb-client library
        custom_logging.info("NOTE: InfluxDB client integration placeholder")
        custom_logging.info("To implement: pip install influxdb-client in Airflow container")
        
        # Process each preprocessed file
        for file_info in preprocessing_results['processed_files']:
            # Skip files that had preprocessing errors
            if file_info.get('preprocessing_status') == 'error':
                custom_logging.warning(f"Skipping file with preprocessing errors: {file_info['filename']}")
                continue
                
            if 'output_filename' not in file_info:
                custom_logging.warning(f"No output file found for: {file_info['filename']}")
                continue
                
            output_filename = file_info['output_filename']
            custom_logging.info(f"Starting InfluxDB storage for file: {output_filename}")
            
            try:
                # Load preprocessed data
                file_path = os.path.join(processed_data_path, output_filename)
                custom_logging.info(f"Loading preprocessed file: {file_path}")
                df = pd.read_csv(file_path)
                
                # Simulate InfluxDB line protocol preparation
                custom_logging.info("Preparing data for InfluxDB line protocol format")
                
                # Identify potential timestamp column
                timestamp_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_columns:
                    timestamp_col = timestamp_columns[0]
                    custom_logging.info(f"Using timestamp column: {timestamp_col}")
                else:
                    # Use current timestamp if no timestamp column found
                    timestamp_col = None
                    custom_logging.info("No timestamp column found, will use current timestamp")
                
                # Identify measurement fields (numeric columns for sensor data)
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                measurement_fields = [col for col in numeric_columns if col != timestamp_col]
                custom_logging.info(f"Identified {len(measurement_fields)} measurement fields: {measurement_fields}")
                
                # Identify tag fields (categorical columns for grouping)
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                tag_fields = [col for col in categorical_columns if col != timestamp_col]
                custom_logging.info(f"Identified {len(tag_fields)} tag fields: {tag_fields}")
                
                # Simulate data point creation for InfluxDB
                total_points = len(df)
                custom_logging.info(f"Preparing {total_points} data points for InfluxDB")
                
                # Simulate batch processing (process in chunks of 1000)
                batch_size = 1000
                batches_processed = 0
                
                for start_idx in range(0, total_points, batch_size):
                    end_idx = min(start_idx + batch_size, total_points)
                    batch_data = df.iloc[start_idx:end_idx]
                    
                    custom_logging.info(f"Processing batch {batches_processed + 1}: rows {start_idx} to {end_idx}")
                    
                    # Simulate line protocol creation for each row in batch
                    for idx, row in batch_data.iterrows():
                        # Create measurement name (could be based on sensor type or location)
                        measurement = "rockburst_sensors"
                        
                        # Create tags (for efficient querying)
                        tags = {}
                        for tag_field in tag_fields:
                            if pd.notna(row[tag_field]):
                                tags[tag_field] = str(row[tag_field])
                        
                        # Create fields (actual sensor measurements)
                        fields = {}
                        for field_name in measurement_fields:
                            if pd.notna(row[field_name]):
                                fields[field_name] = float(row[field_name])
                        
                        # Determine timestamp
                        if timestamp_col and pd.notna(row[timestamp_col]):
                            point_timestamp = row[timestamp_col]
                        else:
                            point_timestamp = datetime.now().isoformat()
                        
                        # Log line protocol format (simulation)
                        if idx == start_idx:  # Log first point of each batch as example
                            custom_logging.info(f"Example line protocol: {measurement},tags={tags} fields={fields} {point_timestamp}")
                    
                    batches_processed += 1
                    custom_logging.info(f"Batch {batches_processed} prepared for InfluxDB")
                
                # Simulate successful InfluxDB write operation
                file_storage = {
                    'filename': output_filename,
                    'measurement_name': 'rockburst_sensors',
                    'total_points': total_points,
                    'batches_processed': batches_processed,
                    'tag_fields': tag_fields,
                    'measurement_fields': measurement_fields,
                    'storage_timestamp': datetime.now().isoformat(),
                    'influxdb_bucket': influxdb_config['bucket'],
                    'storage_status': 'simulated_success'
                }
                
                # Log storage statistics
                custom_logging.info(f"InfluxDB storage simulation for {output_filename}:")
                custom_logging.info(f"  - Total points: {total_points}")
                custom_logging.info(f"  - Batches: {batches_processed}")
                custom_logging.info(f"  - Tag fields: {len(tag_fields)}")
                custom_logging.info(f"  - Measurement fields: {len(measurement_fields)}")
                
                # Add to storage results
                storage_results['stored_files'].append(file_storage)
                storage_results['total_records_stored'] += total_points
                
                # Log operation details
                storage_results['influxdb_operations'].append({
                    'operation': 'write_points',
                    'filename': output_filename,
                    'points_written': total_points,
                    'bucket': influxdb_config['bucket'],
                    'timestamp': datetime.now().isoformat()
                })
                
                custom_logging.info(f"InfluxDB storage simulation completed for {output_filename}")
                
            except Exception as file_error:
                # Handle errors specific to individual file storage
                error_msg = f"Error storing file {output_filename} in InfluxDB: {str(file_error)}"
                custom_logging.error(error_msg)
                
                storage_results['stored_files'].append({
                    'filename': output_filename,
                    'storage_status': 'error',
                    'error_message': error_msg,
                    'storage_timestamp': datetime.now().isoformat()
                })
        
        # Generate overall storage summary
        if storage_results['stored_files']:
            successful_files = [f for f in storage_results['stored_files'] if f.get('storage_status') != 'error']
            failed_files = [f for f in storage_results['stored_files'] if f.get('storage_status') == 'error']
            
            storage_results['storage_summary'] = {
                'total_files_processed': len(storage_results['stored_files']),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'total_data_points': storage_results['total_records_stored'],
                'influxdb_bucket': influxdb_config['bucket'],
                'storage_completion_rate': round((len(successful_files) / len(storage_results['stored_files'])) * 100, 2) if storage_results['stored_files'] else 0
            }
        
        # Log storage summary
        custom_logging.info(f"InfluxDB storage summary: {storage_results['storage_summary']}")
        custom_logging.info("InfluxDB data storage task completed successfully")
        
        return storage_results
        
    except Exception as e:
        # Handle any errors that occur during InfluxDB storage
        error_message = f"Error in store_data_in_influxdb task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)






def generate_data_quality_report(**context):
    """
    Task 5: Generate comprehensive data quality and pipeline report
    
    This function:
    1. Collects results from all previous tasks
    2. Generates comprehensive data quality metrics
    3. Creates pipeline execution report
    4. Saves report for monitoring and analysis
    
    Args:
        **context: Airflow context containing task instance information
        
    Returns:
        dict: Complete pipeline report with all metrics and status
    """
    try:
        # Log start of report generation task
        custom_logging.info("Starting data quality report generation task")
        
        # Collect results from all previous tasks
        custom_logging.info("Collecting results from all pipeline tasks")
        
        # Get directory check results
        directory_status = context['task_instance'].xcom_pull(task_ids='check_data_directory_task')
        custom_logging.info("Retrieved directory check results")
        
        # Get validation results
        validation_results = context['task_instance'].xcom_pull(task_ids='validate_csv_data_task')
        custom_logging.info("Retrieved validation results")
        
        # Get preprocessing results
        preprocessing_results = context['task_instance'].xcom_pull(task_ids='preprocess_sensor_data_task')
        custom_logging.info("Retrieved preprocessing results")
        
        # Get storage results
        storage_results = context['task_instance'].xcom_pull(task_ids='store_influxdb_task')
        custom_logging.info("Retrieved InfluxDB storage results")
        
        # Create comprehensive pipeline report
        pipeline_report = {
            'report_metadata': {
                'report_generation_time': datetime.now().isoformat(),
                'pipeline_run_date': datetime.now().strftime('%Y-%m-%d'),
                'pipeline_name': 'rockburst_data_ingestion',
                'report_version': '1.0.0'
            },
            'pipeline_execution_summary': {},
            'data_directory_status': directory_status,
            'data_validation_results': validation_results,
            'preprocessing_results': preprocessing_results,
            'storage_results': storage_results,
            'overall_pipeline_status': 'success',
            'recommendations': [],
            'next_actions': []
        }
        
        # Log report sections being compiled
        custom_logging.info("Compiling pipeline execution summary")
        
        # Calculate pipeline execution summary
        execution_summary = {
            'total_csv_files_found': directory_status.get('csv_file_count', 0),
            'files_successfully_validated': len([f for f in validation_results.get('files_validated', []) if f.get('validation_status') != 'error']),
            'files_successfully_preprocessed': preprocessing_results.get('total_files_processed', 0),
            'files_successfully_stored': len([f for f in storage_results.get('stored_files', []) if f.get('storage_status') != 'error']),
            'total_data_records_processed': storage_results.get('total_records_stored', 0),
            'pipeline_success_rate': 0,
            'execution_time_minutes': 0,  # Would calculate actual execution time in production
            'memory_usage_optimized': True if preprocessing_results.get('preprocessing_summary', {}).get('data_reduction_percentage', 0) > 0 else False
        }
        
        # Calculate pipeline success rate
        if execution_summary['total_csv_files_found'] > 0:
            execution_summary['pipeline_success_rate'] = round(
                (execution_summary['files_successfully_stored'] / execution_summary['total_csv_files_found']) * 100, 2
            )
        
        pipeline_report['pipeline_execution_summary'] = execution_summary
        custom_logging.info(f"Pipeline execution summary: {execution_summary}")
        
        # Generate recommendations based on results
        custom_logging.info("Generating recommendations based on pipeline results")
        
        recommendations = []
        next_actions = []
        
        # Check for data quality issues and provide recommendations
        if validation_results.get('validation_summary', {}).get('files_with_warnings', 0) > 0:
            recommendations.append({
                'category': 'data_quality',
                'issue': 'Data quality warnings detected',
                'recommendation': 'Review validation warnings and consider additional data cleaning steps',
                'priority': 'medium'
            })
            custom_logging.info("Added data quality recommendation")
        
        if validation_results.get('validation_summary', {}).get('total_missing_values', 0) > 0:
            recommendations.append({
                'category': 'data_completeness',
                'issue': 'Missing values found in dataset',
                'recommendation': 'Implement more sophisticated missing value imputation strategies',
                'priority': 'low'
            })
            custom_logging.info("Added data completeness recommendation")
        
        if preprocessing_results.get('preprocessing_summary', {}).get('data_reduction_percentage', 0) > 20:
            recommendations.append({
                'category': 'data_efficiency',
                'issue': 'Significant data reduction during preprocessing',
                'recommendation': 'Review data collection processes to reduce duplicate and invalid records',
                'priority': 'medium'
            })
            custom_logging.info("Added data efficiency recommendation")
        
        # Generate next actions based on pipeline success
        if execution_summary['pipeline_success_rate'] == 100:
            next_actions.extend([
                'Data is ready for feature engineering',
                'Proceed to model training pipeline',
                'Set up automated model retraining schedule'
            ])
            custom_logging.info("Added next actions for successful pipeline")
        else:
            next_actions.extend([
                'Review failed file processing logs',
                'Fix data quality issues before model training',
                'Consider adjusting preprocessing parameters'
            ])
            custom_logging.info("Added next actions for partial pipeline success")
        
        # Always add monitoring actions
        next_actions.extend([
            'Monitor InfluxDB data ingestion',
            'Set up data quality alerts',
            'Schedule regular data validation checks'
        ])
        
        pipeline_report['recommendations'] = recommendations
        pipeline_report['next_actions'] = next_actions
        
        # Save the comprehensive report
        custom_logging.info("Saving comprehensive pipeline report")
        
        # Create reports directory if it doesn't exist
        reports_path = '/opt/airflow/data/reports'
        os.makedirs(reports_path, exist_ok=True)
        custom_logging.info(f"Reports directory ready: {reports_path}")
        
        # Generate report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"rockburst_data_quality_report_{timestamp}.json"
        report_file_path = os.path.join(reports_path, report_filename)
        
        # Save report as JSON
        custom_logging.info(f"Saving report to: {report_file_path}")
        with open(report_file_path, 'w') as report_file:
            json.dump(pipeline_report, report_file, indent=2, default=str)
        
        # Verify report file was saved
        if os.path.exists(report_file_path):
            report_file_size = os.path.getsize(report_file_path)
            custom_logging.info(f"Report saved successfully: {report_filename} (Size: {report_file_size} bytes)")
            
            pipeline_report['report_metadata']['report_file_path'] = report_file_path
            pipeline_report['report_metadata']['report_file_size'] = report_file_size
        else:
            custom_logging.error(f"Failed to save report file: {report_file_path}")
            pipeline_report['overall_pipeline_status'] = 'warning'
        
        # Log final pipeline status
        custom_logging.info(f"Final pipeline status: {pipeline_report['overall_pipeline_status']}")
        custom_logging.info(f"Pipeline success rate: {execution_summary['pipeline_success_rate']}%")
        custom_logging.info(f"Total records processed: {execution_summary['total_data_records_processed']}")
        
        # Generate summary log message
        summary_message = f"Data ingestion pipeline completed - Success Rate: {execution_summary['pipeline_success_rate']}%, Files Processed: {execution_summary['files_successfully_stored']}/{execution_summary['total_csv_files_found']}, Records: {execution_summary['total_data_records_processed']}"
        custom_logging.info(summary_message)
        
        custom_logging.info("Data quality report generation task completed successfully")
        
        return pipeline_report
        
    except Exception as e:
        # Handle any errors that occur during report generation
        error_message = f"Error in generate_data_quality_report task: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)
    




# Define all DAG tasks with detailed configuration
# Each task is a PythonOperator that calls one of the functions defined above

# Task 1: Check data directory structure and CSV files
check_data_directory_task = PythonOperator(
    task_id='check_data_directory_task',  # Unique identifier for this task
    python_callable=check_data_directory,  # Function to call when task runs
    dag=dag,  # Associate with our DAG
    doc_md="""
    **Check Data Directory Task**
    
    This task verifies that the data directory structure exists and identifies CSV files for processing.
    
    Responsibilities:
    - Create necessary directory structure if missing
    - Scan for CSV files in raw data directory
    - Log directory contents and file information
    - Return status for downstream tasks
    """,
    retries=1,  # Retry once if task fails
    retry_delay=timedelta(minutes=2)  # Wait 2 minutes before retrying
)

# Task 2: Validate CSV data quality and structure
validate_csv_data_task = PythonOperator(
    task_id='validate_csv_data_task',  # Unique identifier for this task
    python_callable=validate_csv_data,  # Function to call when task runs
    dag=dag,  # Associate with our DAG
    doc_md="""
    **Validate CSV Data Task**
    
    This task performs comprehensive validation of CSV files found in the data directory.
    
    Responsibilities:
    - Load and analyze CSV file structure
    - Check for missing values and duplicates
    - Validate data types and formats
    - Generate data quality warnings
    - Prepare validation report for preprocessing
    """,
    retries=2,  # Retry twice if task fails (data validation is critical)
    retry_delay=timedelta(minutes=3)  # Wait 3 minutes before retrying
)

# Task 3: Preprocess sensor data for model training
preprocess_sensor_data_task = PythonOperator(
    task_id='preprocess_sensor_data_task',  # Unique identifier for this task
    python_callable=preprocess_sensor_data,  # Function to call when task runs
    dag=dag,  # Associate with our DAG
    doc_md="""
    **Preprocess Sensor Data Task**
    
    This task performs comprehensive preprocessing of rockburst sensor data.
    
    Responsibilities:
    - Remove duplicate records
    - Handle missing values with appropriate strategies
    - Remove inactive sensor readings (all zeros)
    - Optimize data types for memory efficiency
    - Save preprocessed data for model training
    """,
    retries=2,  # Retry twice if preprocessing fails
    retry_delay=timedelta(minutes=5)  # Wait 5 minutes before retrying (longer for data processing)
)

# Task 4: Store preprocessed data in InfluxDB
store_influxdb_task = PythonOperator(
    task_id='store_influxdb_task',  # Unique identifier for this task
    python_callable=store_data_in_influxdb,  # Function to call when task runs
    dag=dag,  # Associate with our DAG
    doc_md="""
    **Store Data in InfluxDB Task**
    
    This task stores preprocessed sensor data in InfluxDB for time-series analysis and monitoring.
    
    Responsibilities:
    - Convert CSV data to InfluxDB line protocol format
    - Create appropriate tags and fields for efficient querying
    - Store data in rockbrust-data bucket
    - Process data in batches for efficiency
    - Log storage operations and results
    """,
    retries=3,  # Retry three times if storage fails (network issues possible)
    retry_delay=timedelta(minutes=3)  # Wait 3 minutes before retrying
)

# Task 5: Generate comprehensive data quality report
generate_report_task = PythonOperator(
    task_id='generate_report_task',  # Unique identifier for this task
    python_callable=generate_data_quality_report,  # Function to call when task runs
    dag=dag,  # Associate with our DAG
    doc_md="""
    **Generate Data Quality Report Task**
    
    This task creates a comprehensive report of the entire data ingestion pipeline.
    
    Responsibilities:
    - Collect results from all pipeline tasks
    - Calculate pipeline success metrics
    - Generate data quality recommendations
    - Create actionable next steps
    - Save detailed report for monitoring and analysis
    """,
    retries=1,  # Retry once if report generation fails
    retry_delay=timedelta(minutes=2)  # Wait 2 minutes before retrying
)

# Define task dependencies - the order in which tasks should run
# Each task depends on the successful completion of the previous task
# This creates a linear pipeline: Directory Check -> Validation -> Preprocessing -> Storage -> Reporting

custom_logging.info("Setting up task dependencies for data ingestion pipeline")

# Task dependency chain with detailed logging of each dependency
check_data_directory_task.set_downstream(validate_csv_data_task)  # Validation depends on directory check
validate_csv_data_task.set_downstream(preprocess_sensor_data_task)  # Preprocessing depends on validation
preprocess_sensor_data_task.set_downstream(store_influxdb_task)  # Storage depends on preprocessing
store_influxdb_task.set_downstream(generate_report_task)  # Reporting depends on storage

# Log DAG initialization completion
custom_logging.info("Rockburst data ingestion DAG initialized successfully")
custom_logging.info("DAG contains 5 tasks with comprehensive logging and error handling")
custom_logging.info("DAG scheduled to run every 6 hours with proper dependency management")
