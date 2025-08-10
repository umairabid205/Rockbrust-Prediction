"""
Advanced Feature Engineering for Rockburst Prediction
======================================================
This module provides comprehensive feature engineering capabilities for rockburst prediction
using geological and seismic data. It creates domain-specific features based on geological
expertise and mining engineering principles.

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

# Standard library imports
import sys
import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import joblib

# Scientific computing imports
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Machine learning imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer, KNNImputer

# Add project root to Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/dags')

# Import custom logging and exception handling
try:
    from exception_logging.logger import logging as custom_logging
    from exception_logging.exception import CustomException
    custom_logging.info("Successfully imported custom logging for feature engineering")
except ImportError as e:
    logging.warning(f"Could not import custom logging modules: {str(e)}")
    custom_logging = logging
    CustomException = Exception

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RockburstFeatureEngineering:
    """
    Advanced feature engineering class for rockburst prediction.
    
    This class provides comprehensive feature engineering capabilities specifically
    designed for geological and seismic data analysis in mining environments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RockburstFeatureEngineering class.
        
        Args:
            config: Optional configuration dictionary for feature engineering parameters
        """
        # Log initialization start
        custom_logging.info("Initializing RockburstFeatureEngineering class")
        
        # Set default configuration
        self.config = config or self._get_default_config()
        
        # Initialize scalers and transformers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(), 
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        # Initialize feature selectors
        self.feature_selectors = {
            'f_classif': SelectKBest(score_func=f_classif),
            'mutual_info': SelectKBest(score_func=mutual_info_classif)
        }
        
        # Initialize dimensionality reduction methods
        self.dim_reducers = {
            'pca': PCA(),
            'ica': FastICA(random_state=42),
            'tsne': TSNE(random_state=42, n_jobs=-1)
        }
        
        # Initialize imputers
        self.imputers = {
            'simple_mean': SimpleImputer(strategy='mean'),
            'simple_median': SimpleImputer(strategy='median'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        # Initialize feature metadata
        self.feature_metadata = {}
        self.transformation_history = []
        
        custom_logging.info("RockburstFeatureEngineering initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for feature engineering.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        return {
            'geological_features': {
                'stress_ratios': True,
                'energy_density_features': True,
                'frequency_domain_features': True,
                'cumulative_features': True,
                'stability_indicators': True
            },
            'seismic_features': {
                'magnitude_statistics': True,
                'event_clustering': True,
                'temporal_patterns': True,
                'spectral_analysis': True,
                'peak_detection': True
            },
            'temporal_features': {
                'rolling_windows': [3, 7, 14, 30],  # days
                'lag_features': [1, 2, 3, 7],  # days
                'difference_features': [1, 7],  # days
                'seasonal_features': True
            },
            'interaction_features': {
                'polynomial_degree': 2,
                'cross_products': True,
                'ratio_features': True
            },
            'scaling_methods': ['standard', 'robust'],
            'feature_selection': {
                'methods': ['f_classif', 'mutual_info'],
                'top_k_features': 50
            },
            'dimensionality_reduction': {
                'pca_components': 0.95,  # Retain 95% variance
                'ica_components': 20
            }
        }
    
    def create_geological_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create geological and mining-specific features.
        
        Args:
            df: Input DataFrame with geological data
            target_col: Name of target column (optional)
            
        Returns:
            DataFrame with additional geological features
        """
        try:
            custom_logging.info("Creating geological features for rockburst prediction")
            
            # Create a copy to avoid modifying original data
            feature_df = df.copy()
            original_columns = feature_df.columns.tolist()
            
            # Define expected user data columns based on our schema
            user_columns = {
                'Duration_days': 'Duration_days',
                'Energy_Unit_log': 'Energy_Unit_log', 
                'Energy_density_Joule_sqr': 'Energy_density_Joule_sqr',
                'Volume_m3_sqr': 'Volume_m3_sqr',
                'Event_freq_unit_per_day_log': 'Event_freq_unit_per_day_log',
                'Energy_Joule_per_day_sqr': 'Energy_Joule_per_day_sqr',
                'Volume_m3_per_day_sqr': 'Volume_m3_per_day_sqr',
                'Energy_per_Volume_log': 'Energy_per_Volume_log',
                'Intensity_Level_encoded': 'Intensity_Level_encoded'
            }
            
            # Verify required columns exist
            available_cols = [col for col in user_columns.values() if col in feature_df.columns]
            custom_logging.info(f"Found {len(available_cols)} user data columns for geological feature creation")
            
            if len(available_cols) < 5:  # Minimum required columns
                custom_logging.warning("Insufficient columns for geological feature engineering")
                return feature_df
            
            # 1. Energy-based geological features
            if self.config['geological_features']['energy_density_features']:
                custom_logging.info("Creating energy-based geological features")
                
                if 'Energy_Unit_log' in feature_df.columns and 'Volume_m3_sqr' in feature_df.columns:
                    # Energy per volume ratios (already have Energy_per_Volume_log)
                    feature_df['Energy_Volume_Ratio_Enhanced'] = (
                        feature_df['Energy_Unit_log'] / (feature_df['Volume_m3_sqr'] + 1e-8)
                    )
                    
                    # Energy density indicators
                    feature_df['High_Energy_Density_Flag'] = (
                        feature_df['Energy_density_Joule_sqr'] > 
                        feature_df['Energy_density_Joule_sqr'].quantile(0.75)
                    ).astype(int)
                    
                    # Normalized energy features
                    energy_median = feature_df['Energy_Unit_log'].median()
                    feature_df['Energy_Deviation_From_Median'] = feature_df['Energy_Unit_log'] - energy_median
                    feature_df['Energy_Normalized_By_Volume'] = (
                        feature_df['Energy_Unit_log'] / (feature_df['Volume_m3_sqr'].mean() + 1e-8)
                    )
            
            # 2. Stress accumulation and release patterns
            if self.config['geological_features']['stress_ratios']:
                custom_logging.info("Creating stress accumulation features")
                
                if 'Duration_days' in feature_df.columns and 'Energy_Joule_per_day_sqr' in feature_df.columns:
                    # Stress accumulation rate
                    feature_df['Stress_Accumulation_Rate'] = (
                        feature_df['Energy_Joule_per_day_sqr'] / (feature_df['Duration_days'] + 1e-8)
                    )
                    
                    # Cumulative stress indicators
                    feature_df['Cumulative_Energy_Density'] = feature_df['Energy_density_Joule_sqr'].cumsum()
                    feature_df['Cumulative_Volume'] = feature_df['Volume_m3_sqr'].cumsum()
                    
                    # Stress release patterns
                    if len(feature_df) > 1:
                        feature_df['Energy_Release_Rate'] = feature_df['Energy_Unit_log'].diff()
                        feature_df['Volume_Change_Rate'] = feature_df['Volume_m3_sqr'].diff()
                        
                        # Fill first NaN values
                        feature_df['Energy_Release_Rate'].fillna(0, inplace=True)
                        feature_df['Volume_Change_Rate'].fillna(0, inplace=True)
            
            # 3. Frequency and event-based features
            if self.config['geological_features']['frequency_domain_features']:
                custom_logging.info("Creating frequency and event-based features")
                
                if 'Event_freq_unit_per_day_log' in feature_df.columns:
                    # Event frequency patterns
                    feature_df['High_Frequency_Events'] = (
                        feature_df['Event_freq_unit_per_day_log'] > 
                        feature_df['Event_freq_unit_per_day_log'].quantile(0.8)
                    ).astype(int)
                    
                    # Event intensity correlations
                    if 'Intensity_Level_encoded' in feature_df.columns:
                        # Calculate correlation between frequency and intensity (per row)
                        feature_df['Freq_Intensity_Product'] = (
                            feature_df['Event_freq_unit_per_day_log'] * 
                            feature_df['Intensity_Level_encoded']
                        )
                        
                        # Frequency-based risk indicators
                        feature_df['Risk_Indicator_Freq'] = np.where(
                            (feature_df['Event_freq_unit_per_day_log'] > 
                             feature_df['Event_freq_unit_per_day_log'].quantile(0.75)) &
                            (feature_df['Intensity_Level_encoded'] > 
                             feature_df['Intensity_Level_encoded'].quantile(0.75)),
                            1, 0
                        )
            
            # 4. Geological stability indicators
            if self.config['geological_features']['stability_indicators']:
                custom_logging.info("Creating geological stability indicators")
                
                # Multi-factor stability index
                stability_factors = []
                
                if 'Energy_per_Volume_log' in feature_df.columns:
                    stability_factors.append(feature_df['Energy_per_Volume_log'])
                
                if 'Duration_days' in feature_df.columns:
                    # Longer duration might indicate more stable conditions
                    stability_factors.append(-feature_df['Duration_days'])  # Negative for inverse relationship
                
                if 'Volume_m3_per_day_sqr' in feature_df.columns:
                    stability_factors.append(-feature_df['Volume_m3_per_day_sqr'])  # Negative for inverse
                
                if stability_factors:
                    # Combine stability factors using weighted average
                    feature_df['Geological_Stability_Index'] = np.mean(stability_factors, axis=0)
                    
                    # Stability risk categories
                    stability_quantiles = feature_df['Geological_Stability_Index'].quantile([0.33, 0.67])
                    feature_df['Stability_Risk_Category'] = pd.cut(
                        feature_df['Geological_Stability_Index'],
                        bins=[-np.inf, stability_quantiles[0.33], stability_quantiles[0.67], np.inf],
                        labels=[0, 1, 2]  # 0: Low risk, 1: Medium risk, 2: High risk
                    ).astype(int)
                
                # Volume-energy stability ratios
                if ('Volume_m3_sqr' in feature_df.columns and 
                    'Energy_density_Joule_sqr' in feature_df.columns):
                    feature_df['Volume_Energy_Stability_Ratio'] = (
                        feature_df['Volume_m3_sqr'] / (feature_df['Energy_density_Joule_sqr'] + 1e-8)
                    )
            
            # 5. Advanced ratio and interaction features
            custom_logging.info("Creating advanced ratio and interaction features")
            
            # Energy ratios across different metrics
            if ('Energy_Unit_log' in feature_df.columns and 
                'Energy_Joule_per_day_sqr' in feature_df.columns):
                feature_df['Total_Daily_Energy_Ratio'] = (
                    feature_df['Energy_Unit_log'] / (feature_df['Energy_Joule_per_day_sqr'] + 1e-8)
                )
            
            # Volume ratios and patterns  
            if ('Volume_m3_sqr' in feature_df.columns and 
                'Volume_m3_per_day_sqr' in feature_df.columns):
                feature_df['Volume_Daily_Ratio'] = (
                    feature_df['Volume_m3_sqr'] / (feature_df['Volume_m3_per_day_sqr'] + 1e-8)
                )
            
            # Duration-based efficiency metrics
            if 'Duration_days' in feature_df.columns:
                if 'Energy_Unit_log' in feature_df.columns:
                    feature_df['Energy_Efficiency_Per_Day'] = (
                        feature_df['Energy_Unit_log'] / (feature_df['Duration_days'] + 1e-8)
                    )
                
                if 'Volume_m3_sqr' in feature_df.columns:
                    feature_df['Volume_Efficiency_Per_Day'] = (
                        feature_df['Volume_m3_sqr'] / (feature_df['Duration_days'] + 1e-8)
                    )
            
            # Log feature creation results
            new_features = [col for col in feature_df.columns if col not in original_columns]
            custom_logging.info(f"Created {len(new_features)} geological features:")
            for feature in new_features:
                custom_logging.info(f"  - {feature}")
            
            # Store feature metadata
            self.feature_metadata['geological_features'] = {
                'total_features_created': len(new_features),
                'feature_names': new_features,
                'creation_timestamp': datetime.now().isoformat(),
                'source_columns': available_cols
            }
            
            custom_logging.info("Geological feature engineering completed successfully")
            return feature_df
            
        except Exception as e:
            error_message = f"Error in geological feature creation: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
        """
        Create temporal features for time-series analysis.
        
        Args:
            df: Input DataFrame
            date_col: Name of date/timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        try:
            custom_logging.info("Creating temporal features for time-series analysis")
            
            feature_df = df.copy()
            original_columns = feature_df.columns.tolist()
            
            # If no date column specified, try to find one
            if date_col is None:
                date_candidates = [col for col in feature_df.columns 
                                 if any(keyword in col.lower() 
                                       for keyword in ['date', 'time', 'timestamp'])]
                if date_candidates:
                    date_col = date_candidates[0]
                    custom_logging.info(f"Using {date_col} as date column")
            
            # If we have a date column, create temporal features
            if date_col and date_col in feature_df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(feature_df[date_col]):
                    feature_df[date_col] = pd.to_datetime(feature_df[date_col], errors='coerce')
                
                # Extract basic temporal features
                feature_df['Year'] = feature_df[date_col].dt.year
                feature_df['Month'] = feature_df[date_col].dt.month
                feature_df['Day'] = feature_df[date_col].dt.day
                feature_df['DayOfWeek'] = feature_df[date_col].dt.dayofweek
                feature_df['DayOfYear'] = feature_df[date_col].dt.dayofyear
                feature_df['WeekOfYear'] = feature_df[date_col].dt.isocalendar().week
                feature_df['Quarter'] = feature_df[date_col].dt.quarter
                
                # Seasonal features
                if self.config['temporal_features']['seasonal_features']:
                    feature_df['Season'] = feature_df['Month'].map({
                        12: 0, 1: 0, 2: 0,  # Winter
                        3: 1, 4: 1, 5: 1,   # Spring
                        6: 2, 7: 2, 8: 2,   # Summer
                        9: 3, 10: 3, 11: 3  # Fall
                    })
                    
                    # Cyclical features for better ML performance
                    feature_df['Month_Sin'] = np.sin(2 * np.pi * feature_df['Month'] / 12)
                    feature_df['Month_Cos'] = np.cos(2 * np.pi * feature_df['Month'] / 12)
                    feature_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * feature_df['DayOfWeek'] / 7)
                    feature_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * feature_df['DayOfWeek'] / 7)
            
            # Create rolling window features for numerical columns
            if self.config['temporal_features']['rolling_windows']:
                custom_logging.info("Creating rolling window features")
                
                numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
                user_data_cols = [col for col in numerical_cols 
                                if any(keyword in col for keyword in ['Energy', 'Volume', 'Duration', 'Event', 'Intensity'])]
                
                for window in self.config['temporal_features']['rolling_windows']:
                    if len(feature_df) >= window:
                        for col in user_data_cols:
                            feature_df[f'{col}_Rolling_{window}d_Mean'] = (
                                feature_df[col].rolling(window=window, min_periods=1).mean()
                            )
                            feature_df[f'{col}_Rolling_{window}d_Std'] = (
                                feature_df[col].rolling(window=window, min_periods=1).std()
                            )
                            feature_df[f'{col}_Rolling_{window}d_Max'] = (
                                feature_df[col].rolling(window=window, min_periods=1).max()
                            )
                            feature_df[f'{col}_Rolling_{window}d_Min'] = (
                                feature_df[col].rolling(window=window, min_periods=1).min()
                            )
            
            # Create lag features
            if self.config['temporal_features']['lag_features']:
                custom_logging.info("Creating lag features")
                
                for lag in self.config['temporal_features']['lag_features']:
                    if len(feature_df) > lag:
                        for col in user_data_cols:
                            feature_df[f'{col}_Lag_{lag}'] = feature_df[col].shift(lag)
            
            # Create difference features
            if self.config['temporal_features']['difference_features']:
                custom_logging.info("Creating difference features")
                
                for diff in self.config['temporal_features']['difference_features']:
                    if len(feature_df) > diff:
                        for col in user_data_cols:
                            feature_df[f'{col}_Diff_{diff}'] = feature_df[col].diff(periods=diff)
            
            # Fill NaN values created by rolling, lag, and diff operations
            feature_df.fillna(method='bfill', inplace=True)
            feature_df.fillna(method='ffill', inplace=True)
            feature_df.fillna(0, inplace=True)
            
            new_features = [col for col in feature_df.columns if col not in original_columns]
            custom_logging.info(f"Created {len(new_features)} temporal features")
            
            # Store metadata
            self.feature_metadata['temporal_features'] = {
                'total_features_created': len(new_features),
                'feature_names': new_features,
                'creation_timestamp': datetime.now().isoformat(),
                'date_column_used': date_col
            }
            
            return feature_df
            
        except Exception as e:
            error_message = f"Error in temporal feature creation: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 50) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: Input DataFrame
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        try:
            custom_logging.info(f"Creating interaction features (max: {max_interactions})")
            
            feature_df = df.copy()
            original_columns = feature_df.columns.tolist()
            
            # Get numerical columns for interactions
            numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column if present
            target_candidates = [col for col in numerical_cols 
                               if 'intensity' in col.lower() or 'target' in col.lower()]
            for target in target_candidates:
                if target in numerical_cols:
                    numerical_cols.remove(target)
            
            # Limit to core user data columns and most important derived features
            priority_cols = []
            for col in numerical_cols:
                if any(keyword in col for keyword in ['Energy', 'Volume', 'Duration', 'Event', 'Stability', 'Risk']):
                    priority_cols.append(col)
            
            # If we have too many columns, select top ones by variance
            if len(priority_cols) > 15:
                variances = feature_df[priority_cols].var().sort_values(ascending=False)
                priority_cols = variances.head(15).index.tolist()
            
            custom_logging.info(f"Using {len(priority_cols)} columns for interaction features")
            
            interaction_count = 0
            
            # Create polynomial features
            if self.config['interaction_features']['polynomial_degree'] >= 2:
                custom_logging.info("Creating polynomial features")
                for col in priority_cols[:10]:  # Limit to avoid explosion
                    if interaction_count >= max_interactions:
                        break
                    
                    # Squared features
                    feature_df[f'{col}_Squared'] = feature_df[col] ** 2
                    interaction_count += 1
                    
                    # Cubed features for energy-related columns
                    if 'energy' in col.lower() and interaction_count < max_interactions:
                        feature_df[f'{col}_Cubed'] = feature_df[col] ** 3
                        interaction_count += 1
            
            # Create cross-product features  
            if self.config['interaction_features']['cross_products']:
                custom_logging.info("Creating cross-product features")
                
                for i, col1 in enumerate(priority_cols):
                    if interaction_count >= max_interactions:
                        break
                    for col2 in priority_cols[i+1:]:
                        if interaction_count >= max_interactions:
                            break
                        
                        # Create interaction feature
                        feature_df[f'{col1}_X_{col2}'] = feature_df[col1] * feature_df[col2]
                        interaction_count += 1
            
            # Create ratio features
            if self.config['interaction_features']['ratio_features']:
                custom_logging.info("Creating ratio features")
                
                for i, col1 in enumerate(priority_cols):
                    if interaction_count >= max_interactions:
                        break
                    for col2 in priority_cols[i+1:]:
                        if interaction_count >= max_interactions:
                            break
                        
                        # Create ratio feature (with small epsilon to avoid division by zero)
                        feature_df[f'{col1}_Ratio_{col2}'] = (
                            feature_df[col1] / (feature_df[col2].abs() + 1e-8)
                        )
                        interaction_count += 1
            
            new_features = [col for col in feature_df.columns if col not in original_columns]
            custom_logging.info(f"Created {len(new_features)} interaction features")
            
            # Store metadata
            self.feature_metadata['interaction_features'] = {
                'total_features_created': len(new_features),
                'feature_names': new_features,
                'creation_timestamp': datetime.now().isoformat(),
                'source_columns_used': priority_cols
            }
            
            return feature_df
            
        except Exception as e:
            error_message = f"Error in interaction feature creation: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def apply_feature_scaling(self, df: pd.DataFrame, target_col: str = None, 
                             method: str = 'robust') -> Tuple[pd.DataFrame, object]:
        """
        Apply feature scaling to numerical columns.
        
        Args:
            df: Input DataFrame
            target_col: Target column to exclude from scaling
            method: Scaling method ('standard', 'minmax', 'robust', 'power')
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler object)
        """
        try:
            custom_logging.info(f"Applying {method} scaling to features")
            
            feature_df = df.copy()
            
            # Identify numerical columns for scaling
            numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column and metadata columns
            exclude_cols = []
            if target_col and target_col in numerical_cols:
                exclude_cols.append(target_col)
            
            # Remove metadata columns
            metadata_keywords = ['timestamp', 'date', 'file', 'source', 'creation', 'year', 'month', 'day']
            for col in numerical_cols:
                if any(keyword in col.lower() for keyword in metadata_keywords):
                    exclude_cols.append(col)
            
            scaling_cols = [col for col in numerical_cols if col not in exclude_cols]
            custom_logging.info(f"Scaling {len(scaling_cols)} numerical columns")
            
            if not scaling_cols:
                custom_logging.warning("No columns found for scaling")
                return feature_df, None
            
            # Get the appropriate scaler
            if method not in self.scalers:
                custom_logging.warning(f"Unknown scaling method {method}, using 'robust'")
                method = 'robust'
            
            scaler = self.scalers[method]
            
            # Fit and transform the features
            scaled_features = scaler.fit_transform(feature_df[scaling_cols])
            
            # Replace original columns with scaled values
            for i, col in enumerate(scaling_cols):
                feature_df[col] = scaled_features[:, i]
            
            # Add scaling metadata
            feature_df[f'scaling_method'] = method
            feature_df[f'scaling_timestamp'] = datetime.now().isoformat()
            
            custom_logging.info(f"Feature scaling completed using {method} method")
            
            return feature_df, scaler
            
        except Exception as e:
            error_message = f"Error in feature scaling: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       method: str = 'f_classif', k: int = 50) -> Tuple[pd.DataFrame, object]:
        """
        Perform feature selection to identify most important features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            method: Selection method ('f_classif', 'mutual_info')
            k: Number of top features to select
            
        Returns:
            Tuple of (DataFrame with selected features, fitted selector)
        """
        try:
            custom_logging.info(f"Performing feature selection using {method} (k={k})")
            
            if target_col not in df.columns:
                custom_logging.error(f"Target column '{target_col}' not found in DataFrame")
                return df, None
            
            feature_df = df.copy()
            
            # Get feature columns (exclude target)
            feature_cols = [col for col in feature_df.columns if col != target_col]
            numerical_cols = feature_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) <= k:
                custom_logging.info(f"Number of features ({len(numerical_cols)}) <= k ({k}), returning all features")
                return feature_df, None
            
            # Prepare data
            X = feature_df[numerical_cols].fillna(0)
            y = feature_df[target_col]
            
            # Get selector
            if method not in self.feature_selectors:
                custom_logging.warning(f"Unknown selection method {method}, using 'f_classif'")
                method = 'f_classif'
            
            selector = self.feature_selectors[method]
            selector.set_params(k=min(k, len(numerical_cols)))
            
            # Fit and transform
            X_selected = selector.fit_transform(X, y)
            selected_features = selector.get_feature_names_out(numerical_cols)
            
            # Create result DataFrame
            result_df = pd.DataFrame(X_selected, columns=selected_features, index=feature_df.index)
            result_df[target_col] = feature_df[target_col]
            
            # Add non-numerical columns back
            non_numerical_cols = [col for col in feature_cols if col not in numerical_cols]
            for col in non_numerical_cols:
                result_df[col] = feature_df[col]
            
            custom_logging.info(f"Selected {len(selected_features)} features out of {len(numerical_cols)}")
            
            return result_df, selector
            
        except Exception as e:
            error_message = f"Error in feature selection: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def reduce_dimensionality(self, df: pd.DataFrame, target_col: str = None, 
                             method: str = 'pca', n_components: int = None) -> Tuple[pd.DataFrame, object]:
        """
        Apply dimensionality reduction techniques.
        
        Args:
            df: Input DataFrame
            target_col: Target column to preserve
            method: Reduction method ('pca', 'ica', 'tsne')
            n_components: Number of components to keep
            
        Returns:
            Tuple of (reduced DataFrame, fitted reducer)
        """
        try:
            custom_logging.info(f"Applying {method} dimensionality reduction")
            
            feature_df = df.copy()
            
            # Get numerical feature columns
            exclude_cols = [target_col] if target_col and target_col in feature_df.columns else []
            numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numerical_cols if col not in exclude_cols]
            
            if len(feature_cols) < 2:
                custom_logging.warning("Insufficient features for dimensionality reduction")
                return feature_df, None
            
            # Prepare data
            X = feature_df[feature_cols].fillna(0)
            
            # Set default n_components if not specified
            if n_components is None:
                if method == 'pca':
                    n_components = min(len(feature_cols), 20)
                elif method == 'ica':
                    n_components = min(len(feature_cols), 15)
                elif method == 'tsne':
                    n_components = 2  # Usually 2 or 3 for t-SNE
            
            # Get reducer
            if method not in self.dim_reducers:
                custom_logging.warning(f"Unknown reduction method {method}, using 'pca'")
                method = 'pca'
            
            reducer = self.dim_reducers[method]
            
            # Set parameters
            if hasattr(reducer, 'n_components'):
                reducer.set_params(n_components=min(n_components, len(feature_cols)))
            
            # Fit and transform
            X_reduced = reducer.fit_transform(X)
            
            # Create result DataFrame
            component_names = [f'{method.upper()}_Component_{i+1}' for i in range(X_reduced.shape[1])]
            result_df = pd.DataFrame(X_reduced, columns=component_names, index=feature_df.index)
            
            # Add target and other non-feature columns back
            for col in feature_df.columns:
                if col not in feature_cols:
                    result_df[col] = feature_df[col]
            
            custom_logging.info(f"Reduced {len(feature_cols)} features to {X_reduced.shape[1]} components")
            
            return result_df, reducer
            
        except Exception as e:
            error_message = f"Error in dimensionality reduction: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def create_comprehensive_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set using all available techniques.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with comprehensive features
        """
        try:
            custom_logging.info("Creating comprehensive feature set for rockburst prediction")
            
            # Start with original data
            feature_df = df.copy()
            
            # 1. Create geological features
            feature_df = self.create_geological_features(feature_df, target_col)
            
            # 2. Create temporal features if possible
            feature_df = self.create_temporal_features(feature_df)
            
            # 3. Create interaction features
            feature_df = self.create_interaction_features(feature_df, max_interactions=30)
            
            # 4. Handle missing values
            feature_df = self.handle_missing_values(feature_df)
            
            # Log final feature count
            original_cols = len(df.columns)
            final_cols = len(feature_df.columns)
            new_features = final_cols - original_cols
            
            custom_logging.info(f"Comprehensive feature engineering completed:")
            custom_logging.info(f"  - Original features: {original_cols}")
            custom_logging.info(f"  - Final features: {final_cols}")
            custom_logging.info(f"  - New features created: {new_features}")
            
            # Store comprehensive metadata
            self.feature_metadata['comprehensive'] = {
                'original_features': original_cols,
                'final_features': final_cols,
                'new_features_created': new_features,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return feature_df
            
        except Exception as e:
            error_message = f"Error in comprehensive feature creation: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'knn') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            method: Imputation method ('simple_mean', 'simple_median', 'knn')
            
        Returns:
            DataFrame with imputed values
        """
        try:
            custom_logging.info(f"Handling missing values using {method} method")
            
            feature_df = df.copy()
            
            # Check for missing values
            missing_counts = feature_df.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0]
            
            if len(columns_with_missing) == 0:
                custom_logging.info("No missing values found")
                return feature_df
            
            custom_logging.info(f"Found missing values in {len(columns_with_missing)} columns")
            
            # Get numerical columns for imputation
            numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            missing_numerical = [col for col in columns_with_missing.index if col in numerical_cols]
            
            if missing_numerical:
                # Get appropriate imputer
                if method not in self.imputers:
                    custom_logging.warning(f"Unknown imputation method {method}, using 'simple_median'")
                    method = 'simple_median'
                
                imputer = self.imputers[method]
                
                # Impute numerical columns
                imputed_values = imputer.fit_transform(feature_df[missing_numerical])
                
                for i, col in enumerate(missing_numerical):
                    feature_df[col] = imputed_values[:, i]
            
            # Handle categorical missing values with mode
            categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
            missing_categorical = [col for col in columns_with_missing.index if col in categorical_cols]
            
            for col in missing_categorical:
                mode_value = feature_df[col].mode()
                if len(mode_value) > 0:
                    feature_df[col].fillna(mode_value[0], inplace=True)
                else:
                    feature_df[col].fillna('Unknown', inplace=True)
            
            custom_logging.info("Missing value imputation completed")
            return feature_df
            
        except Exception as e:
            error_message = f"Error in missing value handling: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)
    
    def save_feature_engineering_artifacts(self, output_dir: str) -> Dict:
        """
        Save feature engineering artifacts (scalers, selectors, etc.) to disk.
        
        Args:
            output_dir: Directory to save artifacts
            
        Returns:
            Dictionary with saved artifact information
        """
        try:
            custom_logging.info(f"Saving feature engineering artifacts to {output_dir}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            artifacts_saved = {}
            
            # Save scalers
            for name, scaler in self.scalers.items():
                if hasattr(scaler, 'scale_'):  # Check if fitted
                    scaler_path = os.path.join(output_dir, f'scaler_{name}.joblib')
                    joblib.dump(scaler, scaler_path)
                    artifacts_saved[f'scaler_{name}'] = scaler_path
            
            # Save feature selectors
            for name, selector in self.feature_selectors.items():
                if hasattr(selector, 'scores_'):  # Check if fitted
                    selector_path = os.path.join(output_dir, f'selector_{name}.joblib')
                    joblib.dump(selector, selector_path)
                    artifacts_saved[f'selector_{name}'] = selector_path
            
            # Save dimensionality reducers
            for name, reducer in self.dim_reducers.items():
                if hasattr(reducer, 'components_') or hasattr(reducer, 'embedding_'):  # Check if fitted
                    reducer_path = os.path.join(output_dir, f'reducer_{name}.joblib')
                    joblib.dump(reducer, reducer_path)
                    artifacts_saved[f'reducer_{name}'] = reducer_path
            
            # Save feature metadata
            metadata_path = os.path.join(output_dir, 'feature_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.feature_metadata, f, indent=2)
            artifacts_saved['feature_metadata'] = metadata_path
            
            # Save configuration
            config_path = os.path.join(output_dir, 'feature_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            artifacts_saved['feature_config'] = config_path
            
            custom_logging.info(f"Saved {len(artifacts_saved)} feature engineering artifacts")
            return artifacts_saved
            
        except Exception as e:
            error_message = f"Error saving feature engineering artifacts: {str(e)}"
            custom_logging.error(error_message)
            raise CustomException(e, sys)


def load_feature_engineering_artifacts(artifacts_dir: str) -> RockburstFeatureEngineering:
    """
    Load feature engineering artifacts and return configured instance.
    
    Args:
        artifacts_dir: Directory containing saved artifacts
        
    Returns:
        Configured RockburstFeatureEngineering instance
    """
    try:
        custom_logging.info(f"Loading feature engineering artifacts from {artifacts_dir}")
        
        # Load configuration
        config_path = os.path.join(artifacts_dir, 'feature_config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create instance
        feature_engineer = RockburstFeatureEngineering(config)
        
        # Load scalers
        for scaler_name in ['standard', 'minmax', 'robust', 'power']:
            scaler_path = os.path.join(artifacts_dir, f'scaler_{scaler_name}.joblib')
            if os.path.exists(scaler_path):
                feature_engineer.scalers[scaler_name] = joblib.load(scaler_path)
        
        # Load selectors
        for selector_name in ['f_classif', 'mutual_info']:
            selector_path = os.path.join(artifacts_dir, f'selector_{selector_name}.joblib')
            if os.path.exists(selector_path):
                feature_engineer.feature_selectors[selector_name] = joblib.load(selector_path)
        
        # Load reducers
        for reducer_name in ['pca', 'ica', 'tsne']:
            reducer_path = os.path.join(artifacts_dir, f'reducer_{reducer_name}.joblib')
            if os.path.exists(reducer_path):
                feature_engineer.dim_reducers[reducer_name] = joblib.load(reducer_path)
        
        # Load metadata
        metadata_path = os.path.join(artifacts_dir, 'feature_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                feature_engineer.feature_metadata = json.load(f)
        
        custom_logging.info("Feature engineering artifacts loaded successfully")
        return feature_engineer
        
    except Exception as e:
        error_message = f"Error loading feature engineering artifacts: {str(e)}"
        custom_logging.error(error_message)
        raise CustomException(e, sys)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the RockburstFeatureEngineering class
    """
    custom_logging.info("Testing RockburstFeatureEngineering class")
    
    # Create sample data with user input structure
    np.random.seed(42)
    sample_data = {
        'Duration_days': np.random.uniform(1, 30, 100),
        'Energy_Unit_log': np.random.uniform(2, 8, 100),
        'Energy_density_Joule_sqr': np.random.uniform(10, 1000, 100),
        'Volume_m3_sqr': np.random.uniform(5, 500, 100),
        'Event_freq_unit_per_day_log': np.random.uniform(0, 5, 100),
        'Energy_Joule_per_day_sqr': np.random.uniform(20, 2000, 100),
        'Volume_m3_per_day_sqr': np.random.uniform(10, 800, 100),
        'Energy_per_Volume_log': np.random.uniform(0, 6, 100),
        'Intensity_Level_encoded': np.random.randint(0, 3, 100)
    }
    
    df = pd.DataFrame(sample_data)
    custom_logging.info(f"Created sample dataset with shape: {df.shape}")
    
    # Initialize feature engineering
    feature_engineer = RockburstFeatureEngineering()
    
    # Create comprehensive features
    enhanced_df = feature_engineer.create_comprehensive_features(df, 'Intensity_Level_encoded')
    custom_logging.info(f"Enhanced dataset shape: {enhanced_df.shape}")
    
    custom_logging.info("RockburstFeatureEngineering testing completed successfully")
