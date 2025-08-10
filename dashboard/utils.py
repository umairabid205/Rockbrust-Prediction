"""
Dashboard Utilities
==================
Common utility functions for the rockburst monitoring dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
import time
import os


class APIClient:
    """API client for communicating with the prediction service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make single prediction"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict", 
                json=features
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"predictions": features_list}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics"""
        try:
            response = self.session.get(f"{self.base_url}/model/metrics")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


class DataGenerator:
    """Generate synthetic geological data for testing"""
    
    @staticmethod
    def generate_realistic_features() -> Dict[str, float]:
        """Generate realistic geological features"""
        np.random.seed(int(time.time()) % 1000)
        
        # Base geological parameters
        depth = np.random.uniform(100, 2000)  # meters
        stress_level = np.random.uniform(20, 200)  # MPa
        seismic_activity = np.random.uniform(0, 100)  # arbitrary units
        water_pressure = np.random.uniform(0, 80)  # MPa
        temperature = np.random.uniform(15, 60)  # Celsius
        
        # Derived parameters with some correlation
        ground_displacement = np.random.uniform(0, 50) + stress_level * 0.1
        elastic_modulus = np.random.uniform(10000, 80000)  # MPa
        poisson_ratio = np.random.uniform(0.1, 0.4)
        
        # Rock properties
        ucs = np.random.uniform(20, 300)  # MPa (Unconfined Compressive Strength)
        density = np.random.uniform(2000, 3000)  # kg/m³
        
        # Mining parameters
        extraction_ratio = np.random.uniform(0.3, 0.9)
        support_density = np.random.uniform(0.5, 2.0)  # supports per m²
        
        # Environmental factors
        humidity = np.random.uniform(30, 90)  # %
        ventilation_rate = np.random.uniform(0.5, 5.0)  # m³/s
        
        # Add some noise and correlations
        if stress_level > 150:
            seismic_activity += np.random.uniform(10, 30)
            ground_displacement += np.random.uniform(5, 15)
        
        if depth > 1000:
            temperature += depth * 0.02
            stress_level += depth * 0.05
        
        return {
            'depth': round(depth, 2),
            'stress_level': round(stress_level, 2),
            'seismic_activity': round(seismic_activity, 2),
            'water_pressure': round(water_pressure, 2),
            'temperature': round(temperature, 2),
            'ground_displacement': round(ground_displacement, 2),
            'elastic_modulus': round(elastic_modulus, 2),
            'poisson_ratio': round(poisson_ratio, 3),
            'ucs': round(ucs, 2),
            'density': round(density, 2),
            'extraction_ratio': round(extraction_ratio, 3),
            'support_density': round(support_density, 3),
            'humidity': round(humidity, 2),
            'ventilation_rate': round(ventilation_rate, 2)
        }
    
    @staticmethod
    def generate_batch_features(count: int = 10) -> List[Dict[str, float]]:
        """Generate batch of realistic features"""
        return [DataGenerator.generate_realistic_features() for _ in range(count)]
    
    @staticmethod
    def generate_high_risk_scenario() -> Dict[str, float]:
        """Generate a high-risk geological scenario"""
        features = DataGenerator.generate_realistic_features()
        
        # Make it high risk
        features['stress_level'] = np.random.uniform(150, 200)
        features['seismic_activity'] = np.random.uniform(70, 100)
        features['ground_displacement'] = np.random.uniform(30, 50)
        features['water_pressure'] = np.random.uniform(60, 80)
        features['extraction_ratio'] = np.random.uniform(0.8, 0.95)
        
        return features


class ChartGenerator:
    """Generate various charts for the dashboard"""
    
    @staticmethod
    def create_gauge_chart(value: float, title: str, 
                          color_scale: str = "RdYlGn_r",
                          min_val: float = 0, max_val: float = 1) -> go.Figure:
        """Create a gauge chart for risk probability"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': title, 'font': {'size': 24}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    @staticmethod
    def create_time_series_chart(timestamps: List[datetime], 
                               values: List[float],
                               title: str = "Time Series",
                               y_label: str = "Value") -> go.Figure:
        """Create time series chart"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Risk Probability',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            height=400,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(features: Dict[str, float],
                                      title: str = "Feature Importance") -> go.Figure:
        """Create feature importance bar chart"""
        
        # Sort features by importance
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_names, importance_values = zip(*sorted_features)
        
        # Create colors based on positive/negative importance
        colors = ['red' if val < 0 else 'green' for val in importance_values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(feature_names),
                y=list(importance_values),
                marker_color=colors,
                text=[f"{val:.3f}" for val in importance_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Importance",
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxis(tickangle=-45)
        
        return fig
    
    @staticmethod
    def create_risk_distribution_chart(risk_levels: List[str]) -> go.Figure:
        """Create pie chart for risk distribution"""
        
        risk_counts = pd.Series(risk_levels).value_counts()
        
        colors = {
            'Low': '#2ecc71',
            'Medium': '#f1c40f', 
            'High': '#e74c3c',
            'Critical': '#8e44ad'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=[colors.get(label, '#95a5a6') for label in risk_counts.index],
            hole=0.4,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with appropriate precision"""
    if abs(num) >= 1000000:
        return f"{num/1000000:.{decimals}f}M"
    elif abs(num) >= 1000:
        return f"{num/1000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def get_risk_color(probability: float) -> str:
    """Get color based on risk probability"""
    if probability >= 0.8:
        return "#dc2626"  # Red
    elif probability >= 0.6:
        return "#f59e0b"  # Orange
    elif probability >= 0.4:
        return "#eab308"  # Yellow
    else:
        return "#16a34a"  # Green


def get_risk_level(probability: float) -> str:
    """Get risk level text from probability"""
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def cache_data(key: str, data: Any, expiry_minutes: int = 5):
    """Cache data in session state with expiry"""
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    st.session_state.cache[key] = {
        'data': data,
        'timestamp': datetime.now(),
        'expiry_minutes': expiry_minutes
    }


def get_cached_data(key: str) -> Optional[Any]:
    """Get cached data if not expired"""
    if 'cache' not in st.session_state:
        return None
    
    if key not in st.session_state.cache:
        return None
    
    cache_entry = st.session_state.cache[key]
    if datetime.now() - cache_entry['timestamp'] > timedelta(minutes=cache_entry['expiry_minutes']):
        del st.session_state.cache[key]
        return None
    
    return cache_entry['data']


def show_loading_spinner(text: str = "Loading..."):
    """Show loading spinner with text"""
    return st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 10px;">{text}</p>
    </div>
    """, unsafe_allow_html=True)


def create_status_indicator(status: str, label: str = "") -> str:
    """Create colored status indicator"""
    status_colors = {
        'online': '#10b981',
        'offline': '#ef4444', 
        'warning': '#f59e0b',
        'error': '#dc2626',
        'success': '#10b981'
    }
    
    color = status_colors.get(status.lower(), '#6b7280')
    circle = f"<span style='color: {color}; font-size: 1.2em;'>●</span>"
    
    if label:
        return f"{circle} <span style='color: {color}; font-weight: 600;'>{label}</span>"
    else:
        return circle


def export_data_to_csv(data: pd.DataFrame, filename: str = None) -> str:
    """Export dataframe to CSV and return download link"""
    if filename is None:
        filename = f"rockburst_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv = data.to_csv(index=False)
    return csv


def validate_geological_features(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Validate geological feature values"""
    errors = []
    
    # Define reasonable ranges for geological parameters
    ranges = {
        'depth': (0, 5000),
        'stress_level': (0, 500),
        'seismic_activity': (0, 100),
        'water_pressure': (0, 200),
        'temperature': (-10, 100),
        'ground_displacement': (0, 200),
        'elastic_modulus': (1000, 100000),
        'poisson_ratio': (0.05, 0.5),
        'ucs': (1, 500),
        'density': (1000, 5000),
        'extraction_ratio': (0, 1),
        'support_density': (0, 10),
        'humidity': (0, 100),
        'ventilation_rate': (0, 20)
    }
    
    for feature, value in features.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if not (min_val <= value <= max_val):
                errors.append(f"{feature}: {value} (should be between {min_val} and {max_val})")
    
    return len(errors) == 0, errors
