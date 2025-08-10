#!/usr/bin/env python3
"""
Rockburst Prediction Monitoring Dashboard
==========================================
Streamlit-based dashboard for monitoring rockburst predictions, model performance,
and real-time alerts for high-risk scenarios.

Features:
- Real-time prediction monitoring
- Model performance analytics
- High-risk alerts and notifications
- Interactive prediction interface
- Historical data visualization
- System health monitoring

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project paths
sys.path.append('.')
sys.path.append('..')
sys.path.append('../models')

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds
HIGH_RISK_THRESHOLD = 0.7
CRITICAL_RISK_THRESHOLD = 0.9

# Page configuration
st.set_page_config(
    page_title="Rockburst Monitoring Dashboard",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .alert-high {
        background: linear-gradient(90deg, #ffe6e6 0%, #ffcccc 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 0.5rem 0;
    }
    
    .alert-critical {
        background: linear-gradient(90deg, #ff9999 0%, #ff6666 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #cc0000;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DashboardAPI:
    """API client for dashboard data"""
    
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_health(self):
        """Get API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get health status: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get model info: {str(e)}")
            return None
    
    def get_metrics(self):
        """Get API metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get metrics: {str(e)}")
            return None
    
    def make_prediction(self, data):
        """Make single prediction"""
        try:
            response = self.session.post(f"{self.base_url}/predict", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to make prediction: {str(e)}")
            return None
    
    def make_batch_prediction(self, data_list):
        """Make batch prediction"""
        try:
            batch_data = {"predictions": data_list}
            response = self.session.post(f"{self.base_url}/predict/batch", json=batch_data, timeout=30)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to make batch prediction: {str(e)}")
            return None


# Initialize API client
@st.cache_resource
def get_api_client():
    return DashboardAPI()

api = get_api_client()


def generate_sample_historical_data(days=30, samples_per_day=24):
    """Generate sample historical prediction data"""
    np.random.seed(42)
    
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')[:days*samples_per_day]
    
    # Generate realistic geological data
    n_samples = len(timestamps)
    
    data = {
        'timestamp': timestamps,
        'seismic_activity': np.random.beta(2, 5, n_samples) * 100,
        'rock_strength': np.random.normal(150, 30, n_samples),
        'depth': np.random.normal(1200, 300, n_samples),
        'stress_level': np.random.beta(3, 4, n_samples) * 200,
        'water_pressure': np.random.beta(2, 3, n_samples) * 100,
        'temperature': np.random.normal(25, 10, n_samples),
        'ground_displacement': np.random.exponential(10, n_samples),
        'mining_activity': np.random.beta(3, 2, n_samples) * 100,
        'geological_conditions': np.random.beta(4, 3, n_samples) * 100,
    }
    
    df = pd.DataFrame(data)
    
    # Generate predictions based on risk factors
    risk_score = (
        df['seismic_activity'] * 0.25 +
        (200 - df['rock_strength']) * 0.2 +
        df['depth'] * 0.001 * 0.15 +
        df['stress_level'] * 0.2 +
        df['water_pressure'] * 0.1 +
        df['ground_displacement'] * 0.05 +
        df['mining_activity'] * 0.05
    ) / 100
    
    # Add noise and normalize
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = np.clip(risk_score, 0, 1)
    
    df['probability'] = risk_score
    df['prediction'] = (risk_score > 0.5).astype(int)
    df['risk_level'] = pd.cut(risk_score, 
                              bins=[0, 0.3, 0.6, 0.8, 1.0], 
                              labels=['Low', 'Medium', 'High', 'Critical'])
    df['confidence'] = np.random.beta(8, 2, n_samples)  # High confidence simulation
    
    return df


@st.cache_data(ttl=60)
def get_historical_data():
    """Get historical prediction data (cached for 1 minute)"""
    return generate_sample_historical_data()


def render_header():
    """Render dashboard header"""
    st.markdown('<h1 class="main-header">⛰️ Rockburst Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_system_status():
    """Render system status section"""
    st.header("🔧 System Status")
    
    # Get system health
    health = api.get_health()
    model_info = api.get_model_info()
    metrics = api.get_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health:
            status_class = "status-healthy" if health['status'] == 'healthy' else "status-error"
            st.markdown(f'<div class="metric-container">**API Status**<br><span class="{status_class}">{health["status"].upper()}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">**API Status**<br><span class="status-error">OFFLINE</span></div>', unsafe_allow_html=True)
    
    with col2:
        if health:
            model_status = "LOADED" if health['model_loaded'] else "NOT LOADED"
            status_class = "status-healthy" if health['model_loaded'] else "status-error"
            st.markdown(f'<div class="metric-container">**Model Status**<br><span class="{status_class}">{model_status}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">**Model Status**<br><span class="status-error">UNKNOWN</span></div>', unsafe_allow_html=True)
    
    with col3:
        if health:
            uptime_hours = health['uptime_seconds'] / 3600
            st.markdown(f'<div class="metric-container">**Uptime**<br>{uptime_hours:.1f} hours</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">**Uptime**<br>Unknown</div>', unsafe_allow_html=True)
    
    with col4:
        if model_info and model_info.get('accuracy'):
            accuracy_pct = model_info['accuracy'] * 100
            st.markdown(f'<div class="metric-container">**Model Accuracy**<br>{accuracy_pct:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">**Model Accuracy**<br>N/A</div>', unsafe_allow_html=True)


def render_alerts():
    """Render high-risk alerts section"""
    st.header("🚨 Risk Alerts")
    
    # Get recent high-risk predictions from historical data
    df = get_historical_data()
    recent_data = df[df['timestamp'] >= datetime.now() - timedelta(hours=24)]
    high_risk = recent_data[recent_data['probability'] >= HIGH_RISK_THRESHOLD]
    critical_risk = recent_data[recent_data['probability'] >= CRITICAL_RISK_THRESHOLD]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk Alerts (24h)", len(high_risk), delta=None)
    
    with col2:
        st.metric("Critical Risk Alerts (24h)", len(critical_risk), delta=None)
    
    with col3:
        avg_risk = recent_data['probability'].mean()
        st.metric("Average Risk Level", f"{avg_risk:.3f}", delta=None)
    
    # Show recent critical alerts
    if len(critical_risk) > 0:
        st.subheader("🔴 Critical Risk Alerts")
        for _, alert in critical_risk.head(5).iterrows():
            st.markdown(f'''
            <div class="alert-critical">
                <strong>CRITICAL RISK DETECTED</strong><br>
                Time: {alert['timestamp'].strftime('%H:%M:%S')}<br>
                Probability: {alert['probability']:.3f}<br>
                Seismic Activity: {alert['seismic_activity']:.1f}<br>
                Stress Level: {alert['stress_level']:.1f} MPa
            </div>
            ''', unsafe_allow_html=True)
    
    # Show high-risk alerts
    elif len(high_risk) > 0:
        st.subheader("🟡 High Risk Alerts")
        for _, alert in high_risk.head(3).iterrows():
            st.markdown(f'''
            <div class="alert-high">
                <strong>High Risk Alert</strong><br>
                Time: {alert['timestamp'].strftime('%H:%M:%S')}<br>
                Probability: {alert['probability']:.3f}<br>
                Risk Level: {alert['risk_level']}
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.success("✅ No high-risk alerts in the last 24 hours")


def render_prediction_trends():
    """Render prediction trends and analytics"""
    st.header("📈 Prediction Trends")
    
    df = get_historical_data()
    
    # Time series of predictions
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability over time
        fig = px.line(df, x='timestamp', y='probability', 
                     title='Rockburst Probability Over Time',
                     labels={'probability': 'Probability', 'timestamp': 'Time'})
        fig.add_hline(y=HIGH_RISK_THRESHOLD, line_dash="dash", line_color="orange", 
                     annotation_text="High Risk Threshold")
        fig.add_hline(y=CRITICAL_RISK_THRESHOLD, line_dash="dash", line_color="red", 
                     annotation_text="Critical Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title='Risk Level Distribution',
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107', 
                        'High': '#fd7e14',
                        'Critical': '#dc3545'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Geological parameters correlation
    st.subheader("🌍 Geological Parameters Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seismic activity vs probability
        fig = px.scatter(df, x='seismic_activity', y='probability', 
                        color='risk_level',
                        title='Seismic Activity vs Rockburst Probability',
                        labels={'seismic_activity': 'Seismic Activity', 
                               'probability': 'Probability'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress level vs probability
        fig = px.scatter(df, x='stress_level', y='probability', 
                        color='risk_level',
                        title='Stress Level vs Rockburst Probability',
                        labels={'stress_level': 'Stress Level (MPa)', 
                               'probability': 'Probability'})
        st.plotly_chart(fig, use_container_width=True)


def render_model_performance():
    """Render model performance metrics"""
    st.header("🎯 Model Performance")
    
    model_info = api.get_model_info()
    df = get_historical_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if model_info:
            accuracy = model_info.get('accuracy', 0) * 100
            st.metric("Model Accuracy", f"{accuracy:.2f}%")
        else:
            st.metric("Model Accuracy", "N/A")
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col3:
        total_predictions = len(df)
        st.metric("Total Predictions", f"{total_predictions:,}")
    
    with col4:
        high_risk_pct = (len(df[df['probability'] >= HIGH_RISK_THRESHOLD]) / len(df)) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    # Confidence distribution
    fig = px.histogram(df, x='confidence', nbins=30, 
                      title='Model Confidence Distribution',
                      labels={'confidence': 'Confidence Score', 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Model info details
    if model_info:
        st.subheader("📋 Model Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model Name**: {model_info.get('model_name', 'N/A')}  
            **Version**: {model_info.get('version', 'N/A')}  
            **Features**: {model_info.get('features_count', 'N/A')}  
            **Model Size**: {model_info.get('model_size_mb', 'N/A')} MB
            """)
        
        with col2:
            last_trained = model_info.get('last_trained', 'N/A')
            training_samples = model_info.get('training_samples', 'N/A')
            st.info(f"""
            **Last Trained**: {last_trained}  
            **Training Samples**: {training_samples:,} samples  
            **Status**: {'Loaded' if model_info.get('is_loaded', False) else 'Not Loaded'}  
            **Accuracy**: {model_info.get('accuracy', 0)*100:.2f}%
            """)


def render_live_prediction():
    """Render live prediction interface"""
    st.header("🔮 Live Prediction Interface")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("🔬 Seismic Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                duration_days = st.slider("Duration (days)", 1.0, 100.0, 15.5)
                energy_unit_log = st.slider("Energy Unit (log)", 1.0, 10.0, 5.2)
                energy_density = st.slider("Energy Density (J/m²)", 100.0, 1000.0, 450.0)
                volume_m3_sqr = st.slider("Volume (m³ squared)", 50.0, 500.0, 120.0)
            
            with col2:
                event_freq_log = st.slider("Event Frequency (log/day)", 1.0, 5.0, 2.8)
                energy_joule_day = st.slider("Energy per Day (J/day)", 100.0, 2000.0, 890.0)
                volume_per_day = st.slider("Volume per Day (m³/day)", 100.0, 1000.0, 350.0)
                energy_per_volume_log = st.slider("Energy per Volume (log)", 1.0, 10.0, 3.4)
            
            submit_button = st.form_submit_button("🔍 Make Prediction")
        
        if submit_button:
            # Prepare prediction data using correct API fields
            prediction_data = {
                "Duration_days": duration_days,
                "Energy_Unit_log": energy_unit_log,
                "Energy_density_Joule_sqr": energy_density,
                "Volume_m3_sqr": volume_m3_sqr,
                "Event_freq_unit_per_day_log": event_freq_log,
                "Energy_Joule_per_day_sqr": energy_joule_day,
                "Volume_m3_per_day_sqr": volume_per_day,
                "Energy_per_Volume_log": energy_per_volume_log
            }
            
            # Make prediction
            with st.spinner("Making prediction..."):
                result = api.make_prediction(prediction_data)
            
            if result:
                # Store result in session state for display
                st.session_state['last_prediction'] = result
            else:
                st.error("Failed to make prediction. Please check API connection.")
    
    with col2:
        st.subheader("Prediction Result")
        
        # Display last prediction result
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            
            # Risk level styling
            risk_level = result['risk_level']
            if risk_level == 'Critical':
                alert_class = "alert-critical"
            elif risk_level == 'High':
                alert_class = "alert-high"
            else:
                alert_class = "metric-container"
            
            prediction_text = "🚨 ROCKBURST PREDICTED" if result['prediction'] == 1 else "✅ NO ROCKBURST"
            
            st.markdown(f'''
            <div class="{alert_class}">
                <h3>{prediction_text}</h3>
                <p><strong>Probability:</strong> {result['probability']:.3f}</p>
                <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.3f}</p>
                <p><strong>Model Version:</strong> {result['model_version']}</p>
                <p><strong>Prediction Time:</strong> {result['prediction_time']}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['probability'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Probability"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Use the form on the left to make a prediction")


def render_batch_analysis():
    """Render batch analysis section"""
    st.header("📦 Batch Analysis")
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df_upload = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(df_upload.head())
            
            # Required columns
            required_cols = [
                'seismic_activity', 'rock_strength', 'depth', 'stress_level',
                'water_pressure', 'temperature', 'ground_displacement', 
                'mining_activity', 'geological_conditions'
            ]
            
            if all(col in df_upload.columns for col in required_cols):
                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        # Convert to list of dicts
                        data_list = df_upload[required_cols].to_dict('records')
                        
                        # Make batch prediction
                        batch_result = api.make_batch_prediction(data_list)
                        
                        if batch_result:
                            st.success(f"✅ Processed {batch_result['total_predictions']} predictions")
                            
                            # Display summary
                            summary = batch_result['summary']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("High Risk", summary['total_high_risk'])
                            with col2:
                                st.metric("Low Risk", summary['total_low_risk'])
                            with col3:
                                st.metric("Avg Probability", f"{summary['average_probability']:.3f}")
                            with col4:
                                st.metric("Processing Time", f"{batch_result['processing_time_seconds']:.2f}s")
                            
                            # Risk distribution
                            risk_dist = summary['risk_distribution']
                            fig = px.bar(x=list(risk_dist.keys()), y=list(risk_dist.values()),
                                       title="Batch Prediction Risk Distribution",
                                       color=list(risk_dist.keys()),
                                       color_discrete_map={
                                           'Low': '#28a745',
                                           'Medium': '#ffc107', 
                                           'High': '#fd7e14',
                                           'Critical': '#dc3545'
                                       })
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to process batch predictions")
            else:
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format
        st.info("📋 Required CSV format:")
        sample_df = pd.DataFrame({
            'seismic_activity': [75.5, 45.2, 90.1],
            'rock_strength': [150.0, 180.0, 120.0],
            'depth': [1200.0, 800.0, 1800.0],
            'stress_level': [85.3, 65.2, 110.5],
            'water_pressure': [45.2, 25.8, 70.1],
            'temperature': [25.0, 15.0, 35.0],
            'ground_displacement': [12.5, 8.2, 25.3],
            'mining_activity': [60.0, 40.0, 85.0],
            'geological_conditions': [70.0, 80.0, 45.0]
        })
        st.dataframe(sample_df)


def main():
    """Main dashboard application"""
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Rockburst+Monitor", width=200)
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Dashboard Page",
        ["🏠 Overview", "🔮 Live Prediction", "📈 Analytics", "📦 Batch Analysis", "⚙️ Settings"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, REFRESH_INTERVAL)
        st.sidebar.info(f"Page will refresh every {refresh_interval} seconds")
        
        # Auto refresh using st.rerun() after sleep
        time.sleep(refresh_interval)
        st.rerun()
    
    # API connection status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 API Connection")
    health = api.get_health()
    if health:
        st.sidebar.success("✅ API Connected")
        st.sidebar.write(f"**Status**: {health['status']}")
        st.sidebar.write(f"**Uptime**: {health['uptime_seconds']/3600:.1f}h")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.write("Please check API server")
    
    # Render header
    render_header()
    
    # Render selected page
    if page == "🏠 Overview":
        render_system_status()
        st.markdown("---")
        render_alerts()
        st.markdown("---")
        render_prediction_trends()
        
    elif page == "🔮 Live Prediction":
        render_live_prediction()
        
    elif page == "📈 Analytics":
        render_model_performance()
        st.markdown("---")
        render_prediction_trends()
        
    elif page == "📦 Batch Analysis":
        render_batch_analysis()
        
    elif page == "⚙️ Settings":
        st.header("⚙️ Settings")
        
        st.subheader("🎛️ Alert Thresholds")
        new_high_risk = st.slider("High Risk Threshold", 0.0, 1.0, HIGH_RISK_THRESHOLD)
        new_critical_risk = st.slider("Critical Risk Threshold", 0.0, 1.0, CRITICAL_RISK_THRESHOLD)
        
        st.subheader("🔗 API Configuration")
        new_api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved! (Note: Restart dashboard to apply)")
        
        st.subheader("📊 Dashboard Info")
        st.info(f"""
        **Dashboard Version**: 1.0.0  
        **API URL**: {API_BASE_URL}  
        **High Risk Threshold**: {HIGH_RISK_THRESHOLD}  
        **Critical Risk Threshold**: {CRITICAL_RISK_THRESHOLD}  
        **Refresh Interval**: {REFRESH_INTERVAL}s
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Rockburst Prediction Monitoring Dashboard** | Data Science Team GAMMA | 2025")


if __name__ == "__main__":
    main()
