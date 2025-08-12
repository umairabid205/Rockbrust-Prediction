#!/usr/bin/env python3
"""
Simple Rockburst Prediction Dashboard
=====================================
Clean and professional dashboard for live rockburst predictions with InfluxDB data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta
import time

# Page configuration - MUST be first
st.set_page_config(
    page_title="Rockburst Prediction Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for professional blue and white theme with dark text
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        color: #1a1a1a;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    .main-title {
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid #4a90e2;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fbff 0%, #e6f2ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #b3d9ff;
        box-shadow: 0 2px 8px rgba(79, 144, 226, 0.1);
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    
    .metric-card h3 {
        color: #1f4e79;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .prediction-form {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #4a90e2;
        box-shadow: 0 4px 12px rgba(79, 144, 226, 0.15);
        margin: 1rem 0;
        color: #1a1a1a;
    }
    
    .prediction-form h3 {
        color: #1f4e79;
        font-weight: 600;
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-offline {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
        color: #006600;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #99ff99;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fff8e6 0%, #ffe6b3 100%);
        color: #cc7a00;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffcc66;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
        color: #cc0000;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ff9999;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .risk-intense {
        background: linear-gradient(135deg, #e6e6ff 0%, #ccccff 100%);
        color: #4d00cc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #9999ff;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .data-card {
        background: #f8fbff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #d1e7ff;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8fbff;
        border: 1px solid #b3d9ff;
        padding: 1rem;
        border-radius: 8px;
        color: #1a1a1a;
    }
    
    div[data-testid="metric-container"] label {
        color: #1f4e79 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #1a1a1a !important;
        font-weight: bold !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-color: #4a90e2;
        color: #1a1a1a;
    }
    
    .stSlider > div > div {
        color: #1f4e79;
    }
    
    .stSlider label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Success and Info Messages */
    .stSuccess {
        background-color: #e6ffe6 !important;
        color: #1a1a1a !important;
        border: 1px solid #99ff99 !important;
    }
    
    .stSuccess > div {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .stInfo {
        background-color: #e6f2ff !important;
        color: #1a1a1a !important;
        border: 1px solid #b3d9ff !important;
    }
    
    .stInfo > div {
        color: #1a1a1a !important;
    }
    
    /* Text elements */
    p, span, div {
        color: #1a1a1a !important;
    }
    
    /* Table styling */
    .stDataFrame {
        color: #1a1a1a !important;
    }
    
    .stDataFrame table {
        color: #1a1a1a !important;
    }
    
    .stDataFrame th {
        background-color: #f0f7ff !important;
        color: #1f4e79 !important;
        font-weight: bold !important;
    }
    
    .stDataFrame td {
        color: #1a1a1a !important;
    }
    
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1f4e79;
        transform: translateY(-1px);
    }
    
    .stMarkdown, .stMarkdown p, .stText {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f4e79 !important;
    }
    
    .stDataFrame {
        color: #1a1a1a;
    }
    
    .stDataFrame th {
        background-color: #4a90e2 !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .stDataFrame td {
        color: #1a1a1a !important;
    }
    
    .element-container, .stColumn {
        color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8fbff;
        color: #1a1a1a;
    }
    
    .stForm {
        color: #1a1a1a;
    }
    
    .stForm label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def make_prediction(data):
    """Make a prediction using the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
    return None

def get_influxdb_data():
    """Get recent predictions from InfluxDB"""
    try:
        from influxdb_client.client.influxdb_client import InfluxDBClient
        
        # InfluxDB connection configuration (match API settings)
        INFLUXDB_CONFIG = {
            "url": "http://localhost:8086",
            "token": "rockbrust-token-12345", 
            "org": "rockbrust",
            "bucket": "rockbrust-data"
        }
        
        # Connect to InfluxDB
        client = InfluxDBClient(
            url=INFLUXDB_CONFIG["url"],
            token=INFLUXDB_CONFIG["token"],
            org=INFLUXDB_CONFIG["org"]
        )
        
        query_api = client.query_api()
        
        # Query for recent predictions (last 24 hours) - separate queries for different data types
        # First get numeric fields (prediction, probability)
        numeric_query = f'''
        from(bucket: "{INFLUXDB_CONFIG["bucket"]}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "rockburst_predictions")
        |> filter(fn: (r) => r["_field"] == "prediction" or r["_field"] == "probability")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"], desc: true)
        '''
        
        # Then get string fields (risk_level)
        string_query = f'''
        from(bucket: "{INFLUXDB_CONFIG["bucket"]}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "rockburst_predictions")
        |> filter(fn: (r) => r["_field"] == "risk_level")
        |> sort(columns: ["_time"], desc: true)
        '''
        
        # Execute queries
        numeric_result = query_api.query(query=numeric_query)
        string_result = query_api.query(query=string_query)
        
        data = []
        
        # Process numeric data (prediction, probability)
        numeric_data = {}
        for table in numeric_result:
            for record in table.records:
                time_key = record.get_time()
                if time_key not in numeric_data:
                    numeric_data[time_key] = {'timestamp': time_key}
                numeric_data[time_key]['prediction'] = int(record.values.get('prediction', 0)) if record.values.get('prediction') is not None else None
                numeric_data[time_key]['probability'] = float(record.values.get('probability', 0.0)) if record.values.get('probability') is not None else None
        
        # Process string data (risk_level)
        string_data = {}
        for table in string_result:
            for record in table.records:
                time_key = record.get_time()
                string_data[time_key] = record.values.get('_value', 'Unknown')
        
        # Combine data
        for time_key in numeric_data:
            entry = numeric_data[time_key]
            entry['risk_level'] = string_data.get(time_key, 'Unknown')
            # Only add if we have all required fields
            if entry.get('prediction') is not None and entry.get('probability') is not None:
                data.append(entry)
        
        client.close()
        
        if data:
            df = pd.DataFrame(data)
            # Convert timestamp to local timezone for display
            # Robustly convert to local time (handles both naive and aware timestamps)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            return df.drop_duplicates().reset_index(drop=True)
        else:
            # If no data found, return empty DataFrame with correct structure
            return pd.DataFrame(columns=['timestamp', 'prediction', 'probability', 'risk_level'])
            
    except Exception as e:
        # Silently fallback to simulated data when InfluxDB is not available
        # st.warning(f"Could not connect to InfluxDB: {str(e)}. Showing simulated data.")
        # Fallback to simulated data
        np.random.seed(int(datetime.now().timestamp()) % 100)
        
        # Generate sample historical data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='1h')
        
        data = []
        for date in dates[-12:]:  # Last 12 hours
            prediction = np.random.choice([0, 1], p=[0.7, 0.3])
            probability = np.random.uniform(0.3, 0.9) if prediction == 1 else np.random.uniform(0.1, 0.6)
            if probability < 0.4:
                risk_level = "Low"
            elif probability < 0.55:
                risk_level = "Medium"
            elif probability < 0.75:
                risk_level = "High"
            else:
                risk_level = "Intense"
            
            data.append({
                'timestamp': date,
                'prediction': prediction,
                'probability': probability,
                'risk_level': risk_level
            })
        
        return pd.DataFrame(data)

# Main Dashboard
def main():
    # Title
    st.markdown('<h1 class="main-title">🛡️ Rockburst Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # System Status
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        health = get_api_health()
        if health:
            st.markdown('<div class="metric-card">',  unsafe_allow_html=True)
            st.markdown("### 🟢 System Status")
            st.markdown('<span class="status-healthy">Online & Ready</span>', 
                       unsafe_allow_html=True)
            st.markdown(f"**Model Loaded:** {'✅ Yes' if health.get('model_loaded') else '❌ No'}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">',  unsafe_allow_html=True)
            st.markdown("### 🔴 System Status")
            st.markdown('<span class="status-offline">Offline</span>', 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">',  unsafe_allow_html=True)
        st.markdown("### ⏰ Last Update")
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Time:** {current_time}")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Get recent data count
        recent_data = get_influxdb_data()
        st.markdown('<div class="metric-card">',  unsafe_allow_html=True)
        st.markdown("### 📊 Data Summary")
        st.markdown(f"**Predictions Today:** {len(recent_data)}")
        high_risk_count = len(recent_data[recent_data['risk_level'] == 'High'])
        st.markdown(f"**High Risk:** {high_risk_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Main Content
    col_left, col_right = st.columns([1, 1])
    
    # Left Column - Live Prediction
    with col_left:
        st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
        st.markdown("### 🔍 Live Prediction")
        
        with st.form("prediction_form", clear_on_submit=False):
            st.markdown("**Seismic Parameters**")
            
            # Input parameters
            duration_days = st.slider("Duration (days)", 0.1, 100.0, 15.5, 0.1)
            energy_unit_log = st.slider("Energy Unit (log)", 0.0, 10.0, 5.2, 0.1)
            energy_density = st.slider("Energy Density (Joule²)", 0.0, 10000.0, 450.0, 1.0)
            volume_m3_sqr = st.slider("Volume (m³²)", 0.0, 1000.0, 120.0, 1.0)
            event_freq_log = st.slider("Event Frequency (log/day)", 0.0, 10.0, 2.8, 0.1)
            energy_joule_day = st.slider("Energy per Day (Joule²)", 0.0, 10000.0, 890.0, 1.0)
            volume_per_day = st.slider("Volume per Day (m³)", 0.0, 2000.0, 350.0, 1.0)
            energy_per_volume_log = st.slider("Energy per Volume (log)", 0.0, 10.0, 3.4, 0.1)
            
            predict_button = st.form_submit_button("🚀 Make Prediction", use_container_width=True)
        
        if predict_button and health:
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
            
            with st.spinner("🔮 Making prediction..."):
                result = make_prediction(prediction_data)
                
                if result:
                    # Display results with proper 4-category risk classification
                    risk_level = result.get('risk_level', 'Unknown')
                    
                    # Map risk levels to CSS classes and display text
                    risk_mapping = {
                        'Low': ('risk-low', 'LOW RISK ✅'),
                        'Medium': ('risk-medium', 'MEDIUM RISK ⚠️'),
                        'High': ('risk-high', 'HIGH RISK ⚠️'),
                        'Intense': ('risk-intense', 'INTENSE RISK 🚨'),
                        'Critical': ('risk-intense', 'INTENSE RISK 🚨')  # fallback for old API responses
                    }
                    
                    risk_class, risk_text = risk_mapping.get(risk_level, ('risk-high', f'{risk_level.upper()} RISK ⚠️'))
                    
                    st.markdown(f'<div class="{risk_class}">{risk_text}</div>', 
                               unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probability", f"{result['probability']:.3f}")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.3f}")
                    with col3:
                        st.metric("Risk Level", result['risk_level'])
                    
                    st.success("✅ Prediction saved to database")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column - Data Visualization
    with col_right:
        st.markdown("### 📊 Recent Predictions (Simulated Data)")
        
        # Get and display recent data
        df = get_influxdb_data()
        
        if not df.empty:
            # Time series chart
            fig = go.Figure()
            
            # Add probability line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['probability'],
                mode='lines+markers',
                name='Risk Probability',
                line=dict(color='#4a90e2', width=3),
                marker=dict(size=6, color='#1f4e79')
            ))
            
            # Add high risk threshold line
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold",
                         annotation_font_color="#1a1a1a")
            
            fig.update_layout(
                title="Risk Probability Over Time",
                xaxis_title="Time",
                yaxis_title="Probability",
                plot_bgcolor='#f8fbff',
                paper_bgcolor='white',
                font=dict(color='#1a1a1a', size=12),
                title_font=dict(color='#1f4e79', size=16),
                xaxis=dict(
                    title_font=dict(color='#1a1a1a'),
                    tickfont=dict(color='#1a1a1a'),
                    gridcolor='#e6f2ff'
                ),
                yaxis=dict(
                    title_font=dict(color='#1a1a1a'),
                    tickfont=dict(color='#1a1a1a'),
                    gridcolor='#e6f2ff'
                ),
                showlegend=True,
                legend=dict(font=dict(color='#1a1a1a')),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk distribution pie chart
            risk_counts = df['risk_level'].value_counts()
            colors = {
                'Low': '#6bcf7f', 
                'Medium': '#ffd93d', 
                'High': '#ff6b6b',
                'Intense': '#9b59b6',
                'Critical': '#9b59b6'  # fallback for old data
            }
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=.3,
                marker_colors=[colors.get(label, '#4a90e2') for label in risk_counts.index]
            )])
            
            fig_pie.update_layout(
                title="Risk Level Distribution (Last 12 Hours)",
                font=dict(color='#1a1a1a', size=12),
                title_font=dict(color='#1f4e79', size=16),
                paper_bgcolor='white',
                plot_bgcolor='#f8fbff',
                showlegend=True,
                legend=dict(font=dict(color='#1a1a1a')),
                height=300
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Recent predictions table
            st.markdown("### 📋 Recent Predictions")
            display_df = df.tail(5).copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M')
            display_df['probability'] = display_df['probability'].round(3)
            display_df = display_df[['timestamp', 'prediction', 'probability', 'risk_level']]
            display_df.columns = ['Time', 'Prediction', 'Probability', 'Risk Level']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent prediction data available")
    
    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🛡️ Rockburst Prediction System | Powered by ML & Simulated Data for Demo"
        "</div>", 
        unsafe_allow_html=True
    )

# Auto-refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Auto refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
if current_time - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = current_time
    st.rerun()

if __name__ == "__main__":
    main()
