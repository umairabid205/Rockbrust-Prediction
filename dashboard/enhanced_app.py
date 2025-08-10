#!/usr/bin/env python3
"""
Enhanced Rockburst Monitoring Dashboard
=======================================
Professional real-time monitoring dashboard for rockburst prediction and safety management.
"""

import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Rockburst Monitor Pro", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
from pathlib import Path

# Add dashboard modules to path
dashboard_dir = Path(__file__).parent
sys.path.insert(0, str(dashboard_dir))

# Import custom modules
try:
    from styles import apply_custom_css, render_header, render_footer
    from alerts import AlertManager, AlertLevel, AlertType, render_alert_card, render_alert_summary
    from utils import APIClient, DataGenerator, ChartGenerator, format_number, get_risk_color, get_risk_level
    from config import DashboardConfig
    modules_loaded = True
except ImportError as e:
    st.warning(f"Some dashboard modules not found: {e}. Using basic functionality.")
    modules_loaded = False

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds

class EnhancedDashboard:
    """Enhanced dashboard with modular components"""
    
    def __init__(self):
        self.api_client = APIClient(API_BASE_URL) if modules_loaded else None
        self.alert_manager = AlertManager() if modules_loaded else None
        self.data_generator = DataGenerator() if modules_loaded else None
        self.chart_generator = ChartGenerator() if modules_loaded else None
        
        # Initialize session state
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def run(self):
        """Main dashboard entry point"""
        # Apply custom styling
        if modules_loaded:
            apply_custom_css()
            render_header()
        else:
            st.title("🛡️ Rockburst Monitoring Dashboard")
            st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content tabs
        tabs = st.tabs([
            "🔍 Real-time Monitor", 
            "📊 Analytics", 
            "🚨 Alerts", 
            "🎯 Live Predictions", 
            "📈 Performance",
            "⚙️ System Status"
        ])
        
        with tabs[0]:
            self.render_realtime_monitor()
        
        with tabs[1]:
            self.render_analytics()
        
        with tabs[2]:
            self.render_alerts()
        
        with tabs[3]:
            self.render_live_predictions()
        
        with tabs[4]:
            self.render_performance()
        
        with tabs[5]:
            self.render_system_status()
        
        # Footer
        if modules_loaded:
            render_footer()
    
    def render_sidebar(self):
        """Render dashboard sidebar"""
        st.sidebar.image("https://via.placeholder.com/300x100/1e3a8a/white?text=ROCKBURST+MONITOR", 
                        use_container_width=True)
        
        st.sidebar.markdown("### 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_rate = st.sidebar.selectbox(
                "Refresh Rate (seconds)",
                [10, 30, 60, 300],
                index=1
            )
            
            # Auto-refresh logic
            if st.sidebar.button("🔄 Refresh Now"):
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### 📊 Quick Stats")
        
        try:
            if self.api_client:
                health = self.api_client.health_check()
                if health.get('status') == 'healthy':
                    st.sidebar.success("✅ API Online")
                else:
                    st.sidebar.error("❌ API Offline")
            
            # Alert summary
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                st.sidebar.metric("🚨 Active Alerts", alert_stats['total'])
                st.sidebar.metric("⚠️ Critical", alert_stats['critical'])
                st.sidebar.metric("📋 High Risk", alert_stats['high'])
            else:
                st.sidebar.info("Alert system loading...")
                
        except Exception as e:
            st.sidebar.error(f"Error loading stats: {str(e)}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Settings")
        
        # Risk thresholds
        high_risk_threshold = st.sidebar.slider(
            "High Risk Threshold", 
            0.0, 1.0, 0.7, 0.1
        )
        critical_risk_threshold = st.sidebar.slider(
            "Critical Risk Threshold", 
            0.0, 1.0, 0.9, 0.1
        )
        
        # Store in session state
        st.session_state['high_risk_threshold'] = high_risk_threshold
        st.session_state['critical_risk_threshold'] = critical_risk_threshold
    
    def render_realtime_monitor(self):
        """Render real-time monitoring tab"""
        st.subheader("🔍 Real-time Rockburst Monitoring")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # System status metrics
        with col1:
            if self.api_client:
                try:
                    health = self.api_client.health_check()
                    if health.get('status') == 'healthy':
                        st.metric("🟢 System Status", "Online", delta="Healthy")
                    else:
                        st.metric("🔴 System Status", "Offline", delta="Check API")
                except:
                    st.metric("🔴 System Status", "Error", delta="Connection failed")
            else:
                st.metric("🟡 System Status", "Limited", delta="Modules loading")
        
        with col2:
            # Current risk level
            try:
                if self.data_generator:
                    features = self.data_generator.generate_realistic_features()
                    if self.api_client:
                        result = self.api_client.predict_single(features)
                        if 'error' not in result:
                            prob = result.get('probability', 0)
                            risk_level = get_risk_level(prob) if modules_loaded else "Unknown"
                            st.metric("⚠️ Current Risk", risk_level, 
                                    delta=f"{prob:.2%}" if prob > 0.5 else None)
                        else:
                            st.metric("⚠️ Current Risk", "Error", delta="API issue")
                    else:
                        st.metric("⚠️ Current Risk", "Simulated", delta="Demo mode")
                else:
                    st.metric("⚠️ Current Risk", "Loading...", delta=None)
            except Exception as e:
                st.metric("⚠️ Current Risk", "Error", delta=str(e)[:20])
        
        with col3:
            # Active alerts count
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                critical_count = alert_stats['critical']
                if critical_count > 0:
                    st.metric("🚨 Critical Alerts", critical_count, delta="Immediate action")
                else:
                    st.metric("✅ Critical Alerts", "0", delta="All clear")
            else:
                st.metric("🚨 Critical Alerts", "N/A", delta="Loading...")
        
        with col4:
            # Last update
            last_update = st.session_state.get('last_update', datetime.now())
            time_diff = datetime.now() - last_update
            minutes_ago = int(time_diff.total_seconds() / 60)
            st.metric("🕒 Last Update", f"{minutes_ago}m ago", 
                     delta="Real-time" if minutes_ago < 2 else None)
        
        # Real-time chart section
        st.markdown("### 📈 Real-time Risk Monitoring")
        
        # Generate mock time series data
        if len(st.session_state.prediction_history) < 20:
            # Initialize with some data
            for i in range(20):
                timestamp = datetime.now() - timedelta(minutes=20-i)
                prob = np.random.beta(2, 5)  # Bias toward lower probabilities
                st.session_state.prediction_history.append({
                    'timestamp': timestamp,
                    'probability': prob,
                    'risk_level': get_risk_level(prob) if modules_loaded else 'Unknown'
                })
        
        # Create time series chart
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        if self.chart_generator and not df_history.empty:
            fig = self.chart_generator.create_time_series_chart(
                df_history['timestamp'].tolist(),
                df_history['probability'].tolist(),
                "Real-time Risk Probability",
                "Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['probability'],
                mode='lines+markers',
                name='Risk Probability'
            ))
            fig.update_layout(title="Real-time Risk Probability", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Latest alerts
        st.markdown("### 🚨 Latest Alerts")
        if self.alert_manager:
            recent_alerts = self.alert_manager.get_active_alerts(hours=1)
            if recent_alerts:
                for alert in recent_alerts[:3]:  # Show latest 3
                    action = render_alert_card(alert)
                    if action:
                        if action['action'] == 'acknowledge':
                            self.alert_manager.acknowledge_alert(action['alert_id'])
                            st.rerun()
                        elif action['action'] == 'resolve':
                            self.alert_manager.resolve_alert(action['alert_id'])
                            st.rerun()
            else:
                st.info("✅ No recent alerts - All systems operating normally")
        else:
            st.info("🔄 Alert system initializing...")
    
    def render_analytics(self):
        """Render analytics tab"""
        st.subheader("📊 Prediction Analytics")
        
        # Generate sample analytics data
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            st.markdown("#### Risk Level Distribution (Last 24h)")
            
            # Generate sample risk distribution
            risk_levels = np.random.choice(
                ['Low', 'Medium', 'High', 'Critical'],
                size=100,
                p=[0.6, 0.25, 0.12, 0.03]
            )
            
            if self.chart_generator:
                fig = self.chart_generator.create_risk_distribution_chart(risk_levels.tolist())
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback pie chart
                risk_counts = pd.Series(risk_levels).value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance
            st.markdown("#### Key Risk Factors")
            
            # Sample feature importance
            features = {
                'Seismic Activity': 0.23,
                'Stress Level': 0.19,
                'Ground Displacement': 0.15,
                'Water Pressure': 0.12,
                'Depth': 0.10,
                'Temperature': 0.08,
                'Extraction Ratio': 0.07,
                'Support Density': 0.06
            }
            
            if self.chart_generator:
                fig = self.chart_generator.create_feature_importance_chart(
                    features, "Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback bar chart
                fig = px.bar(x=list(features.keys()), y=list(features.values()))
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction trends
        st.markdown("#### 📈 Prediction Trends")
        
        # Generate trend data
        dates = pd.date_range(start='2025-01-01', end='2025-08-10', freq='D')
        predictions_per_day = np.random.poisson(50, len(dates))
        high_risk_per_day = np.random.binomial(predictions_per_day, 0.15)
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Total Predictions': predictions_per_day,
            'High Risk Predictions': high_risk_per_day
        })
        
        fig = px.line(trend_df, x='Date', y=['Total Predictions', 'High Risk Predictions'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts(self):
        """Render alerts tab"""
        st.subheader("🚨 Alert Management")
        
        if not self.alert_manager:
            st.warning("Alert system not available in basic mode")
            return
        
        # Alert summary
        alert_stats = self.alert_manager.get_alert_statistics()
        render_alert_summary(alert_stats)
        
        st.markdown("---")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_resolved = st.checkbox("Show Resolved Alerts", value=False)
        
        with col2:
            time_filter = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last Week"],
                index=2
            )
        
        with col3:
            level_filter = st.multiselect(
                "Alert Levels",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium"]
            )
        
        # Map time filter to hours
        time_map = {
            "Last Hour": 1,
            "Last 4 Hours": 4, 
            "Last 24 Hours": 24,
            "Last Week": 168
        }
        hours = time_map[time_filter]
        
        # Get filtered alerts
        if show_resolved:
            alerts = [a for a in self.alert_manager.alerts 
                     if (datetime.now() - a.timestamp).total_seconds() / 3600 <= hours]
        else:
            alerts = self.alert_manager.get_active_alerts(hours)
        
        # Filter by level
        level_map = {
            "Critical": AlertLevel.CRITICAL,
            "High": AlertLevel.HIGH,
            "Medium": AlertLevel.MEDIUM,
            "Low": AlertLevel.LOW
        }
        
        filtered_levels = [level_map[level] for level in level_filter]
        alerts = [a for a in alerts if a.level in filtered_levels]
        
        # Display alerts
        if alerts:
            st.markdown(f"### Found {len(alerts)} alerts")
            
            for alert in sorted(alerts, key=lambda x: x.timestamp, reverse=True):
                action = render_alert_card(alert, show_actions=not alert.resolved)
                if action:
                    if action['action'] == 'acknowledge':
                        self.alert_manager.acknowledge_alert(action['alert_id'])
                        st.success("Alert acknowledged!")
                        st.rerun()
                    elif action['action'] == 'resolve':
                        self.alert_manager.resolve_alert(action['alert_id'])
                        st.success("Alert resolved!")
                        st.rerun()
        else:
            st.info("No alerts found for the selected criteria")
        
        # Test alert button
        st.markdown("---")
        if st.button("🧪 Generate Test Alert"):
            if self.data_generator:
                features = self.data_generator.generate_high_risk_scenario()
                prediction = {
                    'probability': np.random.uniform(0.8, 0.95),
                    'risk_level': 'Critical',
                    'confidence': np.random.uniform(0.7, 0.9),
                    'model_version': '1.0'
                }
                
                alert = self.alert_manager.create_prediction_alert(prediction, features)
                if alert:
                    st.success("Test alert generated!")
                    st.rerun()
    
    def render_live_predictions(self):
        """Render live predictions tab"""
        st.subheader("🎯 Live Prediction Interface")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual Input", "Generate Random", "Load from File"],
            horizontal=True
        )
        
        if input_method == "Manual Input":
            self.render_manual_input()
        elif input_method == "Generate Random":
            self.render_random_generation()
        else:
            self.render_file_upload()
    
    def render_manual_input(self):
        """Render manual input interface"""
        st.markdown("### 📝 Manual Feature Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Geological Parameters")
            depth = st.number_input("Depth (m)", 0.0, 5000.0, 500.0)
            stress_level = st.number_input("Stress Level (MPa)", 0.0, 500.0, 100.0)
            seismic_activity = st.number_input("Seismic Activity", 0.0, 100.0, 30.0)
            water_pressure = st.number_input("Water Pressure (MPa)", 0.0, 200.0, 20.0)
            temperature = st.number_input("Temperature (°C)", -10.0, 100.0, 25.0)
            ground_displacement = st.number_input("Ground Displacement (mm)", 0.0, 200.0, 10.0)
            elastic_modulus = st.number_input("Elastic Modulus (MPa)", 1000.0, 100000.0, 30000.0)
        
        with col2:
            st.markdown("#### Rock & Mining Parameters")
            poisson_ratio = st.number_input("Poisson Ratio", 0.05, 0.5, 0.25)
            ucs = st.number_input("UCS (MPa)", 1.0, 500.0, 100.0)
            density = st.number_input("Density (kg/m³)", 1000.0, 5000.0, 2500.0)
            extraction_ratio = st.number_input("Extraction Ratio", 0.0, 1.0, 0.6)
            support_density = st.number_input("Support Density (/m²)", 0.0, 10.0, 1.0)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
            ventilation_rate = st.number_input("Ventilation Rate (m³/s)", 0.0, 20.0, 2.0)
        
        features = {
            'depth': depth,
            'stress_level': stress_level,
            'seismic_activity': seismic_activity,
            'water_pressure': water_pressure,
            'temperature': temperature,
            'ground_displacement': ground_displacement,
            'elastic_modulus': elastic_modulus,
            'poisson_ratio': poisson_ratio,
            'ucs': ucs,
            'density': density,
            'extraction_ratio': extraction_ratio,
            'support_density': support_density,
            'humidity': humidity,
            'ventilation_rate': ventilation_rate
        }
        
        if st.button("🔮 Make Prediction", type="primary"):
            self.make_prediction(features)
    
    def render_random_generation(self):
        """Render random generation interface"""
        st.markdown("### 🎲 Random Data Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario = st.selectbox(
                "Scenario Type",
                ["Normal Operation", "High Risk", "Critical Risk", "Random"]
            )
            
            num_predictions = st.slider("Number of Predictions", 1, 50, 5)
        
        with col2:
            if st.button("🎲 Generate & Predict", type="primary"):
                if self.data_generator:
                    predictions = []
                    
                    for i in range(num_predictions):
                        if scenario == "High Risk":
                            features = self.data_generator.generate_high_risk_scenario()
                        elif scenario == "Critical Risk":
                            features = self.data_generator.generate_high_risk_scenario()
                            # Make it even more critical
                            features['stress_level'] = min(features['stress_level'] * 1.2, 200)
                            features['seismic_activity'] = min(features['seismic_activity'] * 1.3, 100)
                        else:
                            features = self.data_generator.generate_realistic_features()
                        
                        predictions.append(features)
                    
                    self.make_batch_predictions(predictions)
                else:
                    st.error("Data generator not available")
    
    def render_file_upload(self):
        """Render file upload interface"""
        st.markdown("### 📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with geological features",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} records")
                st.dataframe(df.head())
                
                if st.button("🔮 Make Batch Predictions", type="primary"):
                    predictions = df.to_dict('records')
                    self.make_batch_predictions(predictions)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    def make_prediction(self, features: Dict[str, float]):
        """Make single prediction"""
        try:
            if self.api_client:
                with st.spinner("Making prediction..."):
                    result = self.api_client.predict_single(features)
                
                if 'error' in result:
                    st.error(f"Prediction failed: {result['error']}")
                    return
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    prob = result.get('probability', 0)
                    risk_level = get_risk_level(prob) if modules_loaded else result.get('risk_level', 'Unknown')
                    
                    # Create gauge chart
                    if self.chart_generator:
                        fig = self.chart_generator.create_gauge_chart(
                            prob, f"Risk Probability: {risk_level}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.metric("Risk Probability", f"{prob:.1%}")
                        st.metric("Risk Level", risk_level)
                
                with col2:
                    st.markdown("#### Prediction Details")
                    st.json(result)
                
                # Generate alert if high risk
                if self.alert_manager and prob >= st.session_state.get('high_risk_threshold', 0.7):
                    alert = self.alert_manager.create_prediction_alert(result, features)
                    if alert:
                        st.warning("🚨 High risk prediction - Alert generated!")
            else:
                st.info("API client not available - using simulation")
                prob = np.random.beta(2, 5)
                risk_level = get_risk_level(prob) if modules_loaded else "Simulated"
                st.metric("Simulated Risk Probability", f"{prob:.1%}")
                st.metric("Risk Level", risk_level)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    def make_batch_predictions(self, predictions_list: List[Dict[str, float]]):
        """Make batch predictions"""
        try:
            if self.api_client:
                with st.spinner(f"Making {len(predictions_list)} predictions..."):
                    result = self.api_client.predict_batch(predictions_list)
                
                if 'error' in result:
                    st.error(f"Batch prediction failed: {result['error']}")
                    return
                
                # Process results
                results_df = pd.DataFrame(result.get('predictions', []))
                
                if not results_df.empty:
                    st.success(f"✅ Successfully processed {len(results_df)} predictions")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_prob = results_df['probability'].mean()
                        st.metric("Average Risk", f"{avg_prob:.1%}")
                    
                    with col2:
                        high_risk = (results_df['probability'] >= 0.7).sum()
                        st.metric("High Risk", high_risk)
                    
                    with col3:
                        critical_risk = (results_df['probability'] >= 0.9).sum()
                        st.metric("Critical Risk", critical_risk)
                    
                    with col4:
                        max_prob = results_df['probability'].max()
                        st.metric("Max Risk", f"{max_prob:.1%}")
                    
                    # Results table
                    st.markdown("#### Prediction Results")
                    
                    # Add risk level column
                    if modules_loaded:
                        results_df['risk_level'] = results_df['probability'].apply(get_risk_level)
                        results_df['risk_color'] = results_df['probability'].apply(get_risk_color)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Generate alerts for high risk predictions
                    if self.alert_manager:
                        high_risk_predictions = results_df[
                            results_df['probability'] >= st.session_state.get('high_risk_threshold', 0.7)
                        ]
                        
                        if not high_risk_predictions.empty:
                            for idx, row in high_risk_predictions.iterrows():
                                features = predictions_list[idx] if idx < len(predictions_list) else {}
                                alert = self.alert_manager.create_prediction_alert(
                                    row.to_dict(), features
                                )
                            
                            st.warning(f"🚨 {len(high_risk_predictions)} high-risk predictions - Alerts generated!")
                
            else:
                st.info("API client not available - using simulation")
                # Simulate results
                simulated_results = []
                for _ in predictions_list:
                    prob = np.random.beta(2, 5)
                    simulated_results.append({
                        'probability': prob,
                        'risk_level': get_risk_level(prob) if modules_loaded else 'Simulated'
                    })
                
                results_df = pd.DataFrame(simulated_results)
                st.dataframe(results_df)
                
        except Exception as e:
            st.error(f"Batch prediction error: {str(e)}")
    
    def render_performance(self):
        """Render performance monitoring tab"""
        st.subheader("📈 Model Performance Monitoring")
        
        # Model info
        if self.api_client:
            try:
                with st.spinner("Loading model information..."):
                    model_info = self.api_client.get_model_info()
                
                if 'error' not in model_info:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Model Version", model_info.get('version', 'N/A'))
                    
                    with col2:
                        accuracy = model_info.get('accuracy', 0)
                        st.metric("Accuracy", f"{accuracy:.1%}" if accuracy else 'N/A')
                    
                    with col3:
                        features_count = model_info.get('feature_count', 0)
                        st.metric("Features", features_count if features_count else 'N/A')
                    
                    with col4:
                        last_trained = model_info.get('last_trained', 'N/A')
                        st.metric("Last Trained", last_trained)
                    
                    # Model details
                    if st.expander("📋 Model Details", expanded=False):
                        st.json(model_info)
                
            except Exception as e:
                st.error(f"Could not load model info: {str(e)}")
        else:
            st.info("Model performance monitoring not available in basic mode")
        
        # Performance metrics over time
        st.markdown("### 📊 Performance Metrics")
        
        # Generate sample performance data
        dates = pd.date_range(start='2025-08-01', end='2025-08-10', freq='D')
        accuracy = 0.95 + np.random.normal(0, 0.02, len(dates))
        precision = 0.92 + np.random.normal(0, 0.025, len(dates))
        recall = 0.88 + np.random.normal(0, 0.03, len(dates))
        
        perf_df = pd.DataFrame({
            'Date': dates,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })
        
        # Performance chart
        fig = px.line(perf_df, x='Date', y=['Accuracy', 'Precision', 'Recall'])
        fig.update_layout(height=400, title="Model Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction volume
        st.markdown("### 📈 Prediction Volume")
        
        volume_dates = pd.date_range(start='2025-08-01', end='2025-08-10', freq='H')
        hourly_predictions = np.random.poisson(5, len(volume_dates))
        
        volume_df = pd.DataFrame({
            'Datetime': volume_dates,
            'Predictions': hourly_predictions
        })
        
        fig = px.bar(volume_df, x='Datetime', y='Predictions')
        fig.update_layout(height=300, title="Hourly Prediction Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_system_status(self):
        """Render system status tab"""
        st.subheader("⚙️ System Status & Health")
        
        # API Health Check
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔗 API Services")
            
            if self.api_client:
                try:
                    health = self.api_client.health_check()
                    if health.get('status') == 'healthy':
                        st.success("✅ Prediction API: Online")
                        st.metric("Response Time", f"{health.get('response_time', 0):.3f}s")
                    else:
                        st.error("❌ Prediction API: Offline")
                        st.text(f"Error: {health.get('error', 'Unknown')}")
                except Exception as e:
                    st.error(f"❌ API Connection Failed: {str(e)}")
            else:
                st.warning("🟡 API Client: Not Available")
        
        with col2:
            st.markdown("#### 🧠 Model Status")
            
            if self.api_client:
                try:
                    model_info = self.api_client.get_model_info()
                    if 'error' not in model_info:
                        st.success("✅ Model: Loaded")
                        st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                        
                        last_trained = model_info.get('last_trained', 'Unknown')
                        if last_trained != 'Unknown':
                            trained_date = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
                            hours_since = (datetime.now() - trained_date.replace(tzinfo=None)).total_seconds() / 3600
                            st.metric("Hours Since Training", f"{hours_since:.1f}h")
                    else:
                        st.error("❌ Model: Error")
                        st.text(f"Error: {model_info.get('error', 'Unknown')}")
                except Exception as e:
                    st.error(f"❌ Model Status: {str(e)}")
            else:
                st.warning("🟡 Model Status: Not Available")
        
        with col3:
            st.markdown("#### 🚨 Alert System")
            
            if self.alert_manager:
                st.success("✅ Alerts: Active")
                alert_stats = self.alert_manager.get_alert_statistics()
                st.metric("Active Alerts", alert_stats['total'])
                st.metric("Alert Storage", f"{len(self.alert_manager.alerts)} total")
            else:
                st.warning("🟡 Alert System: Not Available")
        
        # System Information
        st.markdown("---")
        st.markdown("#### 💻 System Information")
        
        sys_col1, sys_col2 = st.columns(2)
        
        with sys_col1:
            st.markdown("**Dashboard Status**")
            st.text(f"Streamlit Version: {st.__version__}")
            st.text(f"Python Version: {sys.version.split()[0]}")
            st.text(f"Dashboard Modules: {'✅ Loaded' if modules_loaded else '❌ Limited'}")
            st.text(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with sys_col2:
            st.markdown("**Configuration**")
            st.text(f"API Base URL: {API_BASE_URL}")
            st.text(f"Refresh Interval: {REFRESH_INTERVAL}s")
            st.text(f"High Risk Threshold: {st.session_state.get('high_risk_threshold', 0.7):.1%}")
            st.text(f"Critical Threshold: {st.session_state.get('critical_risk_threshold', 0.9):.1%}")
        
        # Environment check
        st.markdown("---")
        st.markdown("#### 🌍 Environment Check")
        
        env_checks = {
            "Dashboard Directory": os.path.exists(dashboard_dir),
            "Config File": os.path.exists(os.path.join(dashboard_dir, 'config.py')),
            "Styles Module": os.path.exists(os.path.join(dashboard_dir, 'styles.py')),
            "Utils Module": os.path.exists(os.path.join(dashboard_dir, 'utils.py')),
            "Alerts Module": os.path.exists(os.path.join(dashboard_dir, 'alerts.py'))
        }
        
        for check, status in env_checks.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.warning(f"⚠️ {check}: Missing")


# Main application
def main():
    """Main application entry point"""
    try:
        dashboard = EnhancedDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}")
        st.info("Running in fallback mode...")
        
        # Fallback basic dashboard
        st.title("🛡️ Rockburst Monitoring Dashboard (Basic Mode)")
        st.warning("Some features unavailable. Check system status for details.")
        
        # Basic functionality
        st.subheader("System Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Online")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Offline")


if __name__ == "__main__":
    main()
