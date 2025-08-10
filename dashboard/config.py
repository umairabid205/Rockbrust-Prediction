"""
Dashboard Configuration and Utilities
=====================================
Configuration settings and utility functions for the Rockburst Monitoring Dashboard.
"""

import os
from datetime import datetime, timedelta

# API Configuration
API_CONFIG = {
    'base_url': os.getenv('API_BASE_URL', 'http://localhost:8000'),
    'timeout': int(os.getenv('API_TIMEOUT', 10)),
    'retry_attempts': int(os.getenv('API_RETRY_ATTEMPTS', 3)),
    'retry_delay': float(os.getenv('API_RETRY_DELAY', 1.0))
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'title': 'Rockburst Monitoring Dashboard',
    'icon': '⛰️',
    'layout': 'wide',
    'refresh_interval': int(os.getenv('REFRESH_INTERVAL', 30)),
    'auto_refresh_default': True,
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6'
    }
}

# Alert Configuration
ALERT_CONFIG = {
    'high_risk_threshold': float(os.getenv('HIGH_RISK_THRESHOLD', 0.7)),
    'critical_risk_threshold': float(os.getenv('CRITICAL_RISK_THRESHOLD', 0.9)),
    'alert_history_hours': int(os.getenv('ALERT_HISTORY_HOURS', 24)),
    'max_alerts_display': int(os.getenv('MAX_ALERTS_DISPLAY', 10))
}

# Data Configuration
DATA_CONFIG = {
    'historical_days': int(os.getenv('HISTORICAL_DAYS', 30)),
    'samples_per_day': int(os.getenv('SAMPLES_PER_DAY', 24)),
    'cache_ttl': int(os.getenv('CACHE_TTL', 60)),  # seconds
    'batch_size_limit': int(os.getenv('BATCH_SIZE_LIMIT', 1000))
}

# Visualization Configuration
VIZ_CONFIG = {
    'color_scheme': {
        'low_risk': '#28a745',
        'medium_risk': '#ffc107',
        'high_risk': '#fd7e14',
        'critical_risk': '#dc3545',
        'healthy': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545'
    },
    'chart_height': int(os.getenv('CHART_HEIGHT', 400)),
    'gauge_height': int(os.getenv('GAUGE_HEIGHT', 300))
}

# Model Parameters Configuration
MODEL_PARAMS = {
    'seismic_activity': {'min': 0.0, 'max': 100.0, 'default': 50.0, 'step': 0.1},
    'rock_strength': {'min': 50.0, 'max': 300.0, 'default': 150.0, 'step': 1.0},
    'depth': {'min': 100.0, 'max': 3000.0, 'default': 1200.0, 'step': 10.0},
    'stress_level': {'min': 0.0, 'max': 200.0, 'default': 75.0, 'step': 0.1},
    'water_pressure': {'min': 0.0, 'max': 100.0, 'default': 40.0, 'step': 0.1},
    'temperature': {'min': -20.0, 'max': 60.0, 'default': 25.0, 'step': 0.1},
    'ground_displacement': {'min': 0.0, 'max': 50.0, 'default': 15.0, 'step': 0.1},
    'mining_activity': {'min': 0.0, 'max': 100.0, 'default': 60.0, 'step': 0.1},
    'geological_conditions': {'min': 0.0, 'max': 100.0, 'default': 70.0, 'step': 0.1}
}

def get_risk_color(risk_level):
    """Get color for risk level"""
    return VIZ_CONFIG['color_scheme'].get(f'{risk_level.lower()}_risk', '#666666')

def get_status_color(status):
    """Get color for status"""
    return VIZ_CONFIG['color_scheme'].get(status.lower(), '#666666')

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def get_time_range(hours=24):
    """Get time range for filtering data"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    return start_time, end_time

class DashboardConfig:
    """Dashboard configuration class"""
    
    def __init__(self):
        self.api = API_CONFIG
        self.dashboard = DASHBOARD_CONFIG
        self.alerts = ALERT_CONFIG
        self.data = DATA_CONFIG
        self.viz = VIZ_CONFIG
        self.model_params = MODEL_PARAMS
    
    @property
    def base_url(self):
        return self.api['base_url']
    
    @property 
    def title(self):
        return self.dashboard['title']
    
    @property
    def icon(self):
        return self.dashboard['icon']
    
    @property
    def layout(self):
        return self.dashboard['layout']
    
    @property
    def refresh_interval(self):
        return self.dashboard['refresh_interval']
