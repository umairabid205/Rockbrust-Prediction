"""
Alert Management System for Rockburst Monitoring
===============================================
Advanced alert system for high-risk rockburst predictions with notification capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    PREDICTION = "prediction"
    SYSTEM = "system"
    MODEL = "model"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

class AlertManager:
    """Manages alerts for the dashboard"""
    
    def __init__(self, config_path="alerts.json"):
        self.config_path = config_path
        self.alerts: List[Alert] = []
        self.load_alerts()
    
    def load_alerts(self):
        """Load alerts from storage"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    alert_data = json.load(f)
                    for alert_dict in alert_data:
                        alert = Alert(
                            id=alert_dict['id'],
                            timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                            alert_type=AlertType(alert_dict['alert_type']),
                            level=AlertLevel(alert_dict['level']),
                            title=alert_dict['title'],
                            message=alert_dict['message'],
                            data=alert_dict['data'],
                            acknowledged=alert_dict.get('acknowledged', False),
                            resolved=alert_dict.get('resolved', False)
                        )
                        self.alerts.append(alert)
        except Exception as e:
            st.warning(f"Could not load alerts: {str(e)}")
    
    def save_alerts(self):
        """Save alerts to storage"""
        try:
            alert_data = []
            for alert in self.alerts:
                alert_dict = {
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_type': alert.alert_type.value,
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'data': alert.data,
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                }
                alert_data.append(alert_dict)
            
            with open(self.config_path, 'w') as f:
                json.dump(alert_data, f, indent=2)
        except Exception as e:
            st.error(f"Could not save alerts: {str(e)}")
    
    def create_prediction_alert(self, prediction_result: Dict[str, Any], 
                              geological_data: Dict[str, Any]) -> Optional[Alert]:
        """Create alert from prediction result"""
        probability = prediction_result.get('probability', 0)
        risk_level = prediction_result.get('risk_level', 'Low').lower()
        
        # Determine alert level
        if probability >= 0.9:
            level = AlertLevel.CRITICAL
        elif probability >= 0.7:
            level = AlertLevel.HIGH
        elif probability >= 0.5:
            level = AlertLevel.MEDIUM
        else:
            return None  # No alert for low risk
        
        # Create alert
        alert_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(geological_data)) % 10000}"
        
        title = f"{level.value.upper()} Risk Alert"
        if level == AlertLevel.CRITICAL:
            title = "🚨 CRITICAL ROCKBURST RISK"
        elif level == AlertLevel.HIGH:
            title = "⚠️ HIGH ROCKBURST RISK"
        
        # Create detailed message
        message = self._create_prediction_message(prediction_result, geological_data)
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            alert_type=AlertType.PREDICTION,
            level=level,
            title=title,
            message=message,
            data={
                'prediction': prediction_result,
                'geological_data': geological_data
            }
        )
        
        self.alerts.append(alert)
        self.save_alerts()
        return alert
    
    def create_system_alert(self, title: str, message: str, 
                          level: AlertLevel = AlertLevel.MEDIUM,
                          data: Dict[str, Any] = None) -> Alert:
        """Create system alert"""
        alert_id = f"sys_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            alert_type=AlertType.SYSTEM,
            level=level,
            title=title,
            message=message,
            data=data or {}
        )
        
        self.alerts.append(alert)
        self.save_alerts()
        return alert
    
    def create_model_alert(self, title: str, message: str,
                          level: AlertLevel = AlertLevel.MEDIUM,
                          model_data: Dict[str, Any] = None) -> Alert:
        """Create model-related alert"""
        alert_id = f"mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            alert_type=AlertType.MODEL,
            level=level,
            title=title,
            message=message,
            data=model_data or {}
        )
        
        self.alerts.append(alert)
        self.save_alerts()
        return alert
    
    def _create_prediction_message(self, prediction_result: Dict[str, Any],
                                 geological_data: Dict[str, Any]) -> str:
        """Create detailed prediction alert message"""
        prob = prediction_result.get('probability', 0)
        confidence = prediction_result.get('confidence', 0)
        
        message = f"""
        🔍 **Prediction Details:**
        • Probability: {prob:.3f} ({prob*100:.1f}%)
        • Confidence: {confidence:.3f}
        • Risk Level: {prediction_result.get('risk_level', 'Unknown')}
        
        🌍 **Key Risk Factors:**
        """
        
        # Identify high-risk factors
        risk_factors = []
        if geological_data.get('seismic_activity', 0) > 70:
            risk_factors.append(f"High seismic activity: {geological_data['seismic_activity']:.1f}")
        if geological_data.get('stress_level', 0) > 100:
            risk_factors.append(f"High stress level: {geological_data['stress_level']:.1f} MPa")
        if geological_data.get('water_pressure', 0) > 60:
            risk_factors.append(f"High water pressure: {geological_data['water_pressure']:.1f} MPa")
        if geological_data.get('ground_displacement', 0) > 20:
            risk_factors.append(f"High ground displacement: {geological_data['ground_displacement']:.1f} mm")
        
        if risk_factors:
            for factor in risk_factors[:3]:  # Show top 3 factors
                message += f"• {factor}\\n"
        else:
            message += "• Multiple combined factors contributing to risk\\n"
        
        message += f"""
        ⏰ **Detected at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        🎯 **Model Version:** {prediction_result.get('model_version', 'Unknown')}
        
        **Recommended Actions:**
        """
        
        if prob >= 0.9:
            message += """
            🚨 **IMMEDIATE ACTION REQUIRED**
            • Evacuate personnel from high-risk areas
            • Halt mining operations in affected zones  
            • Activate emergency response protocols
            • Monitor seismic activity continuously
            """
        elif prob >= 0.7:
            message += """
            ⚠️ **HEIGHTENED VIGILANCE REQUIRED**
            • Increase monitoring frequency
            • Prepare evacuation procedures
            • Reduce workforce in affected areas
            • Review safety protocols
            """
        else:
            message += """
            📋 **ENHANCED MONITORING RECOMMENDED**
            • Continue regular monitoring
            • Review geological conditions
            • Assess trend patterns
            """
        
        return message.strip()
    
    def get_active_alerts(self, hours: int = 24) -> List[Alert]:
        """Get active alerts within time range"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts 
                if alert.timestamp >= cutoff_time and not alert.resolved]
    
    def get_critical_alerts(self, hours: int = 24) -> List[Alert]:
        """Get critical alerts within time range"""
        active_alerts = self.get_active_alerts(hours)
        return [alert for alert in active_alerts 
                if alert.level == AlertLevel.CRITICAL]
    
    def get_high_risk_alerts(self, hours: int = 24) -> List[Alert]:
        """Get high and critical alerts within time range"""
        active_alerts = self.get_active_alerts(hours)
        return [alert for alert in active_alerts 
                if alert.level in [AlertLevel.HIGH, AlertLevel.CRITICAL]]
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.save_alerts()
                break
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.save_alerts()
                break
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, int]:
        """Get alert statistics"""
        active_alerts = self.get_active_alerts(hours)
        
        stats = {
            'total': len(active_alerts),
            'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            'high': len([a for a in active_alerts if a.level == AlertLevel.HIGH]),
            'medium': len([a for a in active_alerts if a.level == AlertLevel.MEDIUM]),
            'prediction': len([a for a in active_alerts if a.alert_type == AlertType.PREDICTION]),
            'system': len([a for a in active_alerts if a.alert_type == AlertType.SYSTEM]),
            'model': len([a for a in active_alerts if a.alert_type == AlertType.MODEL]),
            'acknowledged': len([a for a in active_alerts if a.acknowledged]),
            'unacknowledged': len([a for a in active_alerts if not a.acknowledged])
        }
        
        return stats


def render_alert_card(alert: Alert, show_actions: bool = True):
    """Render an individual alert card"""
    
    # Determine styling based on alert level
    if alert.level == AlertLevel.CRITICAL:
        container_class = "alert-critical"
        icon = "🚨"
    elif alert.level == AlertLevel.HIGH:
        container_class = "alert-high" 
        icon = "⚠️"
    elif alert.level == AlertLevel.MEDIUM:
        container_class = "metric-container"
        icon = "📋"
    else:
        container_class = "metric-container"
        icon = "ℹ️"
    
    # Status indicators
    status_text = ""
    if alert.resolved:
        status_text = "✅ RESOLVED"
    elif alert.acknowledged:
        status_text = "👁️ ACKNOWLEDGED"
    else:
        status_text = "🔔 NEW"
    
    # Render card
    st.markdown(f'''
    <div class="{container_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h4>{icon} {alert.title}</h4>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Status:</strong> {status_text}</p>
                <p><strong>Type:</strong> {alert.alert_type.value.title()}</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.8em; opacity: 0.7;">{alert.level.value.upper()}</span>
            </div>
        </div>
        <div style="margin-top: 10px; white-space: pre-line;">
            {alert.message}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Action buttons
    if show_actions and not alert.resolved:
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if not alert.acknowledged:
                if st.button("👁️ Acknowledge", key=f"ack_{alert.id}"):
                    return {"action": "acknowledge", "alert_id": alert.id}
        
        with col2:
            if st.button("✅ Resolve", key=f"resolve_{alert.id}"):
                return {"action": "resolve", "alert_id": alert.id}
    
    return None


def render_alert_summary(alert_stats: Dict[str, int]):
    """Render alert summary metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚨 Critical Alerts", 
            alert_stats['critical'],
            delta=None,
            help="Immediate action required"
        )
    
    with col2:
        st.metric(
            "⚠️ High Risk Alerts", 
            alert_stats['high'],
            delta=None,
            help="Heightened vigilance required"
        )
    
    with col3:
        st.metric(
            "📋 Total Active", 
            alert_stats['total'],
            delta=None,
            help="All active alerts (24h)"
        )
    
    with col4:
        st.metric(
            "🔔 Unacknowledged", 
            alert_stats['unacknowledged'],
            delta=None,
            help="Alerts requiring attention"
        )
