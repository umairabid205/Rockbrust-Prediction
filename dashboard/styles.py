"""
Enhanced Rockburst Monitoring Dashboard 
======================================
Professional monitoring dashboard with advanced styling and navigation.
"""

import streamlit as st

# Enhanced CSS styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e40af !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert containers */
    .alert-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.1);
        animation: pulse 2s infinite;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 6px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.1);
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 6px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Metric containers */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-offline {
        color: #ef4444;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 6px solid #10b981;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 6px solid #f59e0b;
    }
    
    /* Footer */
    .dashboard-footer {
        margin-top: 3rem;
        padding: 2rem;
        background: #f8fafc;
        border-radius: 12px;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .dashboard-title {
            font-size: 1.8rem;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .metric-container {
            padding: 1rem;
        }
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        border-radius: 10px;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the dashboard header"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🛡️ Rockburst Monitoring Dashboard</h1>
        <p class="dashboard-subtitle">
            Real-time monitoring and alerting system for rockburst prediction and safety management
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the dashboard footer"""
    st.markdown("""
    <div class="dashboard-footer">
        <p>🔒 Rockburst Monitoring System v2.0 | 
        Last updated: {timestamp} | 
        🛡️ Safety First - Mining Intelligence</p>
    </div>
    """.format(timestamp=st.session_state.get('last_update', 'Never')), unsafe_allow_html=True)

if __name__ == "__main__":
    # Apply custom CSS
    apply_custom_css()
    
    # Render header
    render_header()
    
    st.write("Enhanced dashboard styling applied! 🎨")
    
    # Test components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container"><h3>✅ System Status</h3><p class="status-online">All Systems Operational</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="alert-critical"><h4>🚨 Critical Alert</h4><p>High rockburst risk detected in Zone A</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="alert-high"><h4>⚠️ Warning Alert</h4><p>Elevated seismic activity</p></div>', unsafe_allow_html=True)
    
    # Render footer
    render_footer()
