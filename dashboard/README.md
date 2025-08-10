# Enhanced Rockburst Monitoring Dashboard 🛡️

Professional real-time monitoring dashboard for rockburst prediction and safety management in mining operations.

## 🌟 Features

### Core Functionality

- **Real-time Monitoring**: Live prediction tracking with automatic updates
- **Advanced Alerts**: Intelligent alert system with multiple severity levels
- **Interactive Predictions**: Manual input, random generation, and batch processing
- **Performance Analytics**: Model performance monitoring and trend analysis
- **System Health**: Comprehensive system status and diagnostics

### Enhanced Components

- **Professional Styling**: Modern, responsive design with custom CSS
- **Modular Architecture**: Separated concerns with dedicated modules
- **Alert Management**: Advanced alerting with acknowledgment and resolution
- **Data Visualization**: Rich charts and graphs using Plotly
- **API Integration**: Seamless connection with prediction API
- **Export Capabilities**: Data export and reporting features

## 📁 Dashboard Structure

```
dashboard/
├── enhanced_app.py          # Main enhanced dashboard application
├── app.py                   # Original dashboard (fallback)
├── styles.py                # Custom CSS styling and themes
├── alerts.py                # Alert management system
├── utils.py                 # Utility functions and API client
├── config.py                # Dashboard configuration
├── start_enhanced.sh        # Enhanced startup script
├── start_dashboard.sh       # Original startup script
└── README.md               # This documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Streamlit
- Required packages (automatically installed)
- Running prediction API (optional for simulation mode)

### Installation & Setup

1. **Navigate to dashboard directory:**

   ```bash
   cd dashboard
   ```

2. **Start the enhanced dashboard:**

   ```bash
   ./start_enhanced.sh
   ```

3. **Access the dashboard:**
   - Open browser to `http://localhost:8501`
   - Dashboard will auto-detect API availability

### Alternative Start Methods

```bash
# Custom port
./start_enhanced.sh --port 8502

# Custom host
./start_enhanced.sh --host localhost

# Use basic dashboard
./start_enhanced.sh --basic

# Direct Streamlit command
streamlit run enhanced_app.py --server.port=8501
```

## 🎯 Dashboard Tabs

### 🔍 Real-time Monitor

- **System Overview**: Current status, risk levels, and alerts
- **Live Metrics**: Real-time risk probability monitoring
- **Time Series**: Historical risk trend visualization
- **Recent Alerts**: Latest system alerts and notifications

### 📊 Analytics

- **Risk Distribution**: Pie chart of risk level distribution
- **Feature Importance**: Key factors contributing to risk
- **Prediction Trends**: Long-term prediction patterns
- **Statistical Analysis**: Comprehensive data insights

### 🚨 Alerts

- **Alert Dashboard**: All alerts with filtering and management
- **Alert Actions**: Acknowledge and resolve alerts
- **Alert History**: Historical alert tracking
- **Test Alerts**: Generate test alerts for validation

### 🎯 Live Predictions

- **Manual Input**: Custom geological parameter input
- **Random Generation**: Automated test data generation
- **Batch Processing**: Multiple prediction processing
- **File Upload**: CSV file batch predictions

### 📈 Performance

- **Model Metrics**: Accuracy, precision, recall tracking
- **Performance Trends**: Model performance over time
- **Prediction Volume**: System usage statistics
- **Model Information**: Current model details

### ⚙️ System Status

- **Health Checks**: API and system status monitoring
- **Environment Info**: System configuration details
- **Module Status**: Dashboard component verification
- **Diagnostic Tools**: Troubleshooting information

## 🔧 Configuration

### Environment Variables

```bash
export API_BASE_URL="http://localhost:8000"  # Prediction API endpoint
export DASHBOARD_PORT="8501"                 # Dashboard port
export DASHBOARD_HOST="0.0.0.0"             # Dashboard host
```

### Dashboard Settings

- **Risk Thresholds**: Configurable via sidebar
- **Refresh Rate**: Auto-refresh interval setting
- **Alert Levels**: Customizable alert severity levels
- **Theme**: Professional blue theme with custom styling

## 📊 Alert System

### Alert Levels

- 🚨 **Critical**: Immediate action required (probability ≥ 0.9)
- ⚠️ **High**: Heightened vigilance (probability ≥ 0.7)
- 📋 **Medium**: Enhanced monitoring (probability ≥ 0.5)
- ℹ️ **Low**: Information only (probability < 0.5)

### Alert Types

- **Prediction Alerts**: High-risk rockburst predictions
- **System Alerts**: API and system issues
- **Model Alerts**: Model performance issues

### Alert Actions

- **Acknowledge**: Mark alert as reviewed
- **Resolve**: Mark alert as handled
- **Auto-resolve**: Time-based resolution

## 🎨 Styling & Themes

### Custom CSS Features

- **Modern Design**: Professional blue color scheme
- **Responsive Layout**: Mobile-friendly responsive design
- **Interactive Elements**: Hover effects and transitions
- **Status Indicators**: Color-coded status displays
- **Alert Styling**: Severity-based alert appearance

### Theme Configuration

```python
--theme.primaryColor="#1e3a8a"              # Primary blue
--theme.backgroundColor="#ffffff"            # White background
--theme.secondaryBackgroundColor="#f8fafc"   # Light gray
--theme.textColor="#1f2937"                 # Dark text
```

## 🔌 API Integration

### Required Endpoints

- `GET /health` - System health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/metrics` - Model performance metrics

### API Client Features

- **Automatic Failover**: Graceful degradation without API
- **Error Handling**: Comprehensive error management
- **Timeout Control**: Configurable request timeouts
- **Response Caching**: Session-based data caching

## 🛠️ Troubleshooting

### Common Issues

**Dashboard won't start:**

```bash
# Check Python and dependencies
python3 --version
pip install streamlit plotly pandas altair

# Check file permissions
chmod +x start_enhanced.sh
```

**API connection issues:**

```bash
# Test API connectivity
curl http://localhost:8000/health

# Start API server
cd .. && python app.py
```

**Missing modules:**

```bash
# Ensure all files exist
ls -la dashboard/
# Files should include: enhanced_app.py, styles.py, alerts.py, utils.py, config.py
```

**Performance issues:**

- Check available memory
- Reduce auto-refresh frequency
- Use basic dashboard mode: `./start_enhanced.sh --basic`

### Debug Mode

```bash
# Enable Streamlit debug mode
streamlit run enhanced_app.py --logger.level=debug
```

## 📈 Performance Optimization

### Best Practices

- **Session Caching**: Utilizes Streamlit session state for data persistence
- **Lazy Loading**: Components load on demand
- **API Caching**: Intelligent caching of API responses
- **Error Boundaries**: Graceful error handling throughout

### Memory Management

- Automatic cleanup of old prediction history
- Configurable data retention periods
- Efficient chart rendering with Plotly

## 🔒 Security Considerations

### Data Protection

- No sensitive data stored locally
- Session-based data isolation
- Secure API communication

### Access Control

- Host-based access restrictions
- Port configuration flexibility
- Environment-based configuration

## 📝 Development

### Adding New Features

1. **Modular Design**: Add new modules to separate files
2. **Style Integration**: Use existing CSS classes
3. **Error Handling**: Implement comprehensive error management
4. **Testing**: Test with and without API availability

### Code Structure

```python
# Main dashboard class
class EnhancedDashboard:
    def __init__(self):
        # Initialize components

    def run(self):
        # Main dashboard logic

    def render_tab_name(self):
        # Individual tab implementations
```

## 📞 Support

### Getting Help

1. Check this README for common solutions
2. Review error messages in terminal
3. Test API connectivity independently
4. Use basic dashboard mode as fallback

### File Issues

If encountering persistent issues:

1. Verify all dashboard files exist
2. Check file permissions
3. Ensure Python environment is correct
4. Test with minimal configuration

## 🎉 Features in Development

### Upcoming Enhancements

- **Email Alerts**: Automatic email notifications for critical alerts
- **Data Export**: Enhanced CSV/Excel export capabilities
- **User Management**: Multi-user dashboard access
- **Mobile App**: Companion mobile application
- **Advanced Analytics**: Machine learning insights

### Version History

- **v2.0**: Enhanced dashboard with modular architecture
- **v1.5**: Professional styling and alert system
- **v1.0**: Basic dashboard functionality

---

## 📋 Quick Reference

### Essential Commands

```bash
./start_enhanced.sh              # Start enhanced dashboard
./start_enhanced.sh --port 8502  # Custom port
./start_enhanced.sh --basic      # Basic mode
streamlit run enhanced_app.py    # Direct start
```

### Key URLs

- **Dashboard**: http://localhost:8501
- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

### File Locations

- **Main App**: `enhanced_app.py`
- **Styles**: `styles.py`
- **Alerts**: `alerts.py`
- **Config**: `config.py`

🛡️ **Stay Safe - Monitor Smart** 🛡️
