#!/bin/bash
"""
Enhanced Rockburst Monitoring Dashboard Startup Script
======================================================
Start the enhanced Streamlit dashboard for monitoring rockburst predictions.
"""

set -e

echo "🛡️ Enhanced Rockburst Monitoring Dashboard"
echo "=========================================="

# Default configuration
PORT=8501
HOST="0.0.0.0"
APP_FILE="enhanced_app.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --basic)
            APP_FILE="app.py"
            echo "🔄 Using basic dashboard mode"
            shift
            ;;
        --help|-h)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to bind to (default: 8501)"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --basic        Use basic dashboard instead of enhanced"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Start enhanced dashboard on port 8501"
            echo "  $0 --port 8502       # Start on port 8502"
            echo "  $0 --basic           # Use basic dashboard"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Check if app file exists
if [[ ! -f "$APP_FILE" ]]; then
    echo "❌ Error: $APP_FILE not found"
    if [[ "$APP_FILE" == "enhanced_app.py" ]]; then
        echo "   Falling back to basic dashboard..."
        APP_FILE="app.py"
    fi
    
    if [[ ! -f "$APP_FILE" ]]; then
        echo "❌ Error: No dashboard files found"
        exit 1
    fi
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" && -d "../venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if required packages are installed
echo "📦 Checking dashboard dependencies..."
python -c "import streamlit, plotly, pandas" 2>/dev/null || {
    echo "❌ Missing required packages. Installing..."
    pip install streamlit plotly pandas altair
}

# Check for enhanced modules
if [[ "$APP_FILE" == "enhanced_app.py" ]]; then
    echo "🔍 Checking enhanced dashboard modules..."
    
    missing_modules=()
    
    for module in "styles.py" "alerts.py" "utils.py" "config.py"; do
        if [[ ! -f "$module" ]]; then
            missing_modules+=("$module")
        fi
    done
    
    if [[ ${#missing_modules[@]} -gt 0 ]]; then
        echo "⚠️  Warning: Missing enhanced modules: ${missing_modules[*]}"
        echo "   Dashboard will run in compatibility mode"
    else
        echo "✅ All enhanced modules found"
    fi
fi

# Check if API is running
echo "🔍 Checking API connection..."
API_URL="http://localhost:8000"

if curl -s --max-time 5 "$API_URL/health" > /dev/null 2>&1; then
    echo "✅ API server accessible at $API_URL"
else
    echo "⚠️  Warning: API server not responding at $API_URL"
    echo "   Dashboard will run in simulation mode"
fi

echo ""
echo "🚀 Starting Dashboard"
echo "===================="
echo "📱 App: $APP_FILE"
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo "📊 URL: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Export environment variables
export API_BASE_URL="$API_URL"
export DASHBOARD_MODE="enhanced"

# Start Streamlit with enhanced theme
streamlit run "$APP_FILE" \
    --server.address="$HOST" \
    --server.port="$PORT" \
    --server.headless=true \
    --server.runOnSave=true \
    --theme.primaryColor="#1e3a8a" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f8fafc" \
    --theme.textColor="#1f2937" \
    --theme.font="sans serif"
