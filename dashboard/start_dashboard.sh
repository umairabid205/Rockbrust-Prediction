#!/bin/bash
"""
Rockburst Monitoring Dashboard Startup Script
=============================================
Start the Streamlit dashboard for monitoring rockburst predictions.

Usage:
    ./start_dashboard.sh            # Default port 8501
    ./start_dashboard.sh --port 8502   # Custom port
"""

set -e

# Default configuration
PORT=8501
HOST="0.0.0.0"

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
        --help|-h)
            echo "Rockburst Monitoring Dashboard Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to bind to (default: 8501)"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Start on default port 8501"
            echo "  $0 --port 8502       # Start on port 8502"
            echo "  $0 --host localhost  # Bind to localhost only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" && -d "../venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if required packages are installed
echo "📦 Checking dashboard dependencies..."
python -c "import streamlit, plotly, pandas" 2>/dev/null || {
    echo "❌ Missing required packages. Installing..."
    pip install streamlit plotly pandas
}

# Check if API is running
echo "🔍 Checking API connection..."
curl -s http://localhost:8000/health > /dev/null || {
    echo "⚠️  Warning: API server not responding at http://localhost:8000"
    echo "   Please ensure the API is running before using the dashboard"
    echo "   Start the API with: python ../app.py --port 8000"
}

echo ""
echo "🚀 Starting Rockburst Monitoring Dashboard"
echo "=========================================="
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo "📊 Dashboard URL: http://$HOST:$PORT"
echo ""

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Start Streamlit
streamlit run app.py \\
    --server.address="$HOST" \\
    --server.port="$PORT" \\
    --server.headless=true \\
    --server.runOnSave=true \\
    --theme.primaryColor="#1f77b4" \\
    --theme.backgroundColor="#ffffff" \\
    --theme.secondaryBackgroundColor="#f0f2f6"
