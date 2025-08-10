#!/bin/bash
"""
Rockburst Prediction API Startup Script
=======================================
Start the FastAPI server with proper configuration for production or development.

Usage:
    ./start_api.sh                # Start with default settings
    ./start_api.sh --dev          # Start in development mode
    ./start_api.sh --prod         # Start in production mode
    ./start_api.sh --help         # Show help
"""

set -e

# Default configuration
HOST="0.0.0.0"
PORT=8000
WORKERS=4
RELOAD=false
DEBUG=false
LOG_LEVEL="info"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            RELOAD=true
            DEBUG=true
            WORKERS=1
            LOG_LEVEL="debug"
            echo "🛠️  Development mode enabled"
            shift
            ;;
        --prod|--production)
            RELOAD=false
            DEBUG=false
            WORKERS=4
            LOG_LEVEL="info"
            echo "🚀 Production mode enabled"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Rockburst Prediction API Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development    Enable development mode (reload, debug)"
            echo "  --prod, --production    Enable production mode"
            echo "  --host HOST            Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT            Port to bind to (default: 8000)"
            echo "  --workers N            Number of worker processes (default: 4)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dev               # Development with auto-reload"
            echo "  $0 --prod --port 8080  # Production on port 8080"
            echo "  $0 --host localhost    # Bind to localhost only"
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
if [[ -z "${VIRTUAL_ENV}" && -d "venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn, pandas, sklearn" 2>/dev/null || {
    echo "❌ Missing required packages. Installing..."
    pip install fastapi uvicorn pandas scikit-learn
}

# Check if model is available
if [[ ! -f "artifacts/models/rockburst_rf_model.pkl" ]]; then
    echo "⚠️  Model not found at artifacts/models/rockburst_rf_model.pkl"
    echo "🔄 Attempting to train model..."
    if [[ -f "models/train_model.py" ]]; then
        cd models && python train_model.py --force-retrain && cd ..
    else
        echo "❌ No training script found. Please ensure model is trained first."
        exit 1
    fi
fi

# Print startup information
echo ""
echo "🚀 Starting Rockburst Prediction API"
echo "=================================="
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo "👥 Workers: $WORKERS"
echo "🔄 Reload: $RELOAD"
echo "🐛 Debug: $DEBUG"
echo "📝 Log Level: $LOG_LEVEL"
echo ""
echo "🔗 API URL: http://$HOST:$PORT"
echo "📚 Documentation: http://$HOST:$PORT/docs"
echo "🔍 Health Check: http://$HOST:$PORT/health"
echo ""

# Create logs directory
mkdir -p logs

# Start the API server
if [[ "$RELOAD" == "true" ]]; then
    # Development mode with auto-reload
    echo "🛠️  Starting in development mode with auto-reload..."
    uvicorn app:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --reload-dir . \
        --reload-exclude "logs/*" \
        --reload-exclude "*.log"
else
    # Production mode with multiple workers
    echo "🚀 Starting in production mode..."
    uvicorn app:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --no-server-header
fi
