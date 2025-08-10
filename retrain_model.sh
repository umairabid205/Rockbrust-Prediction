#!/bin/bash
"""
Automated Model Retraining Scheduler
====================================
This script can be used with cron to automatically retrain the model every 24 hours.

To set up automated retraining, add this to your crontab:
# Retrain rockburst model every day at 2 AM
0 2 * * * /path/to/this/script/retrain_model.sh

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

# Set the project directory
PROJECT_DIR="/Users/umair/Downloads/projects/project_2"
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set up logging
LOG_DIR="./logs/retraining"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/retraining_$(date +%Y%m%d_%H%M%S).log"

echo "🔄 Starting automated model retraining at $(date)" | tee -a "$LOG_FILE"

# Run the training script
python models/train_model.py --force-retrain 2>&1 | tee -a "$LOG_FILE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Model retraining completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "❌ Model retraining failed at $(date)" | tee -a "$LOG_FILE"
fi

# Optional: Send notification (uncomment if needed)
# echo "Rockburst model retrained at $(date)" | mail -s "Model Retraining Complete" admin@example.com

echo "📁 Log saved to: $LOG_FILE"
