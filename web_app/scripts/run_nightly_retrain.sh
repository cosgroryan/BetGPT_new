#!/bin/bash

# Nightly Model Retraining Wrapper Script
# This script is designed to be run by cron

# Set environment variables
export PATH="/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="/home/ubuntu/BetGPT_new/scripts:$PYTHONPATH"

# Change to the project directory
cd /home/ubuntu/BetGPT_new/web_app

# Log file for cron output
LOG_FILE="/var/log/betgpt_cron.log"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log "Starting nightly model retraining..."

# Run the retraining script
python3 scripts/nightly_retrain.py >> "$LOG_FILE" 2>&1

# Check if the retraining was successful
if [ $? -eq 0 ]; then
    log "Nightly retraining completed successfully"
    
    # Optional: Restart the web app to pick up new model
    # Uncomment the following lines if you want to restart the app
    # log "Restarting web application..."
    # docker compose restart betgpt-web
    # log "Web application restarted"
    
else
    log "Nightly retraining failed - check logs for details"
fi

log "Nightly retraining process finished"
