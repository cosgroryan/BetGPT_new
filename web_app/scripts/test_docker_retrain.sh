#!/bin/bash

# Simple test to verify Docker-based retraining works
# This tests just the data update part without full retraining

echo "Testing Docker-based retraining setup..."

# Change to the project directory
cd /home/ubuntu/BetGPT_new/web_app

# Test if we can run the data update script in Docker
echo "Testing data update script in Docker container..."
docker compose exec -T betgpt-web python scripts/web_app_scripts/update_data_retrain_model.py --help

if [ $? -eq 0 ]; then
    echo "✅ Docker retraining setup is working!"
    echo "You can now run the full test with: ./scripts/test_retrain.py"
else
    echo "❌ Docker retraining setup failed!"
    echo "Check that the Docker container is running and has the required dependencies"
fi
