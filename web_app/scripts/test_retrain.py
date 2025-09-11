#!/bin/bash

# Test script for the nightly retraining process
# Run this to test the retraining without waiting for cron

echo "Testing nightly model retraining..."

# Run the retraining script inside the Docker container
docker compose exec -T betgpt-web python scripts/nightly_retrain.py

# Check if the retraining was successful
if [ $? -eq 0 ]; then
    echo "✅ Test retraining completed successfully!"
else
    echo "❌ Test retraining failed!"
    exit 1
fi
