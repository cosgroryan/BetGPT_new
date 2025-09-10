#!/bin/bash

# BetGPT Web App Startup Script
# This script sets up and starts the web application

echo "🐎 Starting BetGPT Web Application..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "web_app/app.py" ]; then
    echo "❌ Error: Please run this script from the BetGPT_new directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to find: web_app/app.py"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required model files exist
echo "🔍 Checking for required model files..."
if [ ! -f "artifacts/model_regression.pth" ]; then
    echo "⚠️  Warning: Model file not found at artifacts/model_regression.pth"
    echo "   The app will run but predictions may not work"
fi

if [ ! -f "artifacts/preprocess.pkl" ]; then
    echo "⚠️  Warning: Preprocessing file not found at artifacts/preprocess.pkl"
    echo "   The app will run but predictions may not work"
fi

# Install dependencies if requirements.txt exists
if [ -f "web_app/requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r web_app/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
else
    echo "⚠️  Warning: requirements.txt not found, skipping dependency installation"
fi

# Change to web_app directory
cd web_app

echo "🚀 Starting the web application..."
echo "   Access the app at: http://localhost:5000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the application
python3 run.py
