#!/bin/bash

# BetGPT Web App Docker Build Script
# This script builds the Docker image for the BetGPT web application

set -e  # Exit on any error

echo "🐳 Building BetGPT Web App Docker Image..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t betgpt-webapp:latest .

echo "✅ Docker image built successfully!"
echo "📋 Image details:"
docker images betgpt-webapp:latest

echo ""
echo "🚀 To run the container:"
echo "   docker run -p 8080:8080 betgpt-webapp:latest"
echo ""
echo "🐳 To run with docker-compose:"
echo "   docker-compose up -d"
