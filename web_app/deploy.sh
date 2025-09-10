#!/bin/bash

# BetGPT Web App Deployment Script

set -e  # Exit on any error

echo "🚀 Starting BetGPT Web App Deployment..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the web_app directory."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p backups

# Check if model artifacts exist
if [ ! -d "../artifacts" ]; then
    echo "⚠️  Warning: Model artifacts directory not found at ../artifacts"
    echo "   Make sure to run the model training script first."
fi

if [ ! -f "../five_year_dataset.parquet" ]; then
    echo "⚠️  Warning: Dataset file not found at ../five_year_dataset.parquet"
    echo "   Make sure the dataset file exists."
fi

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your production values before continuing."
    echo "   Especially change the SECRET_KEY!"
    read -p "Press Enter after editing .env file..."
fi

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test the application
echo "🧪 Testing application..."
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access your app at: http://localhost:8080"
else
    echo "❌ Application health check failed. Checking logs..."
    docker-compose logs web
    exit 1
fi

echo "📊 Deployment Summary:"
echo "   - Application URL: http://localhost:8080"
echo "   - Health Check: http://localhost:8080/health"
echo "   - Logs: docker-compose logs -f web"
echo "   - Stop: docker-compose down"

echo "🎉 Deployment completed successfully!"
