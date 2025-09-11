#!/bin/bash

# BetGPT Web App Clean Start Script
# This script starts fresh with all fixes applied

set -e  # Exit on any error

echo "🚀 Starting BetGPT Web App with Clean Configuration..."

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

# Check if docker compose is installed (Docker Compose v2)
if ! docker compose version &> /dev/null; then
    echo "❌ Error: Docker Compose v2 is not installed. Please install Docker Compose v2 first."
    exit 1
fi

# Verify scripts directory exists
if [ ! -d "scripts/web_app_scripts" ]; then
    echo "❌ Error: scripts/web_app_scripts directory not found."
    echo "   Please ensure the scripts are in the correct location."
    exit 1
fi

# Verify key script files exist
if [ ! -f "scripts/web_app_scripts/recommend_picks_NN.py" ]; then
    echo "❌ Error: recommend_picks_NN.py not found in scripts/web_app_scripts/"
    exit 1
fi

echo "✅ All prerequisites checked"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down --remove-orphans

# Remove any existing images to force rebuild
echo "🧹 Cleaning up existing images..."
docker compose down --rmi all --volumes --remove-orphans 2>/dev/null || true

# Build and start fresh
echo "🔨 Building fresh containers..."
docker compose build --no-cache

echo "🚀 Starting services..."
docker compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 20

# Check container status
echo "📊 Container status:"
docker compose ps

# Test the application
echo "🧪 Testing application..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access your app at: http://localhost"
else
    echo "❌ Application health check failed. Checking logs..."
    echo "📋 Flask app logs:"
    docker compose logs --tail=20 betgpt-web
    echo "📋 Nginx logs:"
    docker compose logs --tail=10 nginx
    exit 1
fi

# Test static files
echo "🧪 Testing static files..."
if curl -f http://localhost/static/css/style.css > /dev/null 2>&1; then
    echo "✅ Static files are being served correctly!"
else
    echo "⚠️  Static files test failed. Check Nginx configuration."
fi

# Test model imports
echo "🧪 Testing model imports..."
if docker compose exec betgpt-web python3 -c "
import sys
sys.path.append('/app/scripts/web_app_scripts')
from recommend_picks_NN import model_win_table
print('✅ Model imports successful!')
" 2>/dev/null; then
    echo "✅ Model imports are working correctly!"
else
    echo "⚠️  Model imports test failed. Check the logs above."
fi

echo ""
echo "🎉 Clean start completed successfully!"
echo "📊 Summary:"
echo "   - Application URL: http://localhost"
echo "   - Health Check: http://localhost/health"
echo "   - Scripts: Mounted from ./scripts/"
echo "   - Model paths: Fixed to use absolute paths"
echo "   - Port configuration: Flask on 8080, Nginx on 80"
echo "   - Static files: Proxied through Nginx"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker compose logs -f betgpt-web"
echo "   - Stop: docker compose down"
echo "   - Restart: docker compose restart"
