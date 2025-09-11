#!/bin/bash

# BetGPT Web App Rebuild Script
# Includes all fixes for ports, proxies, and model paths

set -e  # Exit on any error

echo "🔄 Rebuilding Docker containers with all fixes..."

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

# Stop and remove existing containers
echo "📦 Stopping existing containers..."
docker compose down

# Remove any orphaned containers
echo "🧹 Cleaning up orphaned containers..."
docker compose down --remove-orphans

# Rebuild and start containers
echo "🔨 Rebuilding containers with all fixes..."
echo "   - Model path fixes (absolute paths)"
echo "   - Python import path fixes (Docker vs local)"
echo "   - Port 8080 configuration"
echo "   - Nginx proxy configuration"
echo "   - Static file serving fixes"
docker compose up -d --build

# Wait for containers to be ready
echo "⏳ Waiting for containers to be ready..."
sleep 15

# Check container status
echo "📊 Container status:"
docker compose ps

# Test the application through Nginx proxy
echo "🧪 Testing application through Nginx proxy..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully through Nginx!"
    echo "🌐 Access your app at: http://localhost"
else
    echo "❌ Application health check failed. Checking logs..."
    echo "📋 Flask app logs:"
    docker compose logs --tail=10 betgpt-web
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

# Check logs for any errors
echo "📋 Recent logs (last 20 lines):"
docker compose logs --tail=20

echo ""
echo "✅ Rebuild complete with all fixes applied!"
echo "📊 Summary:"
echo "   - Application URL: http://localhost"
echo "   - Health Check: http://localhost/health"
echo "   - Model paths: Fixed to use absolute paths"
echo "   - Python imports: Fixed for Docker environment"
echo "   - Port configuration: Flask on 8080, Nginx on 80"
echo "   - Static files: Proxied through Nginx"
echo "   - Logs: docker compose logs -f betgpt-web"
echo "   - Stop: docker compose down"
