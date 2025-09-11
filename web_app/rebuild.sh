#!/bin/bash

echo "🔄 Rebuilding Docker containers with model path fixes..."

# Stop and remove existing containers
echo "📦 Stopping existing containers..."
docker compose down

# Rebuild and start containers
echo "🔨 Rebuilding containers..."
docker compose up -d --build

# Wait for containers to be ready
echo "⏳ Waiting for containers to be ready..."
sleep 10

# Check container status
echo "📊 Container status:"
docker compose ps

# Check logs
echo "📋 Recent logs:"
docker compose logs --tail=20

echo "✅ Rebuild complete! Check the logs above for any errors."
echo "🌐 Your app should be available at: http://localhost"
